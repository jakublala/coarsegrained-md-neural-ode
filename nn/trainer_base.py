import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import time
import os
import shutil
import numpy as np
import wandb
import re
import copy
from collections import OrderedDict

from nn.config import Config
from data.dataset import Dataset
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint
from nn.activations import get_activation_functions
from diffmd.utils import *
from nn.losses import set_loss_func, final_mse_pos


class Trainer():

    def __init__(self, config):
        """
            Base class for all trainers

            Args:
                config (str or dict): string of path to config file or dictionary of config parameters

        """

        self.parallel = False
        self.early_stopping = False

        self.config = Config(config)
        
        if self.config.wandb and self.is_master():
            self.wandb = wandb.init(project=self.config.project, config=self.config)
            self.config.assign_folders(self.wandb.name)
            self.config.assign_sweep_config(self.wandb.config)
        else:
            self.config.assign_folders()
        if not self.config.analysis:
            self.config.save_config()
                
        self.device = set_device(self.config.device)
        self.dtype = set_dtype(self.config.dtype)
        self.activation_functions = get_activation_functions(self.config.activation_function, self.config.nn_widths)

        # dataset setup
        self.training_dataset = Dataset(self.config, dataset_type='train', traj_length=self.config.dataset_steps, dataset_fraction=self.config.training_fraction, random_dataset=self.config.random_dataset)
        self.test_dataset = Dataset(self.config, dataset_type='test', traj_length=self.config.eval_dataset_steps)
        self.validation_dataset = Dataset(self.config, dataset_type='validation', traj_length=self.config.eval_dataset_steps)
        self.training_dataloader = self.get_dataloader(self.training_dataset, shuffle=self.config.shuffle) 
        self.test_dataloader = self.get_dataloader(self.test_dataset) 
        self.validation_dataloader = self.get_dataloader(self.validation_dataset)
        assert len(self.training_dataset) > self.config.batch_size, "Batch size is too large for the dataset. Please reduce the batch size or increase the dataset size."

        # TODO: check that loading a folder works well
        if self.config.load_folder != None:
            self.func = self.load_func()
        else:
            self.func = ODEFunc(self.config.nparticles, self.config.dim, self.config.nn_widths, self.activation_functions, self.dtype).to(self.device)
        
        if self.config.wandb:
            wandb.watch(self.func, log='all', log_freq=10)

        self.config.nparameters = count_parameters(self.func)

        self.loss_func = set_loss_func(self.config.loss_func)
        self.optimizer = self.set_optimizer(self.config.optimizer)
        self.scheduler = self.set_scheduler(self.config.scheduler, self.config.scheduling_factor)

    def predict_traj(self, batch_input, traj_steps, steps_per_dt):
        batch_y0, dt, k, r0, inertia =  batch_input
        batch_y0 = tuple(i.to(self.device, non_blocking=True).type(self.dtype) for i in torch.split(batch_y0, [3, 3, 3, 4], dim=-1))
        
        # get timesteps
        effective_dt = dt / steps_per_dt
        batch_t = self.get_batch_t(effective_dt, traj_steps)
        
        # set constants
        if self.parallel:
            self.func.module.k = k.to(self.device, non_blocking=True).type(self.dtype)
            self.func.module.r0 = r0.to(self.device, non_blocking=True).type(self.dtype)
            self.func.module.inertia = inertia.to(self.device, non_blocking=True).type(self.dtype)
        else:
            self.func.k = k.to(self.device, non_blocking=True).type(self.dtype)
            self.func.r0 = r0.to(self.device, non_blocking=True).type(self.dtype)
            self.func.inertia = inertia.to(self.device, non_blocking=True).type(self.dtype)
        
        options = dict(inertia=inertia.to(self.device, non_blocking=True).type(self.dtype))
        
        # TODO: add assertion to check right dimensions
        pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE', options=options)

        pred_y = torch.swapaxes(torch.cat(pred_y, dim=-1), 0, 1)
        
        # TODO: assert energy conservation
        # if self.epoch < 10:
        #     self.training_dataset.assert_energy_conservation(pred_y, self.func.net, self.traj_steps, self.steps_per_dt)
        return pred_y

    def train(self):
        for self.epoch in range(1, self.config.epochs + 1):
            self.batch_loss = []
            if self.parallel:
                self.training_dataloader.sampler.set_epoch(self.epoch)
    
            self.start_time = time.perf_counter()
            
            if self.early_stopping:
                break

            for self.itr, (batch_input, batch_y, batch_energy) in enumerate(self.training_dataloader):
                self.itr_start_time = time.perf_counter()

                # zero out gradients with less memory operations
                for param in self.func.parameters():
                    param.grad = None

                if self.parallel:
                    batch_y = batch_y.to(self.device, non_blocking=True).type(self.dtype)

                self.loss = self.forward_pass(batch_input, batch_y, batch_energy)
                self.batch_loss.append(self.loss.item())

                # backward pass      
                self.loss.backward() 

                if self.config.optimizer == 'LBFGS':
                    raise NotImplementedError
                else:
                    self.optimizer.step()    
                
                if self.is_master():
                    self.log_itr()

            if self.is_master():
                self.log_epoch()
                self.after_epoch()
            if self.parallel:
                torch.distributed.barrier()
        
        # last checkpoint and save
        if self.is_master():
            self.checkpoint(final=True)

    def log_itr(self):
        if self.config.wandb:
            self.wandb.log({
                # TODO: make train loss into a dictionary that is pased, WANDB should understand this
                'batch_training_loss': self.loss.item(),
                # 'itr_time': time.perf_counter() - self.itr_start_time,
            })
      
    def log_epoch(self):
        # perform evaluation on validation test
        if self.epoch % self.config.evaluation_freq == 0 or self.epoch == 1 or self.epoch == self.config.epochs:
            validation_loss = self.evaluate('validation')
        else:
            validation_loss = None

        if self.config.wandb:
            self.wandb.log({
                'epoch': self.epoch, 
                'training_loss': np.mean(self.batch_loss),
                'validation_loss': validation_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                # 'epoch_time': time.perf_counter() - self.start_time,
                })
        
    def after_epoch(self):
        print(f'Epoch {self.epoch}, train loss: {np.mean(self.batch_loss)}, epoch time: {time.perf_counter() - self.start_time:.2f}s')

        if self.config.scheduler == 'CyclicLR':
            self.scheduler.step()
        elif self.epoch % self.config.scheduling_freq == 0 and self.config.scheduler != None:
            self.scheduler.step()

        if self.epoch % self.config.checkpoint_freq == 0:
            self.checkpoint()

        # TODO: implement increasing batch length
        # TODO: implement early stopping via WandB
        
    def load_func(self):
        loaded_state = torch.load(f'{self.config.load_folder}/model.pt')
        if type(loaded_state) == list:
            kwargs, state_dict = torch.load(f'{self.config.load_folder}/model.pt')

            # get specific NN architecture
            self.config.dim = kwargs['dim']
            self.config.nn_widths = kwargs['widths']
            self.config.activation_functions = kwargs['functions']
        else:
            state_dict = loaded_state
            # raise ValueError('model.pt should be a list of kwargs and state_dict, the previous behaviour has been depreciated')
        
        # in case we load a DDP model checkpoint to a non-DDP model
        model_dict = OrderedDict()
        pattern = re.compile('module.')

        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        func = ODEFunc(self.config.nparticles, self.config.dim, self.config.nn_widths, self.config.activation_functions, self.dtype).to(self.device)
        func.load_state_dict(model_dict)
        return func
        
    def get_dataloader(self, dataset, no_batch=False, shuffle=False):
        # TODO: make this cleaner
        params = {'num_workers': self.config.num_workers}
        if no_batch:
            params['batch_size'] = 1
        else:
            params['batch_size'] = self.config.batch_size
        
        if shuffle:
            params['shuffle'] = True
        else:
            params['shuffle'] = False

        return torch.utils.data.DataLoader(dataset, **params)

    def set_optimizer(self, optimizer):
        if optimizer == 'Adadelta':
            return torch.optim.Adadelta(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'Adagrad':
            return torch.optim.Adagrad(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'Adam':
            return torch.optim.Adam(self.func.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif optimizer == 'AdamW':
            return torch.optim.AdamW(self.func.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif optimizer == 'Adamax':
            return torch.optim.Adamax(self.func.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=self.config.weight_decay)
        elif optimizer == 'SGD':
            return torch.optim.SGD(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'ASGD':
            return torch.optim.ASGD(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'RMSProp':
            return torch.optim.RMSprop(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'Rprop':
            return torch.optim.Rprop(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'LBFGS':
            return torch.optim.LBFGS(self.func.parameters(), lr=self.config.learning_rate)
        else:
            raise Exception('optimizer not implemented')

    def set_scheduler(self, scheduler, alpha):
        if scheduler == 'LambdaLR':
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=alpha)
        elif scheduler == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=alpha * self.config.learning_rate, max_lr=self.config.learning_rate, step_size_up=self.config.scheduling_freq, cycle_momentum=False)
        elif scheduler == None:
            return
        else:
            raise Exception('scheduler not implemented')

    def evaluate(self, dataset):
        with torch.no_grad():
            # TODO: maybe move this elsewhere and make it more robust? maybe have an Evaluation class
            
            if dataset == 'validation':
                dataloader = self.validation_dataloader
                dataset = self.validation_dataset
                steps_per_dt = self.config.eval_steps_per_dt
                dataset_steps = self.config.eval_dataset_steps
            elif dataset == 'test':
                dataloader = self.test_dataloader
                dataset = self.test_dataset
                steps_per_dt = self.config.eval_steps_per_dt
                dataset_steps = self.config.eval_dataset_steps
            elif dataset == 'train':
                dataloader = self.training_dataloader
                dataset = self.training_dataset
                steps_per_dt = self.config.steps_per_dt
                dataset_steps = self.config.dataset_steps
            else:
                raise ValueError(f'dataset {dataset} not recognised')
            
            # orig_length = copy.copy(dataset.traj_length)
            # TODO: finish this similar to plot_traj
            # dataset.update(self.eval_dataset_steps)

            traj_steps = dataset_steps * steps_per_dt

            eval_loss = []
            for batch_input, batch_y, _ in dataloader:
                # forward pass
                # TODO: traj_steps and steps_per_dt should take into account which dataset is used
                pred_y = self.predict_traj(batch_input, traj_steps=traj_steps, steps_per_dt=steps_per_dt)
                
                # loss of the projected trajectory by one dt
                loss, loss_parts = final_mse_pos(pred_y, batch_y, dataset.stds, dataset.means, True)
                eval_loss.append(loss.cpu().item())

                del pred_y, loss, batch_input, batch_y
            
            eval_loss = np.mean(eval_loss)

            # dataset.update(orig_length)
        return eval_loss

    def is_master(self):
        return (self.parallel and is_main_process()) or not self.parallel

    def checkpoint(self, final=False):
        path = f'{self.config.subfolder}/checkpoints/{self.epoch}'
        if not os.path.exists(path):
            os.makedirs(path)
        if self.parallel:
            torch.save([self.func.module.kwargs, self.func.state_dict()], f'{path}/model.pt')
        else:
            torch.save([self.func.kwargs, self.func.state_dict()], f'{path}/model.pt')
        
        return None

    def get_batch_t(self, dt, traj_steps):
        if type(dt) == torch.Tensor:
            return torch.linspace(0.0,dt[0]*(traj_steps),(traj_steps)+1).to(self.device, non_blocking=True).type(self.dtype)
        else:
            return torch.linspace(0.0,dt*(traj_steps),(traj_steps)+1).to(self.device, non_blocking=True).type(self.dtype)

def lambda_lr(epoch):
    return 0.95 ** epoch
            