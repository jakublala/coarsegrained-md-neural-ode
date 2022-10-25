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

from nn.wandb import Wandb
from nn.config import Config
from data.dataset import Dataset
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint
from nn.activations import get_activation_functions
from diffmd.utils import *
from nn.losses import *


class Trainer():

    def __init__(self, config_file):
        self.parallel = False
        self.early_stopping = False

        self.config = Config(config_file)
        
        if self.config.wandb and self.is_master():
            self.wandb = Wandb(self.config)
            # update config based on wandb sweep
            self.config.assign_sweep_config(self.wandb.sweep_values)
            
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

        if self.config.load_folder != None:
            self.func = self.load_func()
            # self.logger.load_previous(self.load_folder)
            self.start_epoch = int(self.logger.epoch[-1])
        else:
            self.func = ODEFunc(self.config.nparticles, self.config.dim, self.config.nn_widths, self.activation_functions, self.dtype).to(self.device)
            self.start_epoch = 0

        self.config.nparameters = count_parameters(self.func)

        self.loss_func = self.set_loss_func(self.config.loss_func)
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
        for self.epoch in range(self.start_epoch + 1, (self.start_epoch + self.config.epochs) + 1):
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
            self.wandb.run.log({
                # TODO: make train loss into a dictionary that is pased, WANDB should understand this
                'batch_training_loss': self.loss.item(),
                # 'itr_time': time.perf_counter() - self.itr_start_time,
            })
      
    def log_epoch(self):
        # perform evaluation on validation test
        if self.epoch % self.config.evaluation_freq == 0 or self.epoch == self.start_epoch + 1:
            validation_loss = self.evaluate('validation')
        else:
            validation_loss = None

        
        if self.config.wandb:
            self.wandb.run.log({
                'epoch': self.epoch, 
                'training_loss': np.mean(self.batch_loss),
                'validation_loss': validation_loss,
                # 'epoch_time': time.perf_counter() - self.start_time,
                })
        

    def after_epoch(self):
        print(f'Epoch {self.epoch}, train loss: {np.mean(self.batch_loss)}, validation loss: {self.evaluate("validation")}, epoch time: {time.perf_counter() - self.start_time:.2f}s')

        if self.config.scheduler == 'CyclicLR':
            self.scheduler.step()
        elif self.epoch % self.config.scheduling_freq == 0 and self.config.scheduler != None:
            self.scheduler.step()

        # if self.epoch % self.evaluation_freq == 0 or self.epoch == self.start_epoch + 1:
        #     self.logger.test_loss[-1] = self.evaluate('test')

        if self.epoch % self.config.checkpoint_freq == 0:
            self.checkpoint()

        # TODO: implement
        # if self.epoch % self.batch_length_freq == 0:
        #     self.increase_batch_length()

        # early stopping
        # if self.epoch % self.stopping_freq == 0:

            
            # divergent / non-convergent
            # if len(self.loss_meter.checkpoints) > 1 + self.stopping_look_back:
            #     if self.loss_meter.checkpoints[-1-self.stopping_look_back] < self.loss_meter.checkpoints[-1]:
            #         print('early stopping as non-convergent')
            #         self.early_stopping = True
            # TODO: add proper stale convergence and test it out
            # stale convergence
            # if np.sd(self.loss_meter.losses[-self.stopping_freq:]) > 0.001:
            #     print('early stopping as stale convergence')
            #     return self.func, self.loss_meter
        
    # TODO: not implemented
    # def increase_batch_length(self):
    #     self.batch_length += self.batch_length_step
    #     self.training_dataset.update(self.batch_length)
    #     self.loss_meter.reset()

    # def avg_epoch_loss(self):
    #     indices = [i for i, e in enumerate(self.logger.epoch) if e == self.epoch]
    #     self.logger.avg_train_loss[-1] = np.mean([self.logger.train_loss[i] for i in indices])
    #     self.logger.run_avg_train_loss[-1] = self.loss_meter.avg
        
    def load_func(self):
        loaded_state = torch.load(f'{self.config.load_folder}/model.pt')
        if type(loaded_state) == list:
            kwargs, state_dict = torch.load(f'{self.config.load_folder}/model.pt')

            # get specific NN architecture
            self.dim = kwargs['dim']
            self.nn_widths = kwargs['widths']
            self.activation_functions = kwargs['functions']
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
        func = ODEFunc(self.nparticles, self.dim, self.nn_widths, self.activation_functions, self.dtype).to(self.device)
        func.load_state_dict(model_dict)
        return func
        
    def plot_traj(self, subfolder, final):
        def get_anchored_text():
            at = AnchoredText(f'epoch: {self.epoch}', prop=dict(size=10), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            return at

        def plot(indices, body_id, title, filename):
            colours = ['r-', 'b-', 'g-', 'm-']
            fig, ax = plt.subplots()
            for c, i in enumerate(indices):
                ax.set_title(title)
                ax.plot(batch_t, batch_y[:,body_id,i], 'k--', alpha=0.3, label=f'true {i}')
                ax.plot(pred_t, pred_y[:,body_id,i], colours[c], alpha=0.5, label=f'pred {i}')
            ax.add_artist(get_anchored_text())
            plt.xlabel('Time')
            fig.savefig(f'{subfolder}/{filename}.png')
            plt.close(fig)            

        # temporarily change traj length for plotting
        if final:
            dataset_steps = 100
            steps_per_dt = self.config.steps_per_dt * 10
        else:
            dataset_steps = 100
            steps_per_dt = self.config.steps_per_dt
            
        
        traj_steps = dataset_steps * self.config.steps_per_dt
        
        # TODO: remove this update and do the plotting differently so that we don't have to change the traj length inside plotting
        # this especially caused problems with using a fraction of the dataset
        # self.training_dataset.update(dataset_steps)
        with torch.no_grad():
            # get the earliest init conditions to ensure trajectories are long enough
            init_index = self.training_dataset.init_IDS.index(min(self.training_dataset.init_IDS, key=len))
            batch_input, batch_y, _ = self.training_dataset[init_index]
            batch_input = list(batch_input)
            batch_input[0] = batch_input[0].unsqueeze(0)
            dataset_dt = batch_input[1]
            
            # if final:
            #     traj_index = int(self.training_dataset.init_IDS[init_index].split('-')[0])
            #     dt = self.training_dataset.trajs[traj_index].lammps_dt
            #     batch_input[1] = dt
            #     traj_steps = 10000
            #     subfolder = f'results/temp/plot'
            #     steps_per_dt = 1

            # else:
            #     dt = batch_input[1] / self.steps_per_dt
            #     steps_per_dt = self.steps_per_dt
            #     subfolder = f'results/temp/plot2'

            dt = batch_input[1] / steps_per_dt
            start = time.perf_counter()
            pred_y = self.predict_traj(batch_input, traj_steps=traj_steps, steps_per_dt=steps_per_dt).squeeze().cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            
            pred_t = self.get_batch_t(dt, traj_steps).cpu().numpy()
            batch_t = self.get_batch_t(dataset_dt, dataset_steps).cpu().numpy()

            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_quat = [9, 10, 11, 12]

            for i in [0, 1]:
                plot(ind_vel, i, f'velocities {i+1}', f'vel{i+1}')
                plot(ind_ang, i, f'angular velocities {i+1}', f'angvel{i+1}')
                plot(ind_quat, i, f'quaternions {i+1}', f'quat{i+1}')
            
            # centre of mass positions (set initial position of first COM to zero)
            batch_y[:,:,6:9] = batch_y[:,:,6:9] - batch_y[:,[0],6:9]
            pred_y[:,:,6:9] = pred_y[:,:,6:9] - pred_y[:,[0],6:9]
            
            # centre of mass separation
            batch_y_sep = np.linalg.norm(batch_y[:,1,6:9] - batch_y[:,0,6:9], axis=-1)
            pred_y_sep = np.linalg.norm(pred_y[:,1,6:9] - pred_y[:,0,6:9], axis=-1)

            fig, ax = plt.subplots()
            ax.set_title('separation')
            ax.plot(batch_t, batch_y_sep, 'k--', alpha=0.3, label=f'true')
            ax.plot(pred_t, pred_y_sep, 'r-', alpha=0.5, label=f'pred')
            ax.add_artist(get_anchored_text())
            fig.savefig(f'{subfolder}/sep.png')
            plt.close(fig)

        # revert changes to traj length
        # self.training_dataset.update(self.dataset_steps)
            
    def plot_losses(self, subfolder):
        fig, ax = plt.subplots()

        if self.config.loss_func == 'energy':
            ax.plot(self.logger.epoch, self.logger.train_loss_energy, label='train')
        else:
            avg_train_loss = [i for i in self.logger.avg_train_loss if i is not None]
            ax.plot(range(1, self.epoch+1), avg_train_loss, 'b-', label='train')
            run_avg_train_loss = [i for i in self.logger.run_avg_train_loss if i is not None]
            ax.plot(range(1, self.epoch+1), run_avg_train_loss, 'b-', alpha=0.5)

        eval_epochs = []
        test_loss = []
        for i, tl in enumerate(self.logger.test_loss):
            if tl is not None:
                eval_epochs.append(self.logger.epoch[i])
                test_loss.append(tl)
        ax.plot(eval_epochs, test_loss, 'r-', label='test')

        ax.set_xlabel('Number of Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.savefig(f'{subfolder}/loss.png')
        plt.close(fig)
        return

    def plot_lr(self, subfolder):
        fig, ax = plt.subplots()
        ax.set_title('learning rate evolution')
        ax.plot(self.loss_meter.lrs)
        fig.savefig(f'{subfolder}/lr.png')
        plt.close(fig)
        return

    def log_metadata(self, config):
        if not os.path.exists(f'{self.config.subfolder}'):
                os.makedirs(f'{self.config.subfolder}')
        shutil.copyfile('config.yml', f'{self.config.subfolder}/config_{self.epoch}.yaml')
        
    def set_loss_func(self, loss_func):
        if 'all-mse' == loss_func:
            raise NotImplementedError('all-mse loss function not implemented with new steps_per_dt')
            return all_mse
        elif 'final-mse' == loss_func:
            return final_mse
        elif 'all-mse-pos' == loss_func:
            raise NotImplementedError('all-mse loss function not implemented with new steps_per_dt')
            return all_mse_pos
        elif 'final-mse-pos' == loss_func:
            return final_mse_pos
        elif 'energy' == loss_func:
            return energy
        elif 'final-mse-pos-and-energy':
            return final_mse_pos_and_energy
        else:
            raise ValueError(f'loss function {loss_func} not recognised')

        
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
            return torch.optim.Adam(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'AdamW':
            return torch.optim.AdamW(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'NAdam':
            return torch.optim.NAdam(self.func.parameters(), lr=self.config.learning_rate)
        elif optimizer == 'RAdam':
            return torch.optim.RAdam(self.func.parameters(), lr=self.config.learning_rate)
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
            elif dataset == 'test':
                dataloader = self.test_dataloader
                dataset = self.test_dataset
            elif dataset == 'train':
                dataloader = self.training_dataloader
                dataset = self.training_dataset
            else:
                raise ValueError(f'dataset {dataset} not recognised')
            
            # orig_length = copy.copy(dataset.traj_length)
            # TODO: finish this similar to plot_traj
            # dataset.update(self.eval_dataset_steps)

            traj_steps = self.config.eval_dataset_steps * self.config.steps_per_dt

            eval_loss = []
            for batch_input, batch_y, _ in dataloader:
                # forward pass
                # TODO: traj_steps and steps_per_dt should take into account which dataset is used
                pred_y = self.predict_traj(batch_input, traj_steps=traj_steps, steps_per_dt=self.config.eval_steps_per_dt)
                
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
        subfolder = f'{self.config.subfolder}/{self.epoch}'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        if self.parallel:
            torch.save([self.func.module.kwargs, self.func.state_dict()], f'{subfolder}/model.pt')
        else:
            torch.save([self.func.kwargs, self.func.state_dict()], f'{subfolder}/model.pt')
        # self.plot_traj(subfolder, final)
        # self.plot_losses(subfolder)
        # self.plot_lr(subfolder)
        # self.logger.save_csv(subfolder)
        return None

    def get_batch_t(self, dt, traj_steps):
        if type(dt) == torch.Tensor:
            return torch.linspace(0.0,dt[0]*(traj_steps),(traj_steps)+1).to(self.device, non_blocking=True).type(self.dtype)
        else:
            return torch.linspace(0.0,dt*(traj_steps),(traj_steps)+1).to(self.device, non_blocking=True).type(self.dtype)

def lambda_lr(epoch):
    return 0.95 ** epoch
            