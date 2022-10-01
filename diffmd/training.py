from distutils.command.config import config
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import time
import os
import shutil
import numpy as np
import sigopt
import re
from collections import OrderedDict

from data.reader import Reader
from data.logger import Logger
from data.dataset import Dataset
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint
from diffmd.utils import *
from diffmd.losses import *


class Trainer():

    def __init__(self, config):
        self.config = config
        self.load_folder = config['load_folder']
        
        if self.load_folder is None:
            self.day, self.time = get_run_ID()
            self.subfolder = self.get_subfolder() 
        else:
            self.day, self.time = self.load_folder.split('/')[-3:-1]
            self.subfolder = '/'.join(self.load_folder.split('/')[:-1])
            
        self.folder = config['folder']
        self.device = set_device(config['device'])
        self.dtype = set_dtype(config['dtype'])
        self.sigopt = config['sigopt']
        self.parallel = False
        
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        
        self.dataset_steps = config['dataset_steps']
        self.steps_per_dt = config['steps_per_dt']
        self.traj_steps = self.dataset_steps * self.steps_per_dt
        
        # TODO: currently not implemented
        # self.batch_length_step = config['batch_length_step']
        # self.batch_length_freq = config['batch_length_freq']

        self.eval_dataset_steps = config['eval_dataset_steps']
        self.eval_steps_per_dt = config['eval_steps_per_dt']
        self.eval_init_skip = config['eval_init_skip']
        
        self.shuffle = config['shuffle']
        self.num_workers = config['num_workers']
        self.batch_size = config['batch_size']
        self.nn_widths = config['nn_widths']
        self.activation_functions = self.get_activation_functions(config['activation_function'])
        self.loss_func_name = config['loss_func']
        self.optimizer_name = config['optimizer']
        self.scheduler_name = config['scheduler']
        self.scheduling_factor = config['scheduling_factor']
        

        # dataset setup
        self.training_dataset = Dataset(config, dataset_type='train', traj_length=self.dataset_steps, dataset_fraction=config['training_fraction'], random_dataset=config['random_dataset'])
        self.test_dataset = Dataset(config, dataset_type='test', traj_length=self.eval_dataset_steps)
        self.validation_dataset = Dataset(config, dataset_type='validation', traj_length=self.eval_dataset_steps)
        self.training_dataloader = self.get_dataloader(self.training_dataset, shuffle=self.shuffle) 
        self.test_dataloader = self.get_dataloader(self.test_dataset) 
        self.validation_dataloader = self.get_dataloader(self.validation_dataset)

        assert len(self.training_dataset) > self.batch_size, "Batch size is too large for the dataset. Please reduce the batch size or increase the dataset size."

        self.logger = Logger()
        self.loss_meter = RunningAverageMeter()
        
        self.nparticles = 2
        self.dim = 3 + (2*4)
        self.printing_freq = config['printing_freq']
        self.itr_printing_freq = config['itr_printing_freq']
        self.stopping_freq = config['stopping_freq']
        self.stopping_look_back = config['stopping_look_back']
        self.early_stopping = False
        self.scheduling_freq = config['scheduling_freq']
        self.evaluation_freq = config['evaluation_freq']
        self.checkpoint_freq = config['checkpoint_freq']

        if self.load_folder != None:
            self.func = self.load_func()
            self.logger.load_previous(self.load_folder)
            self.start_epoch = int(self.logger.epoch[-1])
        else:
            self.func = ODEFunc(self.nparticles, self.dim, self.nn_widths, self.activation_functions, self.dtype).to(self.device)
            self.start_epoch = 0

        self.nparameters = count_parameters(self.func)

        self.weight_decay = config['weight_decay']
        self.loss_func = self.set_loss_func(self.loss_func_name)
        self.normalize_loss = config['normalize_loss']
        self.optimizer = self.set_optimizer(self.optimizer_name)
        self.scheduler = self.set_scheduler(self.scheduler_name, self.scheduling_factor)
        
        

        print(f'device = {self.device}')
        print(f'NN architecture = 11 - {self.nn_widths} - 1')
        print(f'number of parameters = {self.nparameters}')
        print(f'learning rate = {self.learning_rate}, optimizer = {self.optimizer_name}')
        print(f'scheduler = {self.scheduler_name}, scheduling factor = {self.scheduling_factor}, scheduling freq = {self.scheduling_freq}')
        print(f'batch size = {self.batch_size}, dataset_steps = {self.dataset_steps}, steps per dt = {self.steps_per_dt}')

    def get_subfolder(self):

        def add_second(time):
            # HACK: can add above 60 seconds
            time = time.split('-')
            time[2] = str(int(time[2]) + 1)
            return '-'.join(time)

        subfolder = f'results/{self.day}/{self.time}'
        while os.path.exists(f'{subfolder}'):
            self.time = add_second(self.time)
            subfolder = f'results/{self.day}/{self.time}'
        return subfolder

    def forward_pass(self, batch_input, traj_steps, steps_per_dt):
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
        for self.epoch in range(self.start_epoch + 1, (self.start_epoch + self.epochs) + 1):
            if self.parallel:
                self.training_dataloader.sampler.set_epoch(self.epoch)

            if (self.parallel and is_main_process()) or not self.parallel:
                self.log_metadata(self.config)
            
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

                # forward pass                
                pred_y = self.forward_pass(batch_input, self.traj_steps, self.steps_per_dt)

                # compute loss
                if self.loss_func_name == 'energy':
                    loss = self.loss_func(self.func.net, pred_y, batch_energy)
                else:
                    loss, self.loss_parts = self.loss_func(pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means, self.normalize_loss)

                # backward pass      
                loss.backward() 
                
                if self.optimizer_name == 'LBFGS':
                    raise NotImplementedError
                else:
                    self.optimizer.step()    
                
                if (self.parallel and is_main_process()) or not self.parallel:
                    self.loss_meter.update(loss.item(), self.optimizer.param_groups[0]["lr"])

                    if (self.itr+1) % self.itr_printing_freq == 0:
                        self.print_iteration()

                    self.after_itr()

            if self.parallel:
                if is_main_process():
                    self.after_epoch()
                torch.distributed.barrier()
            elif not self.parallel:
                self.after_epoch()
            
        if (self.parallel and is_main_process()) or not self.parallel:
            # last checkpoint and save
            self.checkpoint()
            self.save()
            
        return self.func, self.loss_meter

    def after_itr(self):
        new_entry = {
            'epoch': self.epoch,
            'itr': self.itr+1,
            'lr': self.optimizer.param_groups[0]["lr"],
            'traj_steps': self.dataset_steps,
            'steps_per_dt': self.steps_per_dt,
            'train_loss': sum(self.loss_parts),
            'train_loss_vel': self.loss_parts[0],
            'train_loss_angvel': self.loss_parts[1],
            'train_loss_pos': self.loss_parts[2],
            'train_loss_quat': self.loss_parts[3],
            'test_loss': None,
            'time': time.perf_counter() - self.itr_start_time,
        }
        self.logger.update(new_entry)

    def after_epoch(self):

        if self.scheduler_name == 'CyclicLR':
            self.scheduler.step()
        elif self.epoch % self.scheduling_freq == 0 and self.scheduler_name != None:
            self.scheduler.step()

        if self.epoch % self.printing_freq == 0:
            self.print_loss(self.epoch, self.start_time)

        if self.epoch % self.evaluation_freq == 0:
            self.logger.test_loss[-1] = self.evaluate('test')

        if self.epoch % self.checkpoint_freq == 0:
            self.checkpoint()

        # TODO: implement
        # if self.epoch % self.batch_length_freq == 0:
        #     self.increase_batch_length()

        # early stopping
        if self.epoch % self.stopping_freq == 0:

            self.loss_meter.checkpoint()
            
            # divergent / non-convergent
            if len(self.loss_meter.checkpoints) > 1 + self.stopping_look_back:
                if self.loss_meter.checkpoints[-1-self.stopping_look_back] < self.loss_meter.checkpoints[-1]:
                    print('early stopping as non-convergent')
                    self.early_stopping = True
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

    def load_func(self):
        loaded_state = torch.load(f'{self.load_folder}/model.pt')
        if type(loaded_state) == list:
            kwargs, state_dict = torch.load(f'{self.load_folder}/model.pt')

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


    def print_loss(self, itr, start_time):
        print(f'Epoch: {itr}, Running avg elbo: {self.loss_meter.avg}')
        print(f'Current loss: {self.loss_meter.val}')
        print('Last epoch took:     ', time.perf_counter() - start_time, flush=True)
        print(f'Learning rate: {self.loss_meter.lrs[-1]}')
        print(f'Current learning rate: {self.optimizer.param_groups[0]["lr"]}')
        print(f'Current dataset steps: {self.dataset_steps}, Current steps per dt: {self.steps_per_dt}')
        
    def print_iteration(self):
        print(f'Epoch: {self.epoch}, Iteration: {self.itr+1}, Loss: {self.loss_meter.val}')
        print(f'Last iteration took:', time.perf_counter() - self.itr_start_time, flush=True)

    def plot_traj(self, checkpoint=False):
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
            fig.savefig(f'{subfolder}/{filename}.png')
            plt.close(fig)            

        # temporarily change traj length for plotting
        if checkpoint:
            dataset_steps = 100
            subfolder = f'results/{self.day}/{self.time}/{self.epoch}'
        else:
            dataset_steps = 100
            subfolder = f'results/{self.day}/{self.time}'

        traj_steps = dataset_steps * self.steps_per_dt
        
        self.training_dataset.update(dataset_steps)
        with torch.no_grad():
            # get the earliest init conditions to ensure trajectories are long enough
            init_index = self.training_dataset.init_IDS.index(min(self.training_dataset.init_IDS, key=len))
            batch_input, batch_y, _ = self.training_dataset[init_index]
            batch_input = list(batch_input)
            batch_input[0] = batch_input[0].unsqueeze(0)
            batch_input = tuple(batch_input)

            pred_y = self.forward_pass(batch_input, traj_steps=traj_steps, steps_per_dt=self.steps_per_dt).squeeze().cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            
            effective_dt = batch_input[1] / self.steps_per_dt
            pred_t = self.get_batch_t(effective_dt, traj_steps).cpu().numpy()
            batch_t = self.get_batch_t(batch_input[1], dataset_steps).cpu().numpy()

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
        self.training_dataset.update(self.dataset_steps)
            
    def plot_losses(self, subfolder):
        fig, ax = plt.subplots()
        ax.plot(self.logger.epoch, self.logger.train_loss, label='train')

        eval_epochs = []
        test_loss = []
        for i, tl in enumerate(self.logger.test_loss):
            if tl is not None:
                eval_epochs.append(self.logger.epoch[i])
                test_loss.append(tl)
        ax.plot(eval_epochs, test_loss, label='test')

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
        subfolder = f'results/{self.day}/{self.time}/'
        if not os.path.exists(f'{subfolder}'):
                os.makedirs(f'{subfolder}')
        if self.epoch == 0 or self.epoch == self.start_epoch:
            shutil.copyfile('config.yaml', f'{subfolder}/config_{self.epoch}.yaml')
        else:
            pass

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
        else:
            raise ValueError(f'loss function {loss_func} not recognised')

        
    def get_dataloader(self, dataset, no_batch=False, shuffle=False):
        # TODO: make this cleaner
        params = {'num_workers': self.num_workers}
        if no_batch:
            params['batch_size'] = 1
        else:
            params['batch_size'] = self.batch_size
        
        if shuffle:
            params['shuffle'] = True
        else:
            params['shuffle'] = False

        return torch.utils.data.DataLoader(dataset, **params)

    def set_optimizer(self, optimizer):
        if optimizer == 'Adadelta':
            return torch.optim.Adadelta(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'Adagrad':
            return torch.optim.Adagrad(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'Adam':
            return torch.optim.Adam(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'AdamW':
            return torch.optim.AdamW(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'NAdam':
            return torch.optim.NAdam(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'RAdam':
            return torch.optim.RAdam(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'Adamax':
            return torch.optim.Adamax(self.func.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        elif optimizer == 'SGD':
            return torch.optim.SGD(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'ASGD':
            return torch.optim.ASGD(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'RMSProp':
            return torch.optim.RMSprop(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'Rprop':
            return torch.optim.Rprop(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'LBFGS':
            return torch.optim.LBFGS(self.func.parameters(), lr=self.learning_rate)
        else:
            raise Exception('optimizer not implemented')

    def set_scheduler(self, scheduler, alpha):
        if scheduler == 'LambdaLR':
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=alpha)
        elif scheduler == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=alpha * self.learning_rate, max_lr=self.learning_rate, step_size_up=self.scheduling_freq, cycle_momentum=False)
        elif scheduler == None:
            return
        else:
            raise Exception('scheduler not implemented')

    def evaluate(self, dataset):
        with torch.no_grad():
            # TODO: maybe move this elsewhere and make it more robust? maybe have an Evaluation class
            
            if dataset == 'validate' :
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
            
            eval_loss = []
            for batch_input, batch_y, _ in dataloader:
                # forward pass
                pred_y = self.forward_pass(batch_input, traj_steps=self.eval_dataset_steps, steps_per_dt=self.eval_steps_per_dt)

                # loss of the projected trajectory by one dt
                loss, loss_parts = final_mse(pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means)

                eval_loss.append(loss.cpu().item())

                del pred_y, loss, batch_input, batch_y
            
            eval_loss = np.mean(eval_loss)
            self.loss_meter.evals.append(eval_loss)
            
        return eval_loss

    def save(self):
        subfolder = f'results/{self.day}/{self.time}/{self.epoch}'
        if self.sigopt:
            self.report_sigopt(subfolder)
        return

    def report_sigopt(self, subfolder):
        files = os.listdir(subfolder)
        for f in files:
            if f.endswith('.png'):
                sigopt.log_image(f'{subfolder}/{f}')
        return

    def checkpoint(self):
        subfolder = f'results/{self.day}/{self.time}/{self.epoch}'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        torch.save([self.func.kwargs, self.func.state_dict()], f'{subfolder}/model.pt')
        self.plot_traj(True)
        self.plot_losses(subfolder)
        self.plot_lr(subfolder)
        self.logger.save_csv(subfolder)
        return None

    def print_cuda_memory(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f'Total memory: {t}')
        print(f'Reserved memory: {r}')
        print(f'Allocated memory: {a}')
        print(f'Free memory: {f}')
        print('====================================================')
        return None
    
    def get_batch_t(self, dt, traj_steps):
        if type(dt) == torch.Tensor:
            return torch.linspace(0.0,dt[0]*(traj_steps),(traj_steps)+1).to(self.device, non_blocking=True).type(self.dtype)
        else:
            return torch.linspace(0.0,dt*(traj_steps),(traj_steps)+1).to(self.device, non_blocking=True).type(self.dtype)

    def get_activation_functions(self, function):
        def get_function(string):
            if string == 'relu':
                return nn.ReLU()
            elif string == 'leaky_relu':
                return nn.LeakyReLU()
            elif string == 'tanh':
                return nn.Tanh()
            elif string == 'sigmoid':
                return nn.Sigmoid()
            else:
                raise Exception('activation function not implemented')

        if type(function) == str:
            return [get_function(function) for i in range(len(self.nn_widths))]
        elif type(function) == list and len(function) == len(self.nn_widths):
            return [get_function(i) for i in function]
        else:
            raise Exception('activation function must be a string or a list of strings of the same length as the number of layers')

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
        self.reset()
        self.checkpoints = []
        # TODO: do lrs and evals in separate classes and not in the same class, which makes it messy
        self.lrs = []
        self.evals = []

    def reset(self):
        self.val = None
        self.avg = 0
        self.checkpoints = []

    def update(self, val, lr):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.log(val, lr)
        
    def log(self, val, lr):
        self.losses.append(val)
        self.lrs.append(lr)

    def checkpoint(self):
        self.checkpoints.append(self.avg)

def lambda_lr(epoch):
    return 0.95 ** epoch
            