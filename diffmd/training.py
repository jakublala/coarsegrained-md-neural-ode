from distutils.command.config import config
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
from diffmd.utils import get_run_ID, count_parameters
from diffmd.losses import *

class Trainer():

    def __init__(self, config):
        self.folder = config['folder']
        self.day, self.time = get_run_ID()
        self.device = config['device']
        self.sigopt = config['sigopt']
        
        # TODO: log everything similarly to device into print file
        self.epochs = config['epochs']
        self.start_epoch = config['start_epoch']
        self.learning_rate = config['learning_rate']
        self.batch_length = config['batch_length']
        self.batch_length_step = config['batch_length_step']
        self.batch_length_freq = config['batch_length_freq']
        self.eval_batch_length = config['eval_batch_length']
        self.shuffle = config['shuffle']
        self.num_workers = config['num_workers']
        self.batch_size = config['batch_size']
        self.nn_widths = config['nn_widths']
        self.activation_function = config['activation_function']
        self.load_folder = config['load_folder']
        self.loss_func_name = config['loss_func']
        self.optimizer_name = config['optimizer']
        self.scheduler_name = config['scheduler']
        self.scheduling_factor = config['scheduling_factor']
        
        self.dtype = config['dtype']

        # dataset setup
        self.training_dataset = Dataset(config, dataset_type='train', batch_length=self.batch_length, dataset_fraction=config['training_fraction'])
        self.test_dataset = Dataset(config, dataset_type='test', batch_length=self.eval_batch_length)
        self.validation_dataset = Dataset(config, dataset_type='validation', batch_length=self.eval_batch_length)
        self.training_dataloader = self.get_dataloader(self.training_dataset) 
        self.test_dataloader = self.get_dataloader(self.test_dataset) 
        self.validation_dataloader = self.get_dataloader(self.validation_dataset)

        self.log_metadata(config)
        self.logger = Logger()
        self.loss_meter = RunningAverageMeter()
        
        self.nparticles = 2
        self.dim = 3 + (2*4)
        self.printing_freq = config['printing_freq']
        self.itr_printing_freq = config['itr_printing_freq']
        self.plotting_freq = config['plotting_freq']
        self.stopping_freq = config['stopping_freq']
        self.stopping_look_back = config['stopping_look_back']
        self.scheduling_freq = config['scheduling_freq']
        self.evaluation_freq = config['evaluation_freq']
        self.checkpoint_freq = config['checkpoint_freq']

        self.func = ODEFunc(self.nparticles, self.dim, self.nn_widths, self.activation_function, self.dtype).to(self.device)
        self.nparameters = count_parameters(self.func)

        self.loss_func = self.set_loss_func(self.loss_func_name)
        self.optimizer = self.set_optimizer(self.optimizer_name)
        self.scheduler = self.set_scheduler(self.scheduler_name, self.scheduling_factor)
        
        if self.load_folder != None:
            self.load_func()

        print(f'device = {self.device}')
        print(f'NN architecture = 11 - {self.nn_widths} - 1')
        print(f'number of parameters = {self.nparameters}')
        print(f'learning rate = {self.learning_rate}, optimizer = {self.optimizer_name}')
        print(f'scheduler = {self.scheduler_name}, scheduling factor = {self.scheduling_factor}, scheduling freq = {self.scheduling_freq}')
        print(f'batch size = {self.batch_size}, traj length = {self.batch_length}')

    def forward_pass(self, batch_input, batch_length=None, parallel=False):
        
        batch_y0, dt, k, r0, inertia =  batch_input
        batch_y0 = tuple(i.to(self.device, non_blocking=True).type(self.dtype) for i in torch.split(batch_y0, [3, 3, 3, 4], dim=-1))

        # get timesteps
        # TODO: this is always the same and so we can make it simpler
        batch_t = self.get_batch_t(dt, batch_length)

        # set constants
        if parallel:
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
        
        return pred_y
        
    def train(self):
        for self.epoch in range(self.start_epoch + 1, (self.start_epoch + self.epochs) + 1):
            self.start_time = time.perf_counter()
            
            for self.itr, (batch_input, batch_y) in enumerate(self.training_dataloader):
                self.itr_start_time = time.perf_counter()

                def closure():
                    for param in self.func.parameters():
                        param.grad = None  
                    # forward pass                
                    pred_y = self.forward_pass(batch_input)
                    loss = self.loss_func(pred_y, batch_y)
                    loss.backward()
                    return loss

                # zero out gradients with less memory operations
                for param in self.func.parameters():
                    param.grad = None

                # forward pass                
                pred_y = self.forward_pass(batch_input)
                loss = self.loss_func(pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means)

                # backward pass      
                loss.backward() 
                self.loss_meter.update(loss.item(), self.optimizer.param_groups[0]["lr"])
        
                if self.optimizer_name == 'LBFGS':
                    self.optimizer.step(closure)
                else:
                    self.optimizer.step()    
                
                if (self.itr+1) % self.itr_printing_freq == 0:
                    self.print_iteration()
                
                # log everything
                self.logger.update([self.epoch, self.itr, self.optimizer.param_groups[0]["lr"], self.batch_length, loss.item(), None, time.perf_counter() - self.itr_start_time])
                    
            if self.after_epoch():
                # if True, then early stopping
                return self.func, self.loss_meter
        
        return self.func, self.loss_meter

    def after_epoch(self):

        if self.scheduler_name == 'CyclicLR':
            self.scheduler.step()
        elif self.epoch % self.scheduling_freq == 0 and self.scheduler_name != None:
            self.scheduler.step()

        if self.epoch % self.printing_freq == 0:
            self.print_loss(self.epoch, self.start_time)

        if self.epoch % self.plotting_freq == 0:
            self.plot_traj(self.epoch)

        if self.epoch % self.evaluation_freq == 0:
            self.logger.log[-1][-2] = self.evaluate(validate=False)

        if self.epoch % self.checkpoint_freq == 0:
            self.checkpoint()

        if self.epoch % self.batch_length_freq == 0:
            self.increase_batch_length()

        # early stopping
        if self.epoch % self.stopping_freq == 0:

            self.loss_meter.checkpoint()
            
            # divergent / non-convergent
            if len(self.loss_meter.checkpoints) > 1 + self.stopping_look_back:
                if self.loss_meter.checkpoints[-1-self.stopping_look_back] < self.loss_meter.checkpoints[-1]:
                    print('early stopping as non-convergent')
                    # TODO: make this compliant with multiple GPUs
                    return True
            # TODO: add proper stale convergence and test it out
            # stale convergence
            # if np.sd(self.loss_meter.losses[-self.stopping_freq:]) > 0.001:
            #     print('early stopping as stale convergence')
            #     return self.func, self.loss_meter
        
    def increase_batch_length(self):
        self.batch_length += self.batch_length_step
        self.training_dataset.update(self.batch_length)

    def load_func(self):
        # in case we load a DDP model checkpoint to a non-DDP model
        state_dict = torch.load(f'{self.load_folder}/model.pt')
        model_dict = OrderedDict()
        pattern = re.compile('module.')

        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        self.func.load_state_dict(model_dict)


    def print_loss(self, itr, start_time):
        print(f'Epoch: {itr}, Running avg elbo: {self.loss_meter.avg}')
        print(f'Current loss: {self.loss_meter.val}')
        print('Last epoch took:     ', time.perf_counter() - start_time, flush=True)
        print(f'Learning rate: {self.loss_meter.lrs[-1]}')
        print(f'Current learning rate: {self.optimizer.param_groups[0]["lr"]}')
        print(f'Current batch length: {self.batch_length}')
        
    def print_iteration(self):
        print(f'Epoch: {self.epoch}, Iteration: {self.itr+1}, Loss: {self.loss_meter.val}')
        print(f'Last iteration took:', time.perf_counter() - self.itr_start_time, flush=True)

    def plot_traj(self, itr, subfolder='temp'):
        if itr == self.plotting_freq:
            if subfolder == 'temp' and os.path.exists(f'{subfolder}'):
                shutil.rmtree(f'{subfolder}')

            if subfolder == 'temp' and not os.path.exists(f'{subfolder}'):
                os.makedirs(f'{subfolder}')

        # temporarily change batch length for plotting
        if subfolder == 'temp':
            batch_length = 1000
        else:
            batch_length = 10000 
        self.training_dataset.update(batch_length)

        with torch.no_grad():
            init_index = self.training_dataset.init_IDS.index(min(self.training_dataset.init_IDS, key=len))
            batch_input, batch_y = self.training_dataset[init_index]
            batch_input = list(batch_input)
            batch_input[0] = batch_input[0].unsqueeze(0)
            batch_input = tuple(batch_input)
            
            pred_y = self.forward_pass(batch_input, batch_length=batch_length).squeeze().cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            batch_t = self.get_batch_t(batch_input[1], batch_length=batch_length).cpu().numpy()

            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_pos = [6, 7, 8]
            ind_quat = [9, 10, 11, 12]
            colours = ['r-', 'b-', 'g-']

            for c, i in enumerate(ind_vel):
                plt.title('velocities 1')
                plt.plot(batch_t, batch_y[:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,0,i], colours[c], alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel1.png')
            plt.close()
            
            for c, i in enumerate(ind_vel):
                plt.title('velocities 2')
                plt.plot(batch_t, batch_y[:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,1,i], colours[c], alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel2.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities 1')
                plt.plot(batch_t, batch_y[:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_angvel1.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities 2')
                plt.plot(batch_t, batch_y[:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_angvel2.png')
            plt.close()
            
            # centre of mass positions (set initial position of first COM to zero)
            batch_y[:,:,6:9] = batch_y[:,:,6:9] - batch_y[:,[0],6:9]
            pred_y[:,:,6:9] = pred_y[:,:,6:9] - pred_y[:,[0],6:9]
            
            for i in ind_pos:
                plt.title('positions 1')
                plt.plot(batch_t, batch_y[:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_pos1.png')
            plt.close()
            
            for i in ind_pos:
                plt.title('positions 2')
                plt.plot(batch_t, batch_y[:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_pos2.png')
            plt.close()

            # centre of mass separation
            batch_y_sep = np.linalg.norm(batch_y[:,1,6:9] - batch_y[:,0,6:9], axis=-1)
            pred_y_sep = np.linalg.norm(pred_y[:,1,6:9] - pred_y[:,0,6:9], axis=-1)

            plt.title('separation')
            plt.plot(batch_t, batch_y_sep, 'k--', alpha=0.3, label=f'true')
            plt.plot(batch_t, pred_y_sep, 'r-', alpha=0.5, label=f'pred')
            plt.savefig(f'{subfolder}/{itr}_sep.png')
            plt.close()

            # quaternions
            for i in ind_quat:
                plt.title('quaternions 1')
                plt.plot(batch_t, batch_y[:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_quat1.png')
            plt.close() 
            
            for i in ind_quat:
                plt.title('quaternions 2')
                plt.plot(batch_t, batch_y[:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_quat2.png')
            plt.close() 
    
        # set back training batch length
        self.training_dataset.batch_length = self.batch_length

    def plot_loss(self, subfolder):
        # TODO: add return figure to be plotted into SigOpt
        plt.title('loss function evolution')
        plt.plot(self.loss_meter.losses)
        plt.xlabel('Number of Iterations')
        plt.savefig(f'{subfolder}/loss.png')
        plt.close()
        return

    def plot_lr(self, subfolder):
        plt.title('learning rate evolution')
        plt.plot(self.loss_meter.lrs)
        plt.savefig(f'{subfolder}/lr.png')
        plt.close()
        return

    def log_metadata(self, config):
        subfolder = f'results/{self.day}/{self.time}/'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        with open(f'{subfolder}/metadata.csv', 'w') as f:
            for key, value in config.items():
                f.write(f'{str(key)},{str(value)} \n')        

    def set_loss_func(self, loss_func):
        if 'all-mse' == loss_func:
            return all_mse
        elif 'final-mse' == loss_func:
            return final_mse
        else:
            raise ValueError(f'loss function {loss_func} not recognised')

        
    def get_dataloader(self, dataset, no_batch=False):
        # TODO: make this cleaner
        if no_batch:
            params = {'batch_size': 1, 'shuffle': self.shuffle, 'num_workers': self.num_workers}
        else:
            params = {'batch_size': self.batch_size, 'shuffle': self.shuffle, 'num_workers': self.num_workers}
        
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
            return torch.optim.Adamax(self.func.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
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

    def evaluate(self, validate=False):
        with torch.no_grad():
            # TODO: maybe move this elsewhere and make it more robust? maybe have an Evaluation class
            if validate:
                dataloader = self.validation_dataloader
                dataset = self.validation_dataset
            else:
                dataloader = self.test_dataloader
                dataset = self.test_dataset
            
            eval_loss = []
            for batch_input, batch_y in dataloader:
                    # forward pass
                    pred_y = self.forward_pass(batch_input, batch_length=self.eval_batch_length)

                    # loss across entire trajectory
                    loss = all_mse(pred_y, batch_y, dataset.stds, dataset.means)

                    eval_loss.append(loss.cpu().item())
            
            eval_loss = np.mean(eval_loss)
            self.loss_meter.evals.append(eval_loss)
            
            # delete all variables from GPU memory
            del batch_input, batch_y, pred_y, loss
        return eval_loss

    def plot_evaluation(self, subfolder):
        plt.title('evaluation function evolution')
        eval_itr = np.arange(len(self.loss_meter.evals)) * self.evaluation_freq
        plt.plot(eval_itr, self.loss_meter.evals)
        plt.xlabel('Number of epochs')
        plt.savefig(f'{subfolder}/eval_loss.png')
        plt.close()
        return

    def save(self):
        subfolder = f'results/{self.day}/{self.time}'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        torch.save(self.func.state_dict(), f'{subfolder}/model.pt')
        self.plot_traj(self.start_epoch+self.epochs, subfolder)
        self.plot_loss(subfolder)
        self.plot_lr(subfolder)
        self.plot_evaluation(subfolder)
        self.logger.save_csv(subfolder)
        if self.sigopt:
            self.report_sigopt(subfolder)
        return

    def report_sigopt(self, subfolder):
        sigopt.log_image(f'{subfolder}/loss.png', 'training loss evolution')
        sigopt.log_image(f'{subfolder}/eval_loss.png', 'test loss evolution')
        sigopt.log_image(f'{subfolder}/lr.png', 'learning rate evolution')
        return

    def checkpoint(self):
        subfolder = f'results/{self.day}/{self.time}/{self.epoch}'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        torch.save(self.func.state_dict(), f'{subfolder}/model.pt')
        self.plot_loss(subfolder)
        self.plot_lr(subfolder)
        self.plot_evaluation(subfolder)
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
    
    def get_batch_t(self, dt, batch_length=None):
        if batch_length == None:
            batch_length = self.batch_length

        if type(dt) == torch.Tensor:
            return torch.linspace(0.0,dt[0]*(batch_length-1),batch_length).to(self.device, non_blocking=True).type(self.dtype)
        else:
            return torch.linspace(0.0,dt*(batch_length-1),batch_length).to(self.device, non_blocking=True).type(self.dtype)

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
            