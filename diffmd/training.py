import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
import shutil
import numpy as np
import sigopt

from data.reader import Reader
from data.dataset import Dataset
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint

class Trainer():

    def __init__(self, config):
        self.folder = config['folder']
        
        self.device = config['device']
        
        # TODO: log everything similarly to device into print file
        self.niters = config['niters']
        self.learning_rate = config['learning_rate']
        self.batch_length = config['batch_length']
        self.nbatches = config['nbatches']
        self.nn_width = config['nn_width']
        self.nn_depth = config['nn_depth']
        self.load_folder = config['load_folder']
        self.optimizer_name = config['optimizer']
        self.scheduler_name = config['scheduler']
        self.scheduling_factor = config['scheduling_factor']
        
        self.dataset = Dataset(config)
        self.loss_meter = RunningAverageMeter()
        
        self.nparticles = 2
        self.dim = 3 + (2*4)
        self.printing_freq = config['printing_freq']
        self.plotting_freq = config['plotting_freq']
        self.stopping_freq = config['stopping_freq']
        self.scheduling_freq = config['scheduling_freq']

        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth).to(self.device)
        self.nparameters = count_parameters(self.func)

        self.optimizer = self.set_optimizer(self.optimizer_name)
        self.scheduler = self.set_scheduler(self.scheduler_name, self.scheduling_factor)
        
        if self.load_folder != None:
            self.func.load_state_dict(torch.load(f'{self.load_folder}/model.pt'))

        print(f'device = {self.device}')
        print(f'depth = {self.nn_depth}, width = {self.nn_width}')
        print(f'number of parameters = {self.nparameters}')
        print(f'learning rate = {self.learning_rate}, optimizer = {self.optimizer_name}')
        print(f'scheduler = {self.scheduler_name}, scheduling factor = {self.scheduling_factor}, scheduling freq = {self.scheduling_freq}')
        print(f'number of batches = {self.nbatches}, batch length = {self.batch_length}')



    def train(self):
        for self.itr in range(1, self.niters + 1):
            start_time = time.perf_counter()
            
            if self.optimizer_name == 'LBFGS':
                # TODO: implement this or another optimizer
                self.optimizer.step(self.closure)
            else:
                
                # zero out gradients with less memory operations
                for param in self.func.parameters():
                    param.grad = None
                
                batch_t, batch_y0, batch_y, self.func.k, self.func.inertia = self.dataset.get_batch(self.nbatches, self.batch_length) 
                
                # if self.device == torch.device('cuda'):
                #     self.func = nn.DataParallel(self.func).to(self.device)

                # TODO: add assertion to check right dimensions
                pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE')
            
                pred_y = torch.swapaxes(torch.cat(pred_y, dim=-1), 0, 1)
                
                batch_y = torch.cat(batch_y, dim=-1)
                
                # TODO: train only on specifics and not all of the data
                loss = torch.mean(torch.abs(pred_y - batch_y))
                
                loss.backward() 
                self.optimizer.step()
            
                self.loss_meter.update(loss.item(), self.optimizer.param_groups[0]["lr"])
            
            if self.itr % self.scheduling_freq == 0:
                self.scheduler.step()

            if self.itr % self.printing_freq == 0:
                self.print_loss(self.itr, start_time)

            if self.itr % self.plotting_freq == 0:
                self.plot_traj(self.itr)

            # early stopping
            if self.itr % self.stopping_freq == 0:

                self.loss_meter.checkpoint()

                # divergent / non-convergent
                if len(self.loss_meter.checkpoints) > 1:
                    if self.loss_meter.checkpoints[-2] < self.loss_meter.checkpoints[-1]:
                        print('early stopping as non-convergent')
                        return self.func, self.loss_meter
                
                # TODO: add proper stale convergence and test it out
                # stale convergence
                # if np.sd(self.loss_meter.losses[-self.stopping_freq:]) > 0.001:
                #     print('early stopping as stale convergence')
                #     return self.func, self.loss_meter

        # TODO add checkpointing
        # TODO: add logging in

        return self.func, self.loss_meter

    def closure(self):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        batch_t, batch_y0, batch_y, self.func.k, self.func.inertia = self.dataset.get_batch(self.nbatches, self.batch_length)  
        pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE') 
        pred_y = torch.cat(pred_y, dim=-1)
        batch_y = torch.swapaxes(torch.swapaxes(torch.cat(batch_y, dim=-1), 0, 2), 1, 2)
        loss = torch.mean(torch.abs(pred_y - batch_y))    
        self.loss_meter.update(loss.item())
        if loss.requires_grad:
            loss.backward()
        return loss


    def print_loss(self, itr, start_time):
        print(f'Iter: {itr}, Running avg elbo: {self.loss_meter.avg}')
        print(f'Current loss: {self.loss_meter.val}')
        print('Last iteration took:     ', time.perf_counter() - start_time, flush=True)
        print(f'Learning rate: {self.loss_meter.lrs[-1]}')

    def plot_traj(self, itr, subfolder='temp'):

        if itr == self.plotting_freq:
            if subfolder == 'temp' and os.path.exists(f'{subfolder}'):
                shutil.rmtree(f'{subfolder}')

            if subfolder == 'temp' and not os.path.exists(f'{subfolder}'):
                os.makedirs(f'{subfolder}')

        if subfolder == 'temp':
            traj_length = 100
        else:
            traj_length = 1000

        with torch.no_grad():
            nbatches = 1
            batch_t, batch_y0, batch_y, self.func.k, self.func.inertia = self.dataset.get_batch(nbatches, traj_length)   

            pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE')

            pred_y = torch.cat(pred_y, dim=-1).cpu().numpy()
            batch_y = torch.swapaxes(torch.cat(batch_y, dim=-1), 0, 1).cpu().numpy()
            batch_t = batch_t.cpu().numpy()
            
            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_pos = [6, 7, 8]
            ind_quat = [9, 10, 11, 12]
            
            for i in ind_vel:
                plt.title('velocities 1')
                plt.plot(batch_t, batch_y[:,:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel1.png')
            plt.close()
            
            for i in ind_vel:
                plt.title('velocities 2')
                plt.plot(batch_t, batch_y[:,:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel2.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities 1')
                plt.plot(batch_t, batch_y[:,:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_angvel1.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities 2')
                plt.plot(batch_t, batch_y[:,:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_angvel2.png')
            plt.close()
            
            # centre of mass positions (set initial position of first COM to zero)
            batch_y[:,:,:,6:9] = batch_y[:,:,:,6:9] - batch_y[0,:,0,6:9]
            pred_y[:,:,:,6:9] = pred_y[:,:,:,6:9] - pred_y[0,:,0,6:9]
            
            for i in ind_pos:
                plt.title('positions 1')
                plt.plot(batch_t, batch_y[:,:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_pos1.png')
            plt.close()
            
            for i in ind_pos:
                plt.title('positions 2')
                plt.plot(batch_t, batch_y[:,:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_pos2.png')
            plt.close()

            # centre of mass separation
            batch_y_sep = np.linalg.norm(batch_y[:,:,1,6:9] - batch_y[:,:,0,6:9], axis=2)
            pred_y_sep = np.linalg.norm(pred_y[:,:,1,6:9] - pred_y[:,:,0,6:9], axis=2)

            plt.title('separation')
            plt.plot(batch_t, batch_y_sep, 'k--', alpha=0.3, label=f'true')
            plt.plot(batch_t, pred_y_sep, 'r-', alpha=0.5, label=f'pred')
            plt.savefig(f'{subfolder}/{itr}_sep.png')
            plt.close()

            # quaternions
            for i in ind_quat:
                plt.title('quaternions 1')
                plt.plot(batch_t, batch_y[:,:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_quat1.png')
            plt.close() 
            
            for i in ind_quat:
                plt.title('quaternions 2')
                plt.plot(batch_t, batch_y[:,:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_quat2.png')
            plt.close() 

    def plot_loss(self, subfolder):
        # TODO: add return figure to be plotted into SigOpt
        plt.title('loss function evolution')
        plt.plot(self.loss_meter.losses)
        plt.savefig(f'{subfolder}/loss.png')
        plt.close()
        return

    def plot_lr(self, subfolder):
        plt.title('learning rate evolution')
        plt.plot(self.loss_meter.lrs)
        plt.savefig(f'{subfolder}/lr.png')
        plt.close()
        return

    def log_hyperparameters(self, subfolder):
        with open(f'{subfolder}/hyperparameters.txt', 'w') as f:
            f.write(f'device = {self.device} \n')
            f.write(f'depth = {self.nn_depth}, width = {self.nn_width} \n')
            f.write(f'number of parameters = {self.nparameters} \n')
            f.write(f'learning rate = {self.learning_rate}, optimizer = {self.optimizer_name} \n')
            f.write(f'scheduler = {self.scheduler_name}, scheduling factor = {self.scheduling_factor}, scheduling freq = {self.scheduling_freq} \n')
            f.write(f'number of batches = {self.nbatches}, batch length = {self.batch_length} \n')

        return

    def set_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return torch.optim.Adam(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'AdamW':
            return torch.optim.AdamW(self.func.parameters(), lr=self.learning_rate)
        elif optimizer == 'LBFGS':
            return torch.optim.LBFGS(self.func.parameters(), lr=self.learning_rate)
        else:
            raise Exception('optimizer not implemented')

    def set_scheduler(self, scheduler, alpha):
        if scheduler == 'LambdaLR':
            lambda1 = lambda epoch: alpha ** epoch
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        else:
            raise Exception('scheduler not implemented')

    def save(self):
        subfolder = f'results/depth-{self.nn_depth}-width-{self.nn_width}-lr-{self.learning_rate}'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        torch.save(self.func.state_dict(), f'{subfolder}/model.pt')
        self.plot_traj(self.itr, subfolder)
        self.plot_loss(subfolder)
        self.plot_lr(subfolder)
        self.log_hyperparameters(subfolder)
        return

    


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
        self.reset()
        self.checkpoints = []
        self.lrs = []

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


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
