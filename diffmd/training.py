import torch
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
        print(f'Using {self.device} device')

        # TODO: log everything similarly to device into print file
        self.niters = config['niters']
        self.learning_rate = config['learning_rate']
        self.batch_length = config['batch_length']
        self.nbatches = config['nbatches']
        self.nn_width = config['nn_width']
        self.nn_depth = config['nn_depth']
        self.load_folder = config['load_folder']
        self.optimizer_name = config['optimizer']

        self.dataset = Dataset(config)
        self.loss_meter = RunningAverageMeter()
        
        self.nparticles = 2
        self.dim = 1 + (2*4)
        self.printing_freq = 10
        self.plotting_freq = 10
        self.stopping_freq = 500

        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth).to(self.device)
        self.optimizer = self.set_optimizer(self.optimizer_name)
        
        if self.load_folder != None:
            self.func.load_state_dict(torch.load(f'{self.load_folder}/model.pt'))
        

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

                # TODO: add assertion to check right dimensions
                pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE')

                pred_y = torch.cat(pred_y, dim=-1)
                batch_y = torch.swapaxes(torch.swapaxes(torch.cat(batch_y, dim=-1), 0, 2), 1, 2)

                # TODO: train only on specifics and not all of the data
                loss = torch.mean(torch.abs(pred_y - batch_y))
                
                loss.backward() 
                self.optimizer.step()
            
                self.loss_meter.update(loss.item())
                
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
        print(f'Iter: {itr}, running avg elbo: {self.loss_meter.avg}')
        print(f'current loss: {self.loss_meter.val}')
        print('Last iteration took:     ', time.perf_counter() - start_time, flush=True)

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

            pred_y = torch.cat(pred_y, dim=-1)            
            batch_y = torch.swapaxes(torch.swapaxes(torch.cat(batch_y, dim=-1), 0, 2), 1, 2)
            
            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_pos = [6, 7, 8]
            ind_quat = [9, 10, 11, 12]
            
            for i in ind_vel:
                plt.title('velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,:,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,:,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel1.png')
            plt.close()
            
            for i in ind_vel:
                plt.title('velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,:,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,:,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel2.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_angvel1.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_angvel2.png')
            plt.close()
            
            for i in ind_pos:
                plt.title('positions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_pos1.png')
            plt.close()
            
            for i in ind_pos:
                plt.title('positions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_pos2.png')
            plt.close()
            
            for i in ind_quat:
                plt.title('quaternions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_quat1.png')
            plt.close() 
            
            for i in ind_quat:
                plt.title('quaternions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_quat2.png')
            plt.close() 

    def plot_loss(self, subfolder):
        # TODO: add return figure to be plotted into SigOpt
        plt.title('loss function evolution')
        plt.plot(self.loss_meter.losses)
        plt.savefig(f'{subfolder}/loss.png')
        plt.close()
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

    def save(self):
        subfolder = f'results/depth-{self.nn_depth}-width-{self.nn_width}-lr-{self.learning_rate}'
        if not os.path.exists(f'{subfolder}'):
            os.makedirs(f'{subfolder}')
        torch.save(self.func.state_dict(), f'{subfolder}/model.pt')
        self.plot_traj(self.itr, subfolder)
        self.plot_loss(subfolder)
        return


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
        self.reset()
        self.checkpoints = []

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.log(val)
    
    def log(self, val):
        self.losses.append(val)

    def checkpoint(self):
        self.checkpoints.append(self.avg)


