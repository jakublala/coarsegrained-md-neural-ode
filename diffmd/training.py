import torch
import matplotlib.pyplot as plt
import time
import os
import shutil

from data.reader import Reader
from data.training import get_batch_mod
from data.dataset import Dataset
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint

class Trainer():

    def __init__(self, config):
        self.filename = config['filename']
        
        self.device = config['device']
        print(f'Using {self.device} device')

        # TODO: log everything similarly to device into print file
        self.niters = config['niters']
        self.learning_rate = config['learning_rate']
        self.batch_length = config['batch_length']
        self.nbatches = config['nbatches']

        self.dataset = Dataset(config)
        self.loss_meter = RunningAverageMeter()
        
        self.nparticles = 2
        self.dim = 1 + (2*4)
        self.nn_width = 50
        self.nn_depth = 1

        self.func = ODEFunc(self.nparticles, self.dataset.inertia, self.dataset.k, self.dim, self.nn_width, self.nn_depth).to(self.device)
        self.optimizer = self.set_optimizer(config['optimizer'])
        

    def train(self):
        for itr in range(1, self.niters + 1):
            start_time = time.perf_counter()
            
            # zero out gradients with less memory operations
            for param in self.func.parameters():
                param.grad = None
            
            batch_t, batch_y0, batch_y = get_batch_mod(self.dataset.traj, self.nbatches, self.batch_length, self.dataset.dt, self.device)   
            
            # TODO: add assertion to check right dimensions
            pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE')

            pred_y = torch.cat(pred_y, dim=-1)
            batch_y = torch.swapaxes(torch.swapaxes(torch.cat(batch_y, dim=-1), 0, 2), 1, 2)

            # TODO: train only on specifics and not all of the data   
            loss = torch.mean(torch.abs(pred_y - batch_y))
            
            loss.backward() 
            self.optimizer.step()
            self.loss_meter.update(loss.item())
            
            if itr % 10 == 0:
                self.print_loss(itr, start_time)

            if itr % 100 == 0:
                self.plot_traj(itr)
        
        # TODO add checkpointing
        # TODO: add logging in and model saving

        return self.func, self.loss_meter

    def print_loss(self, itr, start_time):
        print('Iter: {}, running avg elbo: {:.4f}'.format(itr, self.loss_meter.avg))
        print('current loss: {:.4f}'.format(self.loss_meter.val))
        print('Last iteration took:     ', time.perf_counter() - start_time)

    def plot_traj(self, itr):

        if itr == 100:
            if os.path.exists('temp'):
                shutil.rmtree('temp')

            if not os.path.exists('temp'):
                os.makedirs('temp')

        traj_length = 100
        with torch.no_grad():
            batch_t, batch_y0, batch_y = get_batch_mod(self.dataset.traj, 1, traj_length, self.dataset.dt, self.device)   

            pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE')

            pred_y = torch.cat(pred_y, dim=-1)            
            batch_y = torch.swapaxes(torch.swapaxes(torch.cat(batch_y, dim=-1), 0, 2), 1, 2)
            
            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_pos = [6, 7, 8]
            ind_quat = [9, 10, 11, 12]
            
            for i in ind_vel:
                plt.title('velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_vel1.png')
            plt.close()
            
            for i in ind_vel:
                plt.title('velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_vel2.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_angvel1.png')
            plt.close()
            
            for i in ind_ang:
                plt.title('angular velocities')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_angvel2.png')
            plt.close()
            
            for i in ind_pos:
                plt.title('positions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_pos1.png')
            plt.close()
            
            for i in ind_pos:
                plt.title('positions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_pos2.png')
            plt.close()
            
            for i in ind_quat:
                plt.title('quaternions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,0,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,0,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_quat1.png')
            plt.close() 
            
            for i in ind_quat:
                plt.title('quaternions')
                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:,1,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:,1,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'temp/{itr}_quat2.png')
            plt.close() 

    def set_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return torch.optim.Adam(self.func.parameters(), lr=self.lr)
        else:
            raise Exception('optimizer not implemented')
    
    # losses_log.append(loss_meter.losses)



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
        self.reset()

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


