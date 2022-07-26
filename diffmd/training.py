from distutils.command.config import config
from this import d
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
from diffmd.utils import get_run_ID, count_parameters

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
        self.eval_batch_length = config['eval_batch_length']
        self.batch_size = config['batch_size']
        self.nn_width = config['nn_width']
        self.nn_depth = config['nn_depth']
        self.load_folder = config['load_folder']
        self.loss_func_name = config['loss_func']
        self.optimizer_name = config['optimizer']
        self.scheduler_name = config['scheduler']
        self.scheduling_factor = config['scheduling_factor']
        
        self.dtype = config['dtype']

        # dataset setup
        self.training_dataset = Dataset(config, dataset_type='train', batch_length=self.batch_length)
        self.test_dataset = Dataset(config, dataset_type='test', batch_length=self.eval_batch_length)
        self.validation_dataset = Dataset(config, dataset_type='validation', batch_length=self.eval_batch_length)
        self.training_dataloader = self.get_dataloader(config, self.training_dataset) 
        self.test_dataloader = self.get_dataloader(config, self.test_dataset) 
        self.validation_dataloader = self.get_dataloader(config, self.validation_dataset)
        
        self.loss_meter = RunningAverageMeter()
        self.time_meter = TimeMeter()
        
        self.nparticles = 2
        self.dim = 1 + (2*4)
        self.printing_freq = config['printing_freq']
        self.itr_printing_freq = config['itr_printing_freq']
        self.plotting_freq = config['plotting_freq']
        self.stopping_freq = config['stopping_freq']
        self.scheduling_freq = config['scheduling_freq']
        self.evaluation_freq = config['evaluation_freq']
        self.checkpoint_freq = config['checkpoint_freq']

        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth, self.dtype).to(self.device)
        self.nparameters = count_parameters(self.func)

        self.loss_func = self.set_loss_func(self.loss_func_name)
        self.optimizer = self.set_optimizer(self.optimizer_name)
        self.scheduler = self.set_scheduler(self.scheduler_name, self.scheduling_factor)
        
        if self.load_folder != None:
            self.func.load_state_dict(torch.load(f'{self.load_folder}/model.pt'))
        print(f'device = {self.device}')
        print(f'depth = {self.nn_depth}, width = {self.nn_width}')
        print(f'number of parameters = {self.nparameters}')
        print(f'learning rate = {self.learning_rate}, optimizer = {self.optimizer_name}')
        print(f'scheduler = {self.scheduler_name}, scheduling factor = {self.scheduling_factor}, scheduling freq = {self.scheduling_freq}')
        print(f'batch size = {self.batch_size}, traj length = {self.batch_length}')

    def forward_pass(self, batch_input, batch_y, batch_length=None):
        batch_y = batch_y.to(self.device).type(self.dtype)
        
        batch_y0, dt, k, r0, inertia =  batch_input
        batch_y0 = tuple(i.to(self.device).type(self.dtype) for i in torch.split(batch_y0, [3, 3, 3, 4], dim=-1))

        # get timesteps
        # TODO: this is always the same and so we can make it simpler
        batch_t = self.get_batch_t(dt, batch_length)
        
        # set constants
        self.func.k = k.to(self.device).type(self.dtype)
        self.func.r0 = r0.to(self.device).type(self.dtype)
        self.func.inertia = inertia.to(self.device).type(self.dtype)
        options = dict(inertia=inertia.to(self.device).type(self.dtype))
        
        # TODO: add assertion to check right dimensions
        pred_y = odeint_adjoint(self.func, batch_y0, batch_t, method='NVE', options=options)
        pred_y = torch.swapaxes(torch.cat(pred_y, dim=-1), 0, 1)
        
        return pred_y
        
    def train(self):
        for self.epoch in range(self.start_epoch + 1, (self.start_epoch + self.epochs) + 1):
            self.start_time = time.perf_counter()
            
            # zero out gradients with less memory operations
            for param in self.func.parameters():
                param.grad = None

            for self.itr, (batch_input, batch_y) in enumerate(self.training_dataloader):
                self.itr_start_time = time.perf_counter()

                # forward pass
                pred_y = self.forward_pass(batch_input, batch_y)

                # TODO: train only on specifics and not all of the data
                loss = self.loss_func(pred_y, batch_y)

                # backward pass                    
                loss.backward() 
                self.optimizer.step()

                # for p in self.func.parameters():
                #     print(p.grad)
            
                self.loss_meter.update(loss.item(), self.optimizer.param_groups[0]["lr"])
                
                if (self.itr+1) % self.itr_printing_freq == 0:
                    self.print_iteration()
                    self.time_meter.update(self.epoch, self.itr, time.perf_counter() - self.itr_start_time)
                     
            if self.after_epoch():
                # if True, then early stopping
                return self.func, self.loss_meter
        
        return self.func, self.loss_meter

    def after_epoch(self):

        if self.epoch % self.scheduling_freq == 0 and self.scheduler_name != None:
                self.scheduler.step()

        if self.epoch % self.printing_freq == 0:
            self.print_loss(self.epoch, self.start_time)

        if self.epoch % self.plotting_freq == 0:
            self.plot_traj(self.epoch)

        if self.epoch % self.evaluation_freq == 0:
            self.evaluate(validate=False)

        if self.epoch % self.checkpoint_freq == 0:
            self.checkpoint()

        # early stopping
        if self.epoch % self.stopping_freq == 0:

            self.loss_meter.checkpoint()
            
            # divergent / non-convergent
            if len(self.loss_meter.checkpoints) > 2:
                if self.loss_meter.checkpoints[-2] < self.loss_meter.checkpoints[-1]:
                    print('early stopping as non-convergent')
                    # TODO: make this compliant with multiple GPUs
                    return True
            # TODO: add proper stale convergence and test it out
            # stale convergence
            # if np.sd(self.loss_meter.losses[-self.stopping_freq:]) > 0.001:
            #     print('early stopping as stale convergence')
            #     return self.func, self.loss_meter
        

    def print_loss(self, itr, start_time):
        print(f'Epoch: {itr}, Running avg elbo: {self.loss_meter.avg}')
        print(f'Current loss: {self.loss_meter.val}')
        print('Last epoch took:     ', time.perf_counter() - start_time, flush=True)
        print(f'Learning rate: {self.loss_meter.lrs[-1]}')
        
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
        self.training_dataset.batch_length = batch_length

        with torch.no_grad():
            batch_input, batch_y = self.training_dataset[0]
            batch_input = list(batch_input)
            batch_input[0] = batch_input[0].unsqueeze(0)
            batch_input = tuple(batch_input)
            
            pred_y = self.forward_pass(batch_input, batch_y, batch_length=batch_length).squeeze().cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            batch_t = self.get_batch_t(batch_input[1], batch_length=batch_length).cpu().numpy()

            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_pos = [6, 7, 8]
            ind_quat = [9, 10, 11, 12]
            
            for i in ind_vel:
                plt.title('velocities 1')
                plt.plot(batch_t, batch_y[:,0,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,0,i], 'r-', alpha=0.5, label=f'pred {i}')
            plt.savefig(f'{subfolder}/{itr}_vel1.png')
            plt.close()
            
            for i in ind_vel:
                plt.title('velocities 2')
                plt.plot(batch_t, batch_y[:,1,i], 'k--', alpha=0.3, label=f'true {i}')
                plt.plot(batch_t, pred_y[:,1,i], 'r-', alpha=0.5, label=f'pred {i}')
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

    def log_hyperparameters(self, subfolder):
        with open(f'{subfolder}/hyperparameters.txt', 'w') as f:
            f.write(f'device = {self.device} \n')
            f.write(f'training datasets = {self.training_dataset.filenames} in {self.folder} \n')
            f.write(f'depth = {self.nn_depth}, width = {self.nn_width} \n')
            f.write(f'number of parameters = {self.nparameters} \n')
            f.write(f'learning rate = {self.learning_rate}, optimizer = {self.optimizer_name} \n')
            f.write(f'scheduler = {self.scheduler_name}, scheduling factor = {self.scheduling_factor}, scheduling freq = {self.scheduling_freq} \n')
            f.write(f'number of batches = {self.batch_size}, traj length = {self.batch_length} \n')
            f.write(f'loss function = {self.loss_func_name} \n')
        return

    def set_loss_func(self, loss_func):
        if loss_func == 'all':
            return all_loss_func
        elif loss_func == 'final':
            return final_loss_func
        else:
            raise ValueError(f'loss function {loss_func} not recognised')

        
    def get_dataloader(self, config, dataset, no_batch=False):
        # TODO: make this cleaner
        if no_batch:
            params = {'batch_size': 1, 'shuffle':config['shuffle'], 'num_workers':config['num_workers']}
        else:
            params = {'batch_size': config['batch_size'], 'shuffle':config['shuffle'], 'num_workers':config['num_workers']}
        
        return torch.utils.data.DataLoader(dataset, **params)

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
            # lambda1 = lambda epoch: alpha ** epoch
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif scheduler == None:
            return
        else:
            raise Exception('scheduler not implemented')

    def evaluate(self, validate=False):
        with torch.no_grad():
            # TODO: maybe move this elsewhere and make it more robust? maybe have an Evaluation class
            if validate:
                dataloader = self.validation_dataloader
            else:
                dataloader = self.test_dataloader

            eval_loss = []
            for batch_input, batch_y in dataloader:
                    # forward pass
                    pred_y = self.forward_pass(batch_input, batch_y, batch_length=self.eval_batch_length).cpu()

                    # loss across entire trajectory
                    loss = torch.mean(torch.abs(pred_y - batch_y))

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
        self.log_hyperparameters(subfolder)
        self.log_loss(subfolder)
        self.log_eval(subfolder)
        self.log_lr(subfolder)
        if self.sigopt:
            self.report_sigopt(subfolder)

        print(self.time_meter.get_array())
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

    def log_loss(self, subfolder):
        with open(f'{subfolder}/loss.txt', 'w') as f:
            for loss in self.loss_meter.losses:
                f.write(f'{loss}\n')
        return

    def log_eval(self, subfolder):
        with open(f'{subfolder}/eval.txt', 'w') as f:
            for eval in self.loss_meter.evals:
                f.write(f'{eval}\n')
        return

    def log_lr(self, subfolder):
        with open(f'{subfolder}/lr.txt', 'w') as f:
            for lr in self.loss_meter.lrs:
                f.write(f'{lr}\n')
        return

    def get_batch_t(self, dt, batch_length=None):
        if batch_length == None:
            batch_length = self.batch_length

        if type(dt) == torch.Tensor:
            return torch.linspace(0.0,dt[0]*(batch_length-1),batch_length).to(self.device).type(self.dtype)
        else:
            return torch.linspace(0.0,dt*(batch_length-1),batch_length).to(self.device).type(self.dtype)


class TimeMeter(object):

    def __init__(self):
        self.times = []

    def update(self, epoch, itr, time):
        self.times.append([epoch, itr, time])

    def get_array(self):
        return np.array(self.times)


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

# Types of loss functions
def all_loss_func(pred_y, true_y):
    return torch.mean(torch.abs(pred_y - true_y))

def final_loss_func(pred_y, true_y):
    return torch.mean(torch.abs(pred_y[:, -1, :, :] - true_y[:, -1, :, :]))

def lambda_lr(epoch):
    return 0.95 ** epoch
            