import torch
from data.reader import Reader
from data.training import get_data, process_data, process_inertia
from data.dataset import Dataset
from diffmd.diffeqs import ODEFunc

class Trainer():

    def __init__(self, config):
        self.filename = config['filename']
        self.device = config['device']

        self.n_epochs = config['n_epochs']
        self.lr = config['lr']
        self.optimizer = config['optimizer']
        self.batch_length = config['batch_length']
        self.n_batches = config['n_batches']

        self.dataset = Dataset(self.filename)
        self.loss_meter = RunningAverageMeter()
        
        self.nparticles = 2
        self.dim = 1 + (2*4)
        self.nn_width = 50
        self.nn_depth = 1

        self.func = ODEFunc(self.nparticles, self.dataset.inertia, self.dataset.k, self.dim, self.nn_width, self.nn_depth).to(self.device)






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


