import torch
import torch.nn as nn
from diffmd.trainer_base import Trainer

class NODETrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

    def forward_pass(self, batch_input, batch_y, batch_energy):
        # forward pass                
        pred_y = self.predict_traj(batch_input, self.traj_steps, self.steps_per_dt)

        # compute loss
        if self.loss_func_name == 'energy':
            if self.parallel:
                loss = self.loss_func(self.func.module.net, pred_y, batch_energy)
            else:
                loss = self.loss_func(self.func.net, pred_y, batch_energy)
            self.loss_parts = [0 for i in range(4)] + [loss.item()]
        elif self.loss_func_name == 'final-mse-pos-and-energy':
            raise NotImplementedError
            loss, self.loss_parts = final_mse_pos_and_energy(self.func.net, batch_energy, pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means, self.normalize_loss, 1e-6)
        else:
            loss, self.loss_parts = self.loss_func(pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means, self.normalize_loss)
            self.loss_parts += [0]
        return loss






class BaselineTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.training_dataset.update(0)
        self.test_dataset.update(0)
        self.validation_dataset.update(0)
        
        self.func = BaselineModel(self.nparticles, self.dim, self.nn_widths, self.activation_functions, self.dtype).to(self.device, non_blocking=True).type(self.dtype)
    
    def forward_pass(self, batch_input, batch_y, batch_energy):  
        # forward pass
        batch_y0, _, _, _, _ =  batch_input
        v, w, x, q = tuple(i.to(self.device, non_blocking=True).type(self.dtype) for i in torch.split(batch_y0, [3, 3, 3, 4], dim=-1))
        r = x[:, 1, :] - x[:, 0, :]
        rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, self.dim)
        pred_energy = self.func(rq)
        
        loss = torch.mean((pred_energy - batch_energy)**2)
        self.loss_parts = [loss.item(), 0, 0, 0, loss.item()]
        return loss


class BaselineModel(nn.Module):
    def __init__(self, nparticles, dim, widths, functions, dtype):
        super(BaselineModel, self).__init__()
        self.dim = dim
        self.nparticles = nparticles
        self.dtype = dtype
        self.mass = 7.0 # HACK
        self.kwargs = {'dim': dim, 'widths': widths, 'functions': functions, 'dtype': dtype}

        # define neural net
        depth = len(widths) 
        layers = []
        # first layer takes in all configurational variables (xyz and quaternions)
        layers += [nn.Linear(self.dim, widths[0]), functions[0]]
        for i, width in enumerate(widths):
            if i == (depth-1):  
                # last layer outputs a single potential energy value
                layers += [nn.Linear(width, 1)]
            else:
                layers += [nn.Linear(widths[i], widths[i+1]), functions[i+1]]        
        self.net = nn.Sequential(*layers).type(self.dtype)

        # initialise NN parameters
        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,val=0)

    def forward(self, rq):
        return self.net(rq)

