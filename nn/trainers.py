import torch
from nn.trainer_base import Trainer
from diffmd.baseline import BaselineModel

class NODETrainer(Trainer):

    def __init__(self, config_file):
        super().__init__(config_file)

    def forward_pass(self, batch_input, batch_y, batch_energy):
        # forward pass                
        pred_y = self.predict_traj(batch_input, self.config.traj_steps, self.config.steps_per_dt)

        # compute loss
        if self.config.loss_func == 'energy':
            if self.parallel:
                loss = self.loss_func(self.func.module.net, pred_y, batch_energy, self.config.normalize_loss)
            else:
                loss = self.loss_func(self.func.net, pred_y, batch_energy, self.config.normalize_loss)
            self.loss_parts = [0 for i in range(4)] + [loss.item()]
        elif self.config.loss_func == 'final-mse-pos-and-energy':
            raise NotImplementedError
            loss, self.loss_parts = final_mse_pos_and_energy(self.func.net, batch_energy, pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means, self.config.normalize_loss, 1e-6)
        else:
            loss, self.loss_parts = self.loss_func(pred_y, batch_y, self.training_dataset.stds, self.training_dataset.means, self.config.normalize_loss)
            self.loss_parts += [0]
        return loss

class BaselineTrainer(Trainer):

    def __init__(self, config_file):
        super().__init__(config_file)

        self.training_dataset.update(0)
        self.test_dataset.update(0)
        self.validation_dataset.update(0)
        
        self.func = BaselineModel(self.nparticles, self.dim, self.nn_widths, self.activation_functions, self.dtype).to(self.device, non_blocking=True).type(self.dtype)
    
    def forward_pass(self, batch_input, batch_y, batch_energy):  
        # forward pass
        batch_y0, _, _, _, _ =  batch_input
        batch_y0 = tuple(i.to(self.device, non_blocking=True).type(self.dtype) for i in torch.split(batch_y0, [3, 3, 3, 4], dim=-1))
        pred_energy = self.func(batch_y0)
        
        loss = torch.mean((pred_energy - batch_energy)**2)
        self.loss_parts = [loss.item(), 0, 0, 0, loss.item()]
        return loss
