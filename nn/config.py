import os
import shutil
import wandb
import torch
from diffmd.utils import read_yaml, get_run_ID, count_parameters

class Config():
    def __init__(self, config):
        if isinstance(config, str):
            self.file = config
            config = read_yaml(self.file)
        self.__dict__.update(config)

        # constants
        self.nparticles = 2
        self.dim = 3 + (2*4)

        # additional (computed) parameters
        self.compute_params()
        
        if not self.sweep:
            print('Config:')
            for key, value in self.__dict__.items():
                print(f'{key}: {value}')

    def assign_sweep_config(self, values):
        for key, value in values.items():
            self.__dict__[key] = value

        # nn widths
        self.nn_widths = [self.nn_width] * self.nn_depth
        # additional (computed) parameters
        self.compute_params()

        # update wandb config
        wandb.config.update(self.__dict__, allow_val_change=True)

        print('Config of Current Run in the Sweep:')
        for key, value in self.__dict__.items():
            print(f'{key}: {value}')

    def assign_folders(self, run_name=None):
        if self.wandb:
            self.day, _ = get_run_ID()
            self.subfolder = os.path.join(wandb.run.dir, "output")
        else:
            print('No wandb, creating folders manually')
            if self.load_folder is None:
                self.day, self.time = get_run_ID()
                self.subfolder = self.get_subfolder() 
            else:
                self.day, self.time = self.load_folder.split('/')[-3:-1]
                self.subfolder = '/'.join(self.load_folder.split('/')[:-1])
        os.makedirs(self.subfolder, exist_ok=True)
            
    def compute_params(self):
        self.traj_steps = self.dataset_steps * self.steps_per_dt
        self.learning_rate = 10**(self.log_lr)
        self.weight_decay = 10**(self.log_weight)
    
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

    def save_config(self):
        shutil.copyfile(self.file, f'{self.subfolder}/config.yml')