import torch
torch.cuda.empty_cache()
import wandb
from diffmd.utils import read_yaml
from nn.trainers import NODETrainer, BaselineTrainer

import os

def run():
    trainer = NODETrainer(config_file)
    trainer.train()
    del trainer

if __name__ == '__main__':
    sweep_config = read_yaml('sweep.yml')
    config_file = 'config.yml'
    config = read_yaml(config_file)

    if config['sweep']:
        if config['sweep_id'] is None:
            print('Starting new sweep')
            sweep_id = wandb.sweep(sweep_config, project=config['project'])
        else:
            print('Resuming sweep')
            sweep_id = config['sweep_id']
        wandb.agent(sweep_id=sweep_id, project=config['project'], function=run, count=config['n_count'])
    else:
        run()