import wandb
from diffmd.utils import read_yaml

class Wandb():
    def __init__(self, config):
        self.run = wandb.init(project='my-test-project', config=config)

        self.sweep_values = dict()
        if config.sweep:
            sweep_config = read_yaml('sweep.yml')
            
            # assign the hyperparameters to the run
            for key, value in sweep_config['parameters'].items():
                self.sweep_values[key] = wandb.config[key]

            # NN depth and width
            self.sweep_values['nn_widths'] = [wandb.config['nn_width'] for i in range(wandb.config['nn_depth'])]    
                    
            self.run.config.update(self.sweep_values)
