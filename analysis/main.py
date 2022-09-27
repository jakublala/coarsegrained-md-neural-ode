import sys
import os

from parity import parity_plot


sys.path.insert(0, os.path.abspath('..'))
from diffmd.training import Trainer
from diffmd.utils import read_yaml

os.chdir("..")

if __name__ == '__main__':

    if not os.path.exists('figures'):
        os.mkdir('figures')

    config = read_yaml('config.yml')
    config['load_folder'] = 'results/archive/2022-09-25/23-43-55'
    # TODO: add a way to adjust the NN size based on the loaded model

    trainer = Trainer(config)
    parity_plot(trainer)
    
