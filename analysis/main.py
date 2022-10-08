import sys
import os
import shutil
import torch

from plotter import Plotter


sys.path.insert(0, os.path.abspath('..'))
from diffmd.trainers import NODETrainer, BaselineTrainer
from diffmd.utils import read_yaml

os.chdir("..")

if __name__ == '__main__':
    config = read_yaml('config.yml')
    # config['folder'] = 'dataset/archive/oscillation'
    config['load_folder'] = 'results/2022-10-08/11-43-53/25'
    config['device'] = 'cpu'
 
    trainer = NODETrainer(config)
    for p in trainer.func.net.parameters():
        if p.requires_grad:
            print(p)

    # dataset_steps = 1
    # plotter = Plotter(trainer, dataset_steps)
    # plotter.traj_distribution()
    # plotter.LAMMPS_energy_plot(500)
    # plotter.NN_energy(500)
    # plotter.plot_parity()
    # plotter.plot_pair_potential()
    # plotter.plot_hexagon_potential()
    # plotter.plot_traj_potential(1000)
    # plotter.traj_energies()

    # plotter.get_kinetic_energy(plotter.pred_v, plotter.pred_w)

    # TODO: finish this plotting
    # plotter.plot_traj(dataset='train')



