import sys
import os
import shutil

from plotter import Plotter


sys.path.insert(0, os.path.abspath('..'))
from diffmd.training import Trainer
from diffmd.utils import read_yaml

os.chdir("..")

if __name__ == '__main__':
    config = read_yaml('config.yml')
    # config['folder'] = 'dataset/archive/oscillation'
    config['load_folder'] = 'results/2022-10-01/10-54-15/1100'
 
    trainer = Trainer(config)

    dataset_steps = 1
    plotter = Plotter(trainer, dataset_steps)
    
    
    # plotter.LAMMPS_energy_plot(500)
    # plotter.NN_energy(500)
    plotter.plot_parity()
    plotter.plot_pair_potential()
    plotter.plot_hexagon_potential()
    plotter.plot_traj_potential(500)
    plotter.traj_energies()

    # plotter.get_kinetic_energy(plotter.pred_v, plotter.pred_w)

    # TODO: finish this plotting
    # plotter.plot_traj(dataset='test')



