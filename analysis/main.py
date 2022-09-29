import sys
import os
import shutil

from plotter import Plotter


sys.path.insert(0, os.path.abspath('..'))
from diffmd.training import Trainer
from diffmd.utils import read_yaml

os.chdir("..")

if __name__ == '__main__':

    if not os.path.exists('figures'):
        os.mkdir('figures')



    if not os.path.exists("figures/energies"):
        os.makedirs('figures/energies')

    if not os.path.exists("figures/trajs"):
        os.makedirs('figures/trajs')


    config = read_yaml('config.yml')
    config['load_folder'] = 'results/2022-09-29/13-17-29/50'
 
    trainer = Trainer(config)
    plotter = Plotter(trainer)
    plotter.LAMMPS_energy_plot(500)
    plotter.plot_parity()
    plotter.plot_pair_potential()
    plotter.plot_hexagon_potential()
    plotter.plot_traj_potential(500)

    # TODO: finish this plotting
    # plotter.plot_traj(dataset='test')



