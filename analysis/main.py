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



    if os.path.exists("figures/energies"):
        shutil.rmtree('figures/energies')
    os.makedirs('figures/energies')


    config = read_yaml('config.yml')
    config['load_folder'] = 'results/archive/2022-09-27/12-01-08/1000'
    # TODO: add a way to adjust the NN size based on the loaded model

    trainer = Trainer(config)
    plotter = Plotter(trainer)
    plotter.LAMMPS_energy_plot(500)
    plotter.plot_parity()
    plotter.plot_pair_potential()
    plotter.plot_hexagon_potential()
    plotter.plot_traj_potential(500)



