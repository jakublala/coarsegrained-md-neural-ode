import sys
import os
import argparse
import shutil
import torch
import wandb

from plotter import Plotter


sys.path.insert(0, os.path.abspath('..'))
from nn.trainers import NODETrainer, BaselineTrainer
from diffmd.utils import read_yaml

os.chdir("..")

if __name__ == '__main__':
    # parser arguments
    parser = argparse.ArgumentParser(
                    prog = 'CG Neural ODE: Analysis',
                    description = 'Plots various quantities from the NN training',
                    epilog = 'Enjoy the program! :)')
    parser.add_argument('-id', '--run_id', type=str, default=None, help='Run ID to plot')    
    parser.add_argument('-p', '--project', type=str, default='cg-node-hexagon', help='Project name')
    args = parser.parse_args()
    run_id = args.run_id
    project_name = args.project

    # HACK
    user = 'jakublala'

    # download checkpoints
    # TODO: how to deal with early stopping, i.e when there is not all checkpoints?
    if os.path.exists(f'temp/{run_id}'):
        shutil.rmtree(f'temp/{run_id}')
    os.makedirs(f'temp/{run_id}')
    os.chdir(f'temp/{run_id}')
    wandb.restore('output/config.yml', run_path=f"{user}/{project_name}/{run_id}")
    config = read_yaml('output/config.yml')
    for i in range(config['checkpoint_freq'], config['epochs'], config['checkpoint_freq']):
        wandb.restore(f'output/checkpoints/{i}/model.pt', run_path=f"{user}/{project_name}/{run_id}")
    wandb.restore(f"output/checkpoints/{config['epochs']}/model.pt", run_path=f"{user}/{project_name}/{run_id}")
    os.chdir('../..')

    # edit config for plotting purposes
    config['wandb'] = False
    config['sweep'] = False
    config['sweep_id'] = None
    config['load_folder'] = f'temp/{run_id}/output/checkpoints/{config["epochs"]}'
    config['analysis'] = True

    trainer = NODETrainer(config)

    dataset_steps = 1
    plotter = Plotter(trainer, run_id, dataset_steps)
    plotter.traj_distribution()
    plotter.LAMMPS_energy_plot(100)
    plotter.NN_energy(100)
    plotter.plot_parity()
    plotter.plot_traj_potential(100)
    plotter.traj_energies()
    plotter.get_kinetic_energy(plotter.pred_v, plotter.pred_w)

    # currently not working
    # plotter.plot_pair_potential()
    # plotter.plot_hexagon_potential()
    

    # # TODO: finish this plotting
    plotter.plot_traj(dataset='train')

    # delete temp folder
    shutil.rmtree(f'temp/{run_id}')



