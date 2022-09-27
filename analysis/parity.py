import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


def parity_plot(trainer):
    # TODO: add seaborn and play around with it
    natoms = 14
    trajectory_index = 0
    df = pd.read_csv(trainer.training_dataset.trajs[trajectory_index].file_path+'.csv')
    data = trainer.training_dataset.trajs[trajectory_index].traj
    data = [i.squeeze() for i in data]
    v, w, x, q = data
    potential = trainer.func.net
    energies = df['potential_energy'].to_numpy()

    r = x[:, 1, :] - x[:, 0, :]
    rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, 11).type(torch.float32)
    
    predicted_energies = potential(rq).detach().cpu().numpy()    
    indices = np.random.randint(500, size=rq.shape[0])


    fig, ax = plt.subplots()
    ax.plot(energies[indices] * natoms, predicted_energies[indices], 'bo')
    ax.plot([-4, 4], [-4, 4], 'k-')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # set equal aspect ratio
    ax.set_aspect('equal')

    min_y = predicted_energies[indices].min()
    max_y = predicted_energies[indices].max()
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([min_y, max_y])
    fig.savefig('figures/energy_matrix.png')
    plt.close(fig)

