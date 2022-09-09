import torch
import pandas as pd
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
from diffmd.training import Trainer

config = dict(
    # folder = 'dataset/single_temp_massive_long_dt',
    folder = '../dataset/single_temp_small',
    device = torch.device('cuda'), 
    training_fraction=1.0,
    random_dataset=False,
    traj_step=1,
    epochs = 100,
    start_epoch = 0,
    optimizer = 'Adamax',
    weight_decay=0,
    batch_length=20,
    batch_length_step=10,
    batch_length_freq=100,
    batch_size=10000,
    shuffle=True,
    num_workers=0,
    learning_rate=0.02,
    nn_widths=[64, 48, 32],
    activation_function='tanh',
    eval_batch_length=1,
    eval_init_skip=1,
    eval_traj_step=100,
    load_folder='../results/temp',
    dtype=torch.float32,
    itr_printing_freq=1,
    printing_freq=1,
    plotting_freq=6000,
    stopping_freq=6000,
    stopping_look_back=2,
    scheduler=None,
    scheduling_factor=0.5,
    scheduling_freq=25,
    evaluation_freq=5,
    checkpoint_freq=2,
    loss_func = 'final-mse-pos',
    sigopt=False,
    )

if __name__ == '__main__':
    trainer = Trainer(config)

    potential = trainer.func.net

    df = pd.read_csv('../dataset/single_temp_small/train/NVE-temp-0.5_K-2.38111026821677_r-2_s-1.csv')

    energies = df['potential_energy'].to_numpy()
    v, w, x, q = torch.split(trainer.training_dataset.data[0], [3, 3, 3, 4], dim=-1)

    r = x[:, 1, :] - x[:, 0, :]
    rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, 11).type(torch.float32)
    
    predicted_energies = potential(rq).detach().cpu().numpy()
    
    indices = np.random.randint(5000, size=rq.shape[0])

    import matplotlib.pyplot as plt
    plt.plot(energies[indices], predicted_energies[indices], 'o')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('energy_matrix.png')
    plt.close()

    num_steps = 10000

    print('before actual')
    plt.plot(energies[:num_steps], 'k--', label='Actual')

    k = 2.38111026821677
    r0 = 2
    harmonic_energy = (0.5 * k * torch.square(torch.norm(r, dim=1) - r0)).detach().cpu().numpy()
    plt.plot(energies[:num_steps] + harmonic_energy[:num_steps], 'k--', alpha=0.5, label='Actual + Harmonic')

    print('before predicted')
    plt.plot(predicted_energies[:num_steps], 'r', label='Predicted')

    M = 7.0
    kinetic_energy_trans = torch.sum(torch.sum(0.5 * M * v**2, dim=-1), dim=-1)
    
    batch_input, _, _ = trainer.training_dataset[0]
    inertia = batch_input[-1]
    kinetic_energy_rot = torch.sum(torch.sum(0.5 * inertia * w, dim=-1), dim=-1)
    
    kinetic_energy = (kinetic_energy_trans + kinetic_energy_rot).detach().cpu().numpy()

    total_energy = predicted_energies[:num_steps].squeeze() + kinetic_energy[:num_steps]

    print('before total')
    plt.plot(total_energy, 'r--', alpha=0.8, label='Total Predicted Energy')
    
    plt.legend()
    plt.savefig('energy.png')
    plt.close()

    # trainer.plot_traj()
