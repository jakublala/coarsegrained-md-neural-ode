import torch
import pandas as pd
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
from diffmd.training import Trainer

natoms = 14

config = dict(
    folder = '../dataset/test',
    # folder = '../dataset/single_temp_small',
    device = torch.device('cpu'), 
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
    eval_traj_step=1,
    load_folder='../results/2022-09-19/19-17-43',
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
    normalize_loss = False,
    sigopt=False,
    )

if __name__ == '__main__':


    num_steps = 500


    V_diff = np.diff(energies[:num_steps] * natoms)
    r_diff = np.diff(torch.norm(x[:num_steps, 0, :], dim=-1).numpy())
    actual_force = - V_diff / r_diff

    plt.plot(actual_force, 'k', label='Actual')

    plt.savefig('force_compare.png')
    plt.close()




    print('before predicted')
    plt.plot(predicted_energies[:num_steps], 'r', alpha=0.5, label='Potential')
    plt.plot(predicted_energies[:num_steps].squeeze() + harmonic_energy[:num_steps].squeeze(), 'r', alpha=1.0, label='Potential + Harmonic')
    plt.plot(harmonic_energy[:num_steps], 'r--', alpha=0.5, label='Harmonic')

    M = 7.0

    kinetic_energy_trans = torch.sum(torch.sum(0.5 * M * v**2, dim=-1), dim=-1)
    
    batch_input, _, _ = trainer.training_dataset[0]
    inertia = batch_input[-1]

    kinetic_energy_rot = torch.sum(torch.sum(0.5 * inertia * w**2, dim=-1), dim=-1)
    
    kinetic_energy = (kinetic_energy_trans + kinetic_energy_rot).detach().cpu().numpy()

    plt.plot(kinetic_energy[:num_steps], 'b-', label='Kinetic')

    total_energy = predicted_energies[:num_steps].squeeze() + harmonic_energy[:num_steps] + kinetic_energy[:num_steps]

    print('before total')
    plt.plot(total_energy, 'k--', alpha=0.8, label='Total Predicted Energy')
    
    plt.legend()
    plt.savefig('energy.png')
    plt.close()




    
    # actual_kinetic_energies = df['kinetic_energy'].to_numpy() * natoms
    # plt.plot(actual_kinetic_energies[:num_steps], 'r--', alpha=0.5, label='Actual Kinetic Energy')

    # predicted_kinetic_energies = kinetic_energy
    # plt.plot(predicted_kinetic_energies[:num_steps], 'r', alpha=1.0, label='Predicted Kinetic Energy')
    # plt.plot(kinetic_energy_trans.detach().cpu().numpy()[:num_steps], 'b', alpha=0.5, label='Predicted Trans Kinetic')
    # plt.plot(kinetic_energy_rot.detach().cpu().numpy()[:num_steps], 'g', alpha=0.5, label='Predicted Rot Kinetic')

    actual_potential_energies = df['potential_energy'].to_numpy() * natoms
    plt.plot(actual_potential_energies[:num_steps], 'b', label='Actual Potential Energy')

    # actual_total_energies = df['total_energy'].to_numpy() * natoms
    # plt.plot(actual_total_energies[:num_steps], 'k--', label='Actual Total Energy')

    plt.legend()
    plt.savefig('actual_energy.png')
    plt.close()

