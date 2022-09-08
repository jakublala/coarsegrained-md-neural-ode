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
    load_folder='../results/archive/2022-08-11/12-39-15',
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
    _, _, x, q = torch.split(trainer.training_dataset.data[0], [3, 3, 3, 4], dim=-1)

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

    plt.plot(energies[:20000], 'k--', label='Actual')
    plt.plot(predicted_energies[:20000], 'r', label='Predicted')
    plt.legend()
    plt.savefig('energy.png')
    plt.close()

    trainer.plot_traj()
