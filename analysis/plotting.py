
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.offsetbox import AnchoredText
from matplotlib.gridspec import GridSpec
import seaborn as sns

torch.cuda.empty_cache()
from diffmd.trainer_base import Trainer

def load_trainer(folder=None):
    config = dict(
        # folder = 'dataset/approach',
        # folder = 'dataset/oscillation',
        folder = 'dataset/single_temp_small', 
        device = torch.device('cpu'), 
        epochs = 1,
        start_epoch = 0,
        optimizer = 'Adam',
        batch_length=40,
        batch_size=10000,
        batch_length_step=2,
        batch_length_freq=1,
        shuffle=True,
        num_workers=0,
        learning_rate=0,
        nn_widths=[64, 48, 32],
        activation_function='tanh',
        eval_batch_length=100,
        load_folder=folder,
        # load_folder=None,
        dtype=torch.float32,
        itr_printing_freq=1,
        printing_freq=1,
        plotting_freq=1,
        stopping_freq=5,
        stopping_look_back=3,
        scheduler='LambdaLR',
        scheduling_factor=0.95,
        scheduling_freq=10,
        evaluation_freq=5,
        checkpoint_freq=5,
        loss_func = 'final-2',
        sigopt=True,
        )

    return Trainer(config)




fig = plt.figure(figsize=(12,8), dpi=100)
axs = [0 for i in range(6)]
width = 480
height = 600
triple_width = 135
triple_width_delta = (width - 3 * triple_width) // 4
double_width = 220
double_width_delta = (width - 2 * double_width) // 2

gs = GridSpec(height, width, figure=fig)
        
axs[0] = fig.add_subplot(gs[:250, :triple_width])
axs[1] = fig.add_subplot(gs[:250, (triple_width + 2* triple_width_delta):(2*triple_width + 2*triple_width_delta)])
axs[2] = fig.add_subplot(gs[:250, (2*triple_width + 4*triple_width_delta):])

axs[3] = fig.add_subplot(gs[350:, :triple_width])
axs[4] = fig.add_subplot(gs[350:, (triple_width + 2* triple_width_delta):(2*triple_width + 2*triple_width_delta)])
axs[5] = fig.add_subplot(gs[350:, (2*triple_width + 4*triple_width_delta):])

def get_traj(i):
    # trainer = load_trainer(f'results/archive/2022-08-11/12-39-15/{i}')
    trainer = load_trainer()

    with torch.no_grad():
        traj_length = 5000
        start_index = 0 
        trainer.training_dataset.batch_length = traj_length
        batch_input, batch_y = trainer.training_dataset[start_index]
        batch_input = list(batch_input)
        batch_input[0] = batch_input[0].unsqueeze(0)
        batch_input = tuple(batch_input)

        pred_y = trainer.forward_pass(batch_input, batch_length=traj_length).squeeze().cpu().numpy()
        batch_y = batch_y.cpu().numpy()
        batch_t = trainer.get_batch_t(batch_input[1], batch_length=traj_length).cpu().numpy()
        return batch_t, batch_y, pred_y

def animate(i):
    batch_t, batch_y, pred_y = get_traj(i)

    for k in range(6):
        axs[k].clear()
    
    sep = np.linalg.norm(batch_y[:,1,[6, 7, 8]] - batch_y[:, 0,[6, 7, 8]], axis=1).squeeze()
    pred_sep = np.linalg.norm(pred_y[:,1,[6, 7, 8]] - pred_y[:, 0,[6, 7, 8]], axis=1).squeeze()
    
    axs[0].plot(batch_t, sep, 'k--', alpha=0.8)
    axs[0].plot(batch_t, pred_sep, 'r-', alpha=0.8, label='$\Delta x$')
    # axs[0].legend(loc='upper right')

    


    rel_vel = (batch_y[:,1,[0, 1, 2]] - batch_y[:, 0,[0, 1, 2]]).squeeze()
    pred_rel_vel = (pred_y[:,1,[0, 1, 2]] - pred_y[:, 0,[0, 1, 2]]).squeeze()
    axs[3].clear()
    axs[3].plot(batch_t, pred_rel_vel[:, 0], 'r-', alpha=0.8, label='$\Delta v_x$')
    axs[3].plot(batch_t, pred_rel_vel[:, 1], 'b-', alpha=0.8, label='$\Delta v_y$')
    axs[3].plot(batch_t, pred_rel_vel[:, 2], 'g-', alpha=0.8, label='$\Delta v_z$')
    axs[3].plot(batch_t, rel_vel, 'k--', alpha=0.8)

    axs[1].plot(batch_t, pred_y[:,0,[3]].squeeze(), 'r-', alpha=0.8, label='$\omega_x$')
    axs[1].plot(batch_t, pred_y[:,0,[4]].squeeze(), 'b-', alpha=0.8, label='$\omega_y$')
    axs[1].plot(batch_t, pred_y[:,0,[5]].squeeze(), 'g-', alpha=0.8, label='$\omega_z$' )
    axs[1].plot(batch_t, batch_y[:,0,[3, 4, 5]].squeeze(), 'k--', alpha=0.8)

    axs[4].plot(batch_t, pred_y[:,1,[3]].squeeze(), 'r-', alpha=0.8, label='$\omega_x$')
    axs[4].plot(batch_t, pred_y[:,1,[4]].squeeze(), 'b-', alpha=0.8, label='$\omega_y$')
    axs[4].plot(batch_t, pred_y[:,1,[5]].squeeze(), 'g-', alpha=0.8, label='$\omega_z$' )
    axs[4].plot(batch_t, batch_y[:,1,[3, 4, 5]].squeeze(), 'k--', alpha=0.8)
    # axs[4].legend(loc='upper right')

    axs[2].plot(batch_t, batch_y[:,0,[9, 10, 11, 12]].squeeze(), 'k--', alpha=0.8)
    axs[2].plot(batch_t, pred_y[:,0,[9]].squeeze(), 'r-', alpha=0.8, label='$q_1$')
    axs[2].plot(batch_t, pred_y[:,0,[10]].squeeze(), 'b-', alpha=0.8, label='$q_2$')
    axs[2].plot(batch_t, pred_y[:,0,[11]].squeeze(), 'g-', alpha=0.8, label='$q_3$')
    axs[2].plot(batch_t, pred_y[:,0,[12]].squeeze(), 'm-', alpha=0.8, label='$q_4$')

    axs[5].plot(batch_t, batch_y[:,1,[9, 10, 11, 12]].squeeze(), 'k--', alpha=0.8)
    axs[5].plot(batch_t, pred_y[:,1,[9]].squeeze(), 'r-', alpha=0.8, label='$q_0$')
    axs[5].plot(batch_t, pred_y[:,1,[10]].squeeze(), 'b-', alpha=0.8, label='$q_1$')
    axs[5].plot(batch_t, pred_y[:,1,[11]].squeeze(), 'g-', alpha=0.8, label='$q_2$')
    axs[5].plot(batch_t, pred_y[:,1,[12]].squeeze(), 'm-', alpha=0.8, label='$q_3$')
    # axs[5].legend(loc='upper right')

    axs[0].set_title('Relative Separation')
    axs[0].set_ylabel(r'$||\mathbf{x}_2 - \mathbf{x}_1||$ / $\sigma_{unit}$')
    axs[0].set_ylim(1.9, 3.2)

    axs[1].legend(loc='upper right')
    axs[1].set_ylabel(r'$\mathbf{\omega}_1$ / rad $\cdot \tau^{-1}_{unit}$', labelpad=-3)
    axs[1].set_title('Angular Velocity - First Body')
    axs[1].set_ylim(-0.4, 1.0)

    axs[2].set_ylabel('$\mathbf{q}_1$', labelpad=-5)
    axs[2].legend(loc='upper right')
    axs[2].set_title('Quaternion - First Body')
    axs[2].set_ylim(-0.5, 1.0)
    
    axs[3].legend(loc='upper right')
    axs[3].set_ylabel(r'$\mathbf{v}_2 - \mathbf{v}_1 $ / $\sigma_{unit} \cdot \tau_{unit}^{-1}$', labelpad=-5)
    axs[3].set_title('Relative Velocity')
    axs[3].set_ylim(-0.6, 0.9)
    
    axs[4].set_ylabel(r'$\mathbf{\omega}_2$ / rad $\cdot \tau^{-1}_{unit}$', labelpad=2)
    axs[4].set_title('Angular Velocity - Second Body')
    axs[4].set_ylim(-0.7, 0.3)
    axs[4].set_xlabel(r'Time / $\tau_{unit}$')

    axs[5].set_ylabel('$\mathbf{q}_2$', labelpad=-5)
    axs[5].set_title('Quaternion - Second Body')
    axs[5].set_ylim(-0.8, 1.0)
    
    plt.savefig(f'figures/anim/{i}.png')
    return axs

# for i in range(5, 501, 5):
#     animate(i)
animate(0)
# create animation using the animate() function
# anim = animation.FuncAnimation(fig, animate, frames=range(1, 501), interval=10, repeat=True)

# anim.save('figures/test.gif', writer = 'imagemagick', fps = 30)