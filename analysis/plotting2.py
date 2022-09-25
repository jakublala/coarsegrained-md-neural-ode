
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
from diffmd.training import Trainer

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


import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
    
def animate(index):

    # trainer = load_trainer(f'results/archive/2022-08-11/12-39-15/{index}')
    trainer = load_trainer()
    # contour potential
    potential = trainer.func.net

    start, stop, n_values = -5, 5, 50
    vals = np.linspace(start, stop, n_values)

    # model
    X, Y = np.meshgrid(vals, vals)
    XY = np.stack([X, Y], axis=2)

    Z = np.zeros((n_values, n_values))

    for i in range(XY.shape[0]):
        for j in range(XY.shape[1]):
            inp = np.hstack((XY[i, j], np.array([0, 0, 0, 0, 1, 0, 0, 0, 1])))
            Z[i, j] = potential(torch.Tensor(inp)).detach().cpu().item()

    Z -= Z.min()

    ax.clear()
    ax.plot_surface(X, Y, Z, cmap="plasma")
    ax.view_init(30, index*1)
    ax.set_xlabel(r'$\Delta x_1$ / $\sigma_{unit}$')
    ax.set_ylabel(r'$\Delta x_2$ / $\sigma_{unit}$')
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('$V_{CG}$')
    ax.set_zlim(0, 10)
    plt.savefig(f'figures/anim2/{index}.png', dpi=100)

# for index in range(5, 501, 5):
#     animate(index)

animate(0)

# anim = animation.FuncAnimation(fig, animate, frames=range(1, 501), interval=10, repeat=True)
# anim.save('figures/test.gif', writer = 'imagemagick', fps = 30)