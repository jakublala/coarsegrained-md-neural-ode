import torch
torch.cuda.empty_cache()
from diffmd.training import Trainer
# torch.manual_seed(0)

config = dict(
    folder = 'dataset/trajectories/thesis/', 
    device = torch.device("cuda:1"), 
    niters = 5,
    start_niter = 0,
    optimizer = 'Adam',
    batch_length=20,
    nbatches=600,
    learning_rate=0.03,
    nn_depth=2,
    nn_width=10000,
    activation_function=None,
    # load_folder='results/depth-1-width-300-lr-0.1',
    load_folder=None,
    dtype=torch.float32,
    printing_freq=1,
    plotting_freq=250,
    stopping_freq=1000,
    scheduler='LambdaLR',
    scheduling_factor=0.85,
    scheduling_freq=1000,
    evaluation_freq=500,
    loss_func = 'mae-all'
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()

