import torch
torch.cuda.empty_cache()
from diffmd.training import Trainer
torch.manual_seed(0)

config = dict(
    folder = 'dataset/trajectories/smooth/', 
    device = torch.device("cpu"), 
    niters = 10000,
    optimizer = 'Adam',
    batch_length=20,
    nbatches=800,
    learning_rate=1.0,
    nn_depth=1,
    nn_width=400,
    activation_function=None,
    # load_folder='results/depth-1-width-300-lr-0.1',
    load_folder=None,
    dtype=torch.float32,
    printing_freq=50,
    plotting_freq=250,
    stopping_freq=500,
    scheduler='LambdaLR',
    scheduling_factor=0.90,
    scheduling_freq=500,
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()
