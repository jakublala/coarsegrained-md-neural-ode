import torch
torch.cuda.empty_cache()
from diffmd.training import Trainer
torch.manual_seed(0)

config = dict(
    folder = 'dataset/trajectories/smooth/', 
    device = torch.device("cpu"), 
    niters = 500,
    optimizer = 'Adam',
    batch_length=10,
    nbatches=800,
    learning_rate=1.0,
    nn_depth=1,
    nn_width=300,
    activation_function=None,
    # load_folder='results/depth-1-width-300-lr-0.1',
    load_folder=None,
    dtype=torch.float32,
    printing_freq=10,
    plotting_freq=10,
    stopping_freq=500,
    scheduler='LambdaLR',
    scheduling_factor=0.90,
    scheduling_freq=10,
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()
