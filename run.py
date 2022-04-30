import torch
from diffmd.training import Trainer
torch.manual_seed(0)

prefix = 'hexagons/trajectories/smooth/'
dataset = 'NVE-temp-0.45_K-0.090702947845805_r-0_s-5'
config = dict(
    filename = prefix+dataset, 
    device = torch.device("cuda"), 
    niters = 300,
    optimizer = 'Adam',
    batch_length=10,
    nbatches=800,
    learning_rate=0.5,
    nn_depth=2,
    nn_width=2000,
    activation_function=None,
    # load_folder='results/depth-1-width-300-lr-0.1',
    load_folder=None
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()
