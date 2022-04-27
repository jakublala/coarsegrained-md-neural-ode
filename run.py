import torch
from diffmd.training import Trainer
torch.manual_seed(0)

prefix = 'hexagons/trajectories/smooth/'
dataset = 'NVE-temp-0.45_K-0.090702947845805_r-0_s-5'
config = dict(
    filename = prefix+dataset, 
    device = torch.device("cpu"), 
    niters = 100,
    optimizer = 'Adam',
    batch_length=20,
    nbatches=800,
    learning_rate=0.1,
    nn_depth=5,
    nn_width=100,
    activation_function=None,
)

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()
