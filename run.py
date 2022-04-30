import torch
torch.cuda.empty_cache()
from diffmd.training import Trainer
torch.manual_seed(0)

config = dict(
    folder = 'hexagons/trajectories/smooth/', 
    device = torch.device("cpu"), 
    niters = 50,
    optimizer = 'Adam',
    batch_length=10,
    nbatches=800,
    learning_rate=0.5,
    nn_depth=1,
    nn_width=800,
    activation_function=None,
    # load_folder='results/depth-1-width-300-lr-0.1',
    load_folder=None,
    dtype=torch.float32,
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()
