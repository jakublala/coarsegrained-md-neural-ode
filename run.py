import torch
torch.cuda.empty_cache()
from diffmd.training import Trainer
torch.manual_seed(0)

config = dict(
    folder = 'dataset/smooth/', 
    device = torch.device("cpu"), 
    niters = 1000,
    optimizer = 'Adam',
    batch_length=20,
    nbatches=800,
    learning_rate=0.3,
    nn_depth=1,
    nn_width=400,
    activation_function=None,
    # load_folder='results/depth-1-width-300-lr-0.1',
    load_folder=None,
    dtype=torch.float32,
    printing_freq=25,
    plotting_freq=100,
    stopping_freq=1000,
    scheduler='LambdaLR',
    scheduling_factor=0.75,
    scheduling_freq=500,
    evaluation_freq=10,
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()

