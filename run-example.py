import torch
torch.cuda.empty_cache()
from diffmd.trainer_base import Trainer

import numpy as np

# TODO: make this dictionary a file that is read
config = dict(
    folder = 'dataset/single_temp_overfit', 
    device = torch.device('cpu'), 
    epochs = 100,
    start_epoch = 0,
    optimizer = 'Adam',
    batch_length=20,
    batch_size=600,
    shuffle=True,
    num_workers=0,
    learning_rate=0.0003,
    nn_depth=2,
    nn_width=1000,
    activation_function=None,
    eval_batch_length=1000,
    load_folder=None,
    dtype=torch.float32,
    itr_printing_freq=1,
    printing_freq=20,
    plotting_freq=20,
    stopping_freq=20,
    scheduler='LambdaLR',
    scheduling_factor=0.95,
    scheduling_freq=25,
    evaluation_freq=20,
    checkpoint_freq=20,
    loss_func = 'all',
    )

trainer = Trainer(config)
model, train_loss = trainer.train()
trainer.save()

