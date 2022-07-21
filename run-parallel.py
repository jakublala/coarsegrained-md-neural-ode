import torch
import torch.multiprocessing as mp
import sys
import os

torch.cuda.empty_cache()
from diffmd.parallel import ParallelTrainer

# TODO: make this dictionary a file that is read
config = dict(
    folder = 'dataset/single_temp_overfit', 
    device = torch.device('cpu'), 
    epochs = 1,
    start_epoch = 0,
    optimizer = 'Adam',
    batch_length=2,
    batch_size=2000,
    shuffle=True,
    num_workers=0,
    learning_rate=0.003,
    nn_depth=2,
    nn_width=1000,
    activation_function=None,
    eval_batch_length=5,
    load_folder=None,
    dtype=torch.float32,
    itr_printing_freq=10,
    printing_freq=1,
    plotting_freq=1,
    stopping_freq=5,
    scheduler=None,
    scheduling_factor=0.95,
    scheduling_freq=1,
    evaluation_freq=1,
    checkpoint_freq=1,
    loss_func = 'all',
    )

if __name__ == '__main__':

    trainer = ParallelTrainer(config)
    
    world_size = 2
    
    mp.spawn(
        trainer.process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

    trainer.save()



