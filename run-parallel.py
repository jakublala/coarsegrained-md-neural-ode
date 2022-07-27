import torch
import torch.multiprocessing as mp
import sys
import os
import time

torch.cuda.empty_cache()
from diffmd.parallel import ParallelTrainer

# TODO: make this dictionary a file that is read
config = dict(
    folder = 'dataset/single_temp_overfit', 
    device = torch.device('cpu'), 
    epochs = 2,
    start_epoch = 0,
    optimizer = 'Adam',
    batch_length=20,
    batch_size=600,
    shuffle=True,
    num_workers=0,
    learning_rate=0.003,
    nn_depth=2,
    nn_width=1000,
    activation_function=None,
    eval_batch_length=100,
    load_folder=None,
    dtype=torch.float32,
    itr_printing_freq=1,
    printing_freq=1,
    plotting_freq=1,
    stopping_freq=5,
    scheduler=None,
    scheduling_factor=0.95,
    scheduling_freq=1,
    evaluation_freq=1,
    checkpoint_freq=1,
    loss_func = 'all',
    sigopt=False,
    )

if __name__ == '__main__':
    start_time = time.perf_counter()
    trainer = ParallelTrainer(config)
    
    world_size = torch.cuda.device_count()
    
    mp.spawn(
        trainer.process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

    print('Training time: {}'.format(time.perf_counter() - start_time))



