import os
import time

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from diffmd.trainers import NODETrainer
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint
from data.dataset import Dataset
from diffmd.utils import cleanup

from torch.nn.parallel import DistributedDataParallel as DDP


class ParallelTrainer(NODETrainer):
    def __init__(self, config):
        super().__init__(config)
        self.parallel = True
        self.gpu_ids = config["gpu_ids"]
        
    def process(self, rank, world_size):
        self.setup_process(rank, world_size)
        
        # TODO: fix a bug where having a batch length smaller than len(dataset) causes an error
        self.training_dataloader = self.get_parallel_dataloader(self.training_dataset, rank, world_size, self.batch_size)
        assert len(self.training_dataset) > self.batch_size * len(self.gpu_ids), "Batch size is too large for the dataset. Please reduce the batch size, increase the dataset size, or use less GPUs."

        # TODO: add parallel evaluate and validation ?
        # self.test_dataloader = self.get_parallel_dataloader(self.test_dataset, rank, world_size, self.batch_size)
        # self.validation_dataloader = self.get_parallel_dataloader(self.validation_dataset, rank, world_size, self.batch_size)
        
        self.func = DDP(self.func.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=False, static_graph=False)
        
        self.loss_func = self.set_loss_func(self.loss_func_name)
        self.optimizer = self.set_optimizer(self.optimizer_name)
        self.scheduler = self.set_scheduler(self.scheduler_name, self.scheduling_factor)
        
        self.train()

        cleanup()

    def setup_process(self, rank, world_size): # setup the process group
        # rank is the gpu id of the process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'    
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
        self.device = f'cuda:{self.gpu_ids[rank]}'

    def get_parallel_dataloader(self, dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0): # split the dataloader
        # TODO: documentation
        # TODO: drop last ?
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
        return dataloader
