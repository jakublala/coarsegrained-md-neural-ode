import os
import time

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from diffmd.training import Trainer
from diffmd.diffeqs import ODEFunc
from diffmd.solvers import odeint_adjoint

from torch.nn.parallel import DistributedDataParallel as DDP


class ParallelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def process(self, rank, world_size):
        self.setup_process(rank, world_size)
        
        self.training_dataloader = self.get_parallel_dataloader(self.training_dataset, rank, world_size, self.batch_size)
        self.test_dataloader = self.get_parallel_dataloader(self.test_dataset, rank, world_size, self.batch_size)
        self.validation_dataloader = self.get_parallel_dataloader(self.validation_dataset, rank, world_size, self.batch_size)
        
        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth, self.dtype).to(self.device).to(rank)
        self.func = DDP(self.func, device_ids=[rank], output_device=rank, find_unused_parameters=False, static_graph=False)
        
        self.loss_func = self.set_loss_func(self.loss_func_name)
        self.optimizer = self.set_optimizer(self.optimizer_name)
        self.scheduler = self.set_scheduler(self.scheduler_name, self.scheduling_factor)
        
        self.train()

        # TODO: Add final save
        # if is_main_process():
        #     self.save()

        # add early stopping?
        cleanup()

    def train(self):
        for self.epoch in range(self.start_epoch + 1, (self.start_epoch + self.epochs) + 1):
            self.training_dataloader.sampler.set_epoch(self.epoch)       
            self.start_time = time.perf_counter()
        
            for self.itr, (batch_input, batch_y) in enumerate(self.training_dataloader):
                self.itr_start_time = time.perf_counter()

                # zero out gradients with less memory operations
                for param in self.func.parameters():
                    param.grad = None

                batch_y = batch_y.to(self.device, non_blocking=True).type(self.dtype)

                # forward pass
                pred_y = self.forward_pass(batch_input, parallel=True)

                loss = self.loss_func(pred_y, batch_y)

                # backward pass                    
                loss.backward() 
                self.optimizer.step()
            
                if is_main_process():  
                    self.loss_meter.update(loss.item(), self.optimizer.param_groups[0]["lr"])

                    if (self.itr+1) % self.itr_printing_freq == 0:
                        self.print_iteration()

            # logging and waiting for all processes to finish epoch
            if is_main_process():                
                self.after_epoch()

            # torch.distributed.barrier()
        
        return self.func, self.loss_meter

    def setup_process(self, rank, world_size): # setup the process group
        # rank is the gpu id of the process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'    
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
        self.device = f'cuda:{rank}'

    def get_parallel_dataloader(self, dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0): # split the dataloader
        # TODO: documentation
        # TODO: drop last ?
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
        return dataloader


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def cleanup():
    dist.destroy_process_group()


# # in case we load a DDP model checkpoint to a non-DDP modelmodel_dict = OrderedDict()
# pattern = re.compile('module.')
# for k,v in state_dict.items():
#     if re.search("module", k):
#         model_dict[re.sub(pattern, '', k)] = v
#     else:
#         model_dict = state_dictmodel.load_state_dict(model_dict)