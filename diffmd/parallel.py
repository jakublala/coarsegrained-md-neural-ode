import os
import time

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from diffmd.training import Trainer
from diffmd.diffeqs import ODEFunc


from torch.nn.parallel import DistributedDataParallel as DDP


class ParallelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def spawn_processes(self, world_size):
        mp.spawn(
        self.process,
        args=(world_size),
        nprocs=world_size
        )
    
    def process(self, rank, world_size):
        print('setting up process')
        self.setup_process(rank, world_size)

        print('starting dataloader')
        print('rank: ', rank)
        self.training_dataloader = self.get_parallel_dataloader(self.training_dataset, rank, world_size, self.batch_size)
        self.test_dataloader = self.get_parallel_dataloader(self.test_dataset, rank, world_size, self.batch_size)
        self.validation_dataloader = self.get_parallel_dataloader(self.validation_dataset, rank, world_size, self.batch_size)
        
        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth, self.dtype).to(self.device).to(rank)
        self.func = DDP(self.func, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
        self.train()
        
        # add early stopping?
        cleanup()

    def train(self):
        for self.epoch in range(self.start_epoch + 1, (self.start_epoch + self.epochs) + 1):
            self.training_dataloader.sampler.set_epoch(self.epoch)       
            self.start_time = time.perf_counter()
        
            
            # zero out gradients with less memory operations
            for param in self.func.parameters():
                param.grad = None

            for self.itr, (batch_input, batch_y) in enumerate(self.training_dataloader):
                itr_start_time = time.perf_counter()

                # forward pass
                pred_y = self.forward_pass(batch_input, batch_y).cpu()

                loss = self.loss_func(pred_y, batch_y)

                # backward pass                    
                loss.backward() 
                self.optimizer.step()
            
                self.loss_meter.update(loss.item(), self.optimizer.param_groups[0]["lr"])
                
                if (self.itr+1) % self.itr_printing_freq == 0:
                    self.print_iteration(itr_start_time)

            self.print_epoch()
        return self.func, self.loss_meter

    def setup_process(self, rank, world_size): # setup the process group
        # rank is the gpu id of the process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'    
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def get_parallel_dataloader(self, dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0): # split the dataloader
        # TODO: documentation
        # TODO: drop last ?
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
        return dataloader


def cleanup():
    dist.destroy_process_group()


# # in case we load a DDP model checkpoint to a non-DDP modelmodel_dict = OrderedDict()
# pattern = re.compile('module.')
# for k,v in state_dict.items():
#     if re.search("module", k):
#         model_dict[re.sub(pattern, '', k)] = v
#     else:
#         model_dict = state_dictmodel.load_state_dict(model_dict)