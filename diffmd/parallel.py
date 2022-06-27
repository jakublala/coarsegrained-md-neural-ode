import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from diffmd.training import Trainer
from diffmd.diffeqs import ODEFunc


from torch.nn.parallel import DistributedDataParallel as DDP


class ParallelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.setup_process(self.rank, self.world_size)
        
        
        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth, self.dtype).to(self.device).to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # device_ids tell DDP where is your model
        # output_device tells DDP where to output, in our case, it is rank
        # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model


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


    def process(self, rank, world_size):
        self.setup_process(rank, world_size)

        self.training_dataloader = self.get_parallel_dataloader(self.training_dataset, rank, world_size, self.batch_size)
        self.test_dataloader = self.get_parallel_dataloader(self.test_dataset, rank, world_size, self.batch_size)
        self.validation_dataloader = self.get_parallel_dataloader(self.validation_dataset, rank, world_size, self.batch_size)

        self.func = ODEFunc(self.nparticles, self.dim, self.nn_width, self.nn_depth, self.dtype).to(self.device).to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
        self.train()
        
        cleanup()
        




# suppose we have 3 gpus
world_size = 3        
mp.spawn(
    main,
    args=(world_size),
    nprocs=world_size
)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = Your_Model().to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)    #################### The above is defined previously
   
    optimizer = Your_Optimizer()
    loss_fn = Your_Loss()    
    
    for epoch in epochs:
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        
        for step, x in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(x)
            label = x['label']
            
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()    
    

# # in case we load a DDP model checkpoint to a non-DDP modelmodel_dict = OrderedDict()
# pattern = re.compile('module.')
# for k,v in state_dict.items():
#     if re.search("module", k):
#         model_dict[re.sub(pattern, '', k)] = v
#     else:
#         model_dict = state_dictmodel.load_state_dict(model_dict)