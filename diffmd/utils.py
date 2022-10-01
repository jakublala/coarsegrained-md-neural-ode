from multiprocessing import allow_connection_pickling
import torch
from torch.autograd import grad
import torch.distributed as dist
import numpy as np
from datetime import datetime
import time
import random
import yaml

def normalize_quat(q):
    # TODO: documentation
    return q / torch.norm(q, dim=-1).unsqueeze(-1)

def compute_grad(inputs, output, create_graph=True, retain_graph=True, allow_unused=False):
    """
    Compute gradient of the scalar output with respect to inputs.
    
    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 
    
    Returns:
        torch.Tensor: gradients with respect to each input component 
    """
    
    assert inputs.requires_grad
    
    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                   create_graph=create_graph, retain_graph=retain_graph, allow_unused=allow_unused)
    
    return gradspred

def get_run_ID():
    """
    Returns a unique ID for the current run based on the current date and time.
    """
    # wait random amount of time to avoid overwriting
    # time.sleep(random.randint(0, 100))
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S').split('_')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def set_dtype(dtype):
    if dtype == 'float32':
        return torch.float32
    else:
        raise Exception('dtype not implemented')

def set_device(device):
    if device == 'cpu':
        return torch.device('cpu')
    elif 'cuda' in device:
        return torch.device(device)
    else:
        raise Exception('device not implemented')
        


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


