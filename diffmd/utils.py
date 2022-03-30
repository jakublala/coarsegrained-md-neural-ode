import torch
from torch.autograd import grad
import numpy as np

def body_to_lab_frame(w):
    """
    Converts angular velocity from body frame reference to lab frame reference

    Args:
        w (torch.Tensor): angular velocity (body frame), requires_grad=True
    
    Returns:
        torch.Tensor: angular velocity skewsymmetric matrix (lab frame)
    """ 

    # TODO: comment on dimensions of w [nparticles, number of trajectory points, 3]
    # TODO: comment on return dimensions [nparticles, number of trajectory points, 4, 4]
    zeros = torch.zeros((w.shape[0], w.shape[1])).to(w.device)
    row_1 = torch.stack([zeros, -w[:, :, 0], -w[:, :, 1], -w[:, :, 2]], dim=2)
    row_2 = torch.stack([w[:, :, 0], zeros, w[:, :, 2], -w[:, :, 1]], dim=2)
    row_3 = torch.stack([w[:, :, 1], -w[:, :, 2], zeros, w[:, :, 0]], dim=2)
    row_4 = torch.stack([w[:, :, 2], w[:, :, 1], -w[:, :, 0], zeros], dim=2)
    return torch.stack([row_1, row_2, row_3, row_4], dim=3)
# TODO: implement skew symmetric matrix this faster using the below
# def skew(v):
#     if len(v) == 4: 
#         v = v[:3]/v[3]
#         skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
#     return skv - skv.T


def lab_to_body_frame(W):
    """
    Converts angular velocity from lab frame reference to body frame reference

    Args:
        W (torch.Tensor): angular velocity skewsymmetric matrix (lab frame), requires_grad=True
    
    Returns:
        torch.Tensor: angular velocity (body frame)
    """ 
    first_row = W[0]
    return torch.Tensor([-first_row[1], -first_row[2], -first_row[3]])
        
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
                   create_graph=create_graph, retain_graph=retain_graph)
    
    return gradspred