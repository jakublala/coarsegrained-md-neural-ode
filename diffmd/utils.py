import torch
from torch.autograd import grad

def body_to_lab_frame(w):
    """
    Converts angular velocity from body frame reference to lab frame reference

    Args:
        w (torch.Tensor): angular velocity (body frame), requires_grad=True
    
    Returns:
        torch.Tensor: angular velocity skewsymmetric matrix (lab frame)
    """ 
    return torch.Tensor([[0, -w[0], -w[1], -w[2]], 
                        [w[0], 0, w[2], -w[1]],
                        [w[1], -w[2], 0, w[0]],
                        [w[2], w[1], -w[0], 0]])      

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
        
def compute_grad(inputs, output, create_graph=True, retain_graph=True):
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