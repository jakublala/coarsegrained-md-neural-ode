import torch
from torch.autograd import grad
import numpy as np
def G(q):
        return np.array([[-q[1], q[0], q[3], -q[2]], [-q[2], -q[3], q[0], q[1]], [-q[3], q[2], -q[1], q[0]]]).squeeze()
    
def vecquat(a, b):
    # TODO: add documentation
    c = torch.zeros(b.shape).to(b.device).type(torch.float64)
    c[:, :, 0] = -a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 3]
    c[:, :, 1] = b[:, :, 0] * a[:, :, 0] + a[:, :, 1] * b[:, :, 3] - a[:, :, 2] * b[:, :, 2]
    c[:, :, 2] = b[:, :, 0] * a[:, :, 1] + a[:, :, 2] * b[:, :, 1] - a[:, :, 0] * b[:, :, 3]
    c[:, :, 3] = b[:, :, 0] * a[:, :, 2] + a[:, :, 0] * b[:, :, 2] - a[:, :, 1] * b[:, :, 1]
    return c

def quatvec(a, b):
    # TODO: add documentation
    c = torch.zeros(a.shape).to(a.device).type(torch.float64)
    c[:, :, 0] = -a[:, :, 1] * b[:, :, 0] - a[:, :, 2] * b[:, :, 1] - a[:, :, 3] * b[:, :, 2]
    c[:, :, 1] = a[:, :, 0] * b[:, :, 0] + a[:, :, 2] * b[:, :, 2] - a[:, :, 3] * b[:, :, 1]
    c[:, :, 2] = a[:, :, 0] * b[:, :, 1] + a[:, :, 3] * b[:, :, 0] - a[:, :, 1] * b[:, :, 2]
    c[:, :, 3] = a[:, :, 0] * b[:, :, 2] + a[:, :, 1] * b[:, :, 1] - a[:, :, 2] * b[:, :, 0]
    return c

def quatquat(a, b):
    # TODO: add documentation
    # TODO: add this
    return 


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
    # using ? convention
    # row_1 = torch.stack([zeros, -w[:, :, 0], -w[:, :, 1], -w[:, :, 2]], dim=2)
    # row_2 = torch.stack([w[:, :, 0], zeros, w[:, :, 2], -w[:, :, 1]], dim=2)
    # row_3 = torch.stack([w[:, :, 1], -w[:, :, 2], zeros, w[:, :, 0]], dim=2)
    # row_4 = torch.stack([w[:, :, 2], w[:, :, 1], -w[:, :, 0], zeros], dim=2)
    # using Omelyan convention
    # TODO: maybe it does not matter which convention?
    # row_1 = torch.stack([zeros, w[:, :, 2], -w[:, :, 0], -w[:, :, 1]], dim=2)
    # row_2 = torch.stack([-w[:, :, 2], zeros, -w[:, :, 1], w[:, :, 0]], dim=2)
    # row_3 = torch.stack([w[:, :, 0], w[:, :, 1], zeros, w[:, :, 2]], dim=2)
    # row_4 = torch.stack([w[:, :, 1], -w[:, :, 0], -w[:, :, 2], zeros], dim=2)
    # using Kou 2018 convention
    row_1 = torch.stack([zeros, w[:, :, 2], -w[:, :, 1], w[:, :, 0]], dim=2)
    row_2 = torch.stack([-w[:, :, 2], zeros, w[:, :, 0], w[:, :, 1]], dim=2)
    row_3 = torch.stack([w[:, :, 1], -w[:, :, 0], zeros, w[:, :, 2]], dim=2)
    row_4 = torch.stack([-w[:, :, 0], -w[:, :, 1], -w[:, :, 2], zeros], dim=2)
    return torch.stack([row_1, row_2, row_3, row_4], dim=3)
# TODO: implement skew symmetric matrix this faster using the below
# def skew(v):
#     if len(v) == 4: 
#         v = v[:3]/v[3]
#         skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
#     return skv - skv.T


def lab_to_body_frame(a, q):
    # TODO: documentation
    q_conj = conjugate_quat(q)
    return vecquat(quatvec(q_conj, a), q)[:, :, 1:]

def quat_to_euler_angles(q):
    # TODO: documentation
    q0 = q[:, :, -1]
    q1 = q[:, :, 0]
    q2 = q[:, :, 1]
    q3 = q[:, :, 2]
    eul1 = torch.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    eul2 = torch.asin(2*(q0*q2 - q3*q1))
    eul3 = torch.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return torch.stack([eul1, eul2, eul3], dim=2)

def normalize_quat(q, dim):
    # TODO: documentation
    return q / torch.norm(q, dim=dim).unsqueeze(2)

def quat_rotation(v, q, dim, is_vector=True):
    # TODO; documentation and improve generalizibility
    if is_vector:
        shape = [*v.shape]
        assert shape[dim] == 3, 'vector must be 3D' 
        shape[dim] = 1
        v = torch.cat([torch.zeros(shape).to(v.get_device()), v], dim=dim)

    assert q.shape[dim] == 4, 'quaternion must be 4D'
    assert q.shape == v.shape, 'quaternion and vector must have same shape'
    
    return q * v * conjugate_quat(q)

def conjugate_quat(q):
    assert q.shape[-1] == 4, 'quaternion must be 4D in the last dimension'
    return torch.cat((q[:, :, 0:1], -q[:, :, 1:]), dim=-1)

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