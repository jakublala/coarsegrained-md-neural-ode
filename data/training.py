import torch
import quaternion
import pandas as pd
import numpy as np

def get_batch_mod(traj, batch_size, batch_length, dt, device):
    # TODO: move this to the trainer class or create a BATCHING class
    """
    Get a batch of data from a trajectory.
        
    Args:
        traj (tuple): torch tensors containing the data (velocities, angular velocities, centre of masses, quaternions)
        batch_size (int): number of trajectories in the batch
        batch_length (int): length of each trajectory in the batch
        dt (float): time step
    
    Returns:
        batch_t (torch.Tensor): time steps for the batch
        pos_init (torch.Tensor): initial positions for the batch
        batch_trajs (tuple): batch of torch tensors containing the data (velocities, angular velocities, centre of masses, quaternions - in the form (nparticles, batch_size, batch_length, dim))
    """
    nparticles = traj[0].shape[2]
    vel_dim = traj[0].size()[3]
    angvel_dim = traj[1].size()[3]
    com_dim = traj[2].size()[3]
    quat_dim = traj[3].size()[3]

    assert vel_dim == 3, 'velocity dimension must be 3'
    assert angvel_dim == 3, 'angular velocity dimension must be 3'
    assert com_dim == 3, 'centre of mass dimension must be 3'
    assert quat_dim == 4, 'quaternion dimension must be 4'

    sampled_is = torch.randint(traj[0].shape[0],size = (batch_size,)).to(device)
    sampled_js = torch.randint(traj[0].shape[1]-batch_length,size = (batch_size,)).to(device)
    initial_time = sampled_js*dt
   
    batch_t = torch.linspace(0.0,dt*(batch_length-1),batch_length).to(device)
    
    vels = torch.swapaxes(traj[0][sampled_is,sampled_js,:,:], 0, 1)
    ang_vels = torch.swapaxes(traj[1][sampled_is,sampled_js,:,:], 0, 1)
    coms = torch.swapaxes(traj[2][sampled_is,sampled_js,:,:], 0, 1)
    quats = torch.swapaxes(traj[3][sampled_is,sampled_js,:,:], 0, 1)
    
    pos_init = (vels, ang_vels, coms, quats)

    sampled_vels = []
    sampled_ang_vels = []
    sampled_coms = []
    sampled_quats = []
    for i in range(batch_size):
        vels = torch.swapaxes(traj[0][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:], 0, 1)
        ang_vels = torch.swapaxes(traj[1][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:], 0, 1)
        coms = torch.swapaxes(traj[2][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:], 0, 1)
        quats = torch.swapaxes(traj[3][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:], 0, 1)
        
        sampled_vels.append(vels)
        sampled_ang_vels.append(ang_vels)
        sampled_coms.append(coms)
        sampled_quats.append(quats)
        
    sampled_vels = torch.stack(sampled_vels, dim=1).type(torch.float64)
    sampled_ang_vels = torch.stack(sampled_ang_vels, dim=1).type(torch.float64)
    sampled_coms = torch.stack(sampled_coms, dim=1).type(torch.float64)
    sampled_quats = torch.stack(sampled_quats, dim=1).type(torch.float64)
    batch_trajs = (sampled_vels, sampled_ang_vels, sampled_coms, sampled_quats)

    return batch_t, pos_init, batch_trajs
