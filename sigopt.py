from diffmd.solvers import odeint_adjoint
from diffmd.utils import compute_grad
from diffmd.utils import body_to_lab_frame, lab_to_body_frame, quat_to_euler_angles
from analysis.reader import Reader


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import shutil
# TODO: fix quaternion module warning because of NUMBA
import quaternion
from torchviz import make_dot # visualize computation graphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f'Using {device} device')

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.log(val)
    
    def log(self, val):
        self.losses.append(val)

def get_first_batch(trajs,nsample,sampleLength,dt):
    twice_dim = trajs.size()[2]
    dim = twice_dim//2
    
    q0 = trajs[0:nsample, 0, :dim].to(device)
    p0 = trajs[0:nsample, 0, dim:].to(device)
    batch_y0 = (p0, q0)
    
    q = trajs[0:nsample, 0:sampleLength, :dim].to(device)
    p = trajs[0:nsample, 0:sampleLength, dim:].to(device)
    batch_y = torch.cat((p, q), dim=2).swapaxes(0, 1)
    
    batch_t = torch.linspace(0.0,dt*(sampleLength-1),sampleLength).to(device)
    return batch_t, batch_y0, batch_y

class ODEFunc(nn.Module):
    def __init__(self, nparticles, inertia, k, dim, width, depth):
        super(ODEFunc, self).__init__()
        self.dim = dim
        self.nparticles = nparticles
        layers = []
        for i in range(depth):
            if i == 0:
                # first layer takes in all configurational variables (xyz and quaternions)
                layers += [nn.Linear(self.dim, width), nn.Sigmoid()]
            if i == (depth-1):
                # last layer outputs a single potential energy value
                layers += [nn.Linear(width, 1)]
            else:
                layers += [nn.Linear(width, width), nn.Sigmoid()]
        self.net = nn.Sequential(*layers).type(torch.float64)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,val=0)

        # testing only translational motion
        layers = []
        for i in range(depth):
            if i == 0:
                # first layer takes in all configurational variables (xyz and quaternions)
                layers += [nn.Linear(1, width), nn.Sigmoid()]
            if i == (depth-1):
                # last layer outputs a single potential energy value
                layers += [nn.Linear(width, 1)]
            else:
                layers += [nn.Linear(width, width), nn.Sigmoid()]
        self.trans_net = nn.Sequential(*layers).type(torch.float64)

        for m in self.trans_net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,val=0)


        # HACK
        self.mass = 7.0
        self.inertia = inertia
        self.k = k

    def forward(self, state):

        with torch.set_grad_enabled(True):    
            # TODO: make sure batching and dimensions are sent appropriately
            v = state[0]
            # print(v.requires_grad)
            w = state[1]
            x = state[2] # [number of bodies, number of trajectories, number of dimensions]
            q = state[3]
            
            x.requires_grad = True
            q.requires_grad = True
            
            # normalise quaternions to unit length
            q = q / torch.norm(q, dim=2).unsqueeze(2)
            assert torch.norm(q, dim=2).max() < 1.001, 'quaternions not normalised'
            
            # get separation between bodies
            r_vector = x[0, :, :] - x[1, :, :]
            r = torch.norm(r_vector, dim=-1).unsqueeze(1)
            r_vector = r_vector / r
            
            assert torch.norm(r_vector, dim=1).max() < 1.1, 'separation vector not normalised'
            
            # TODO: is this the corect way to get euler angles? is the correct implementation in terms of the cosine and sine formula?
            # TODO: test out different conventions of how to interpret quaternions
            # TODO: write simple unit tests to check we are truly using the correct convention
            theta = quat_to_euler_angles(q).view(-1, 2*3)
            # print(theta.shape)

            rtheta = torch.cat((r, theta), dim=1).view(-1, 2*3+1)
            # print(rtheta.shape)
            # torch.set_printoptions(profile="full")
            # print(rtheta)
            # TODO: check that harmonic restraint is calculated correctly
            # u = self.net(rtheta) - self.harmonic_restraint(r) # [potential energy, number of trajectories]
            # fT = -compute_grad(inputs=rtheta, output=u.T) # [force _ torque, number of trajectories]
            
            # testing out only trans motion
            u = self.trans_net(r) - self.harmonic_restraint(r) # [potential energy, number of trajectories]
            f = -compute_grad(inputs=r, output=u.T) 
            
            # HACK: make sure we are splitting based on number of coordinates
            # f, T = torch.split(fT, [1, 6], dim=1) # capital T means in lab coordinates
            
            T = torch.zeros((f.shape[0], 6)).to(device)
            # TODO: do this without assigning variables to speed up computation
            fA = - f * r_vector # [force, number of trajectories]
            fB = f * r_vector
            f = torch.stack((fA, fB), dim=0)
            
            
            # TODO; do the torque thing

            # translational motion
            # HACK: same mass for all bodies
            dvdt = f / self.mass
            # print(torch.norm(dvdt, dim=[0, 2]).mean())
            dxdt = v


            # rotational motion
            q_conj = torch.cat((q[:, :, 0:1], -q[:, :, 1:]), dim=2) # get conguarient quaternions
            dLdt = torch.cat((torch.zeros(q.shape[0], q.shape[1], 1).to(device), T.reshape(2, -1, 3)), dim=2) # [number of bodies, number of trajectories, 4]
            dldt = q * dLdt * q_conj
            dwdt = dldt[:, :, 1:] / self.inertia[:, None, :]
            dqdt = (0.5 * body_to_lab_frame(w) @ q[:, :, :, None]).squeeze()
            assert torch.all(dqdt[0, 0, :] == (0.5 * body_to_lab_frame(w)[0, 0, :, :] @ q[0, 0, :, None]).squeeze()), 'dqdt tensor update is incorrect'

        return (dvdt, dwdt, dxdt, dqdt)

    def harmonic_restraint(self, diff):
        return 0.5 * self.k * torch.square(diff)

    # def compute_grad(self, q, dq):
    #     # using this over compute_grad from nff seems to make no difference
    #     # HACK: does this only work with q_n => n=1? 
    #     return (self.net(q+dq) - self.net(q-dq)) / (2 * dq)

def get_data():
    # train_split = 0.9
    # test_split = 1 - train_split
    df = pd.read_csv(file_name+'-reduced_traj.csv')
    # HACK: do this based on the column names, not explicitly
    com = ['c_com_1[1]', 'c_com_1[2]', 'c_com_1[3]', 'c_com_2[1]', 'c_com_2[2]', 'c_com_2[3]']
    q = ['c_q_1[1]', 'c_q_1[2]', 'c_q_1[3]', 'c_q_1[4]', 'c_q_2[1]', 'c_q_2[2]', 'c_q_2[3]', 'c_q_2[4]']
    vel = ['c_vel_1[1]', 'c_vel_1[2]', 'c_vel_1[3]', 'c_vel_2[1]', 'c_vel_2[2]', 'c_vel_2[3]'] 
    av = ['c_av_1[1]', 'c_av_1[2]', 'c_av_1[3]', 'c_av_2[1]', 'c_av_2[2]', 'c_av_2[3]']
    am = ['c_am_1[1]', 'c_am_1[2]', 'c_am_1[3]', 'c_am_2[1]', 'c_am_2[2]', 'c_am_2[3]']
    inertia = ['c_i_1[1]', 'c_i_1[2]', 'c_i_1[3]', 'c_i_2[1]', 'c_i_2[2]', 'c_i_2[3]']
    
    centre_of_masses = df.loc[:, ['timestep', *com]]
    quaternions = df.loc[:, ['timestep', *q]]
    velocities = df.loc[:, ['timestep', *vel]]
    ang_velocities = df.loc[:, ['timestep', *av]]
    ang_momenta = df.loc[:, ['timestep', *am]]
    # inertia = df.loc[0, ['timestep', *inertia]]
    inertia = df.loc[:, ['timestep', *inertia]]
    # TODO: use DataLoaders?
    
    # trajs = np.load('data/trajectories/diatomic_spring_narrow.npy')
    # split_index = int(trajs.shape[0] * train_split)

    # np.random.shuffle(trajs)
    # training_trajs = torch.Tensor(trajs[:split_index, :, :]).to(device)
    # testing_trajs = torch.Tensor(trajs[split_index:, :, :]).to(device)
    
    return centre_of_masses, quaternions, velocities, ang_velocities, ang_momenta, inertia

def process_inertia(inertia):
    assert np.all(inertia.std().iloc[1:].to_numpy() == 0), 'inertia is not constant'
    inertia = inertia.iloc[0, 1:].to_numpy().reshape(2, 3)
    return torch.from_numpy(inertia).to(device)

def process_data(centre_of_masses, quaternions, velocities, ang_momenta, inertia):
    # HACK: send a single trajectory
    ntraj = 1

    # HACK: pair potential
    nparticles = 2
    vel_dim = 3
    angvel_dim = 3
    com_dim = 3
    quat_dim = 4

    # Get centre of masses
    com1 = centre_of_masses.loc[:, ['c_com_1[1]', 'c_com_1[2]', 'c_com_1[3]']].to_numpy()
    com2 = centre_of_masses.loc[:, ['c_com_2[1]', 'c_com_2[2]', 'c_com_2[3]']].to_numpy()
    # separation = np.linalg.norm(com1-com2, axis=1).reshape(-1, 1)
    coms = torch.from_numpy(np.hstack((com1, com2))).to(device).view(ntraj, -1, nparticles, com_dim)
    
    # Get quaternion rotations
    quat1 = quaternions.loc[:, ['c_q_1[1]', 'c_q_1[2]', 'c_q_1[3]', "c_q_1[4]"]].to_numpy()
    quat2 = quaternions.loc[:, ['c_q_2[1]', 'c_q_2[2]', 'c_q_2[3]', "c_q_2[4]"]].to_numpy()
    quats = torch.from_numpy(np.hstack((quat1, quat2))).to(device).view(ntraj, -1, nparticles, quat_dim)
    
    # Get translation velocities
    vel1 = velocities.loc[:, ['c_vel_1[1]', 'c_vel_1[2]', 'c_vel_1[3]']].to_numpy()
    vel2 = velocities.loc[:, ['c_vel_2[1]', 'c_vel_2[2]', 'c_vel_2[3]']].to_numpy() 
    # hexagon_mass = 7.0
    # mom = torch.from_numpy(np.hstack((vel1 * hexagon_mass, vel2 * hexagon_mass))).to(device).view(ntraj, -1, nparticles, vel_dim)
    vel = torch.from_numpy(np.hstack((vel1, vel2))).to(device).view(ntraj, -1, nparticles, vel_dim)

    # Get angular velocities
    ang_mom = torch.from_numpy(ang_momenta.iloc[:, 1:].to_numpy()).to(device).view(ntraj, -1, nparticles, angvel_dim)
    ang_vel = ang_mom / inertia.view(ntraj, -1, nparticles, angvel_dim)
    return (vel, ang_vel, coms, quats)


def get_batch_mod(traj, batch_size, batch_length, dt):
    # TODO: change everything from ANGULAR VELOCITY to ANGULAR MOMENTUM
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
    nparticles = trajectory[0].shape[2]
    vel_dim = trajectory[0].size()[3]
    angvel_dim = trajectory[1].size()[3]
    com_dim = trajectory[2].size()[3]
    quat_dim = trajectory[3].size()[3]

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
        vels = traj[0][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:].reshape(nparticles, batch_length, vel_dim)
        ang_vels = traj[1][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:].reshape(nparticles, batch_length, angvel_dim)
        coms = traj[2][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:].reshape(nparticles, batch_length, com_dim)
        quats = traj[3][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:].reshape(nparticles, batch_length, quat_dim)
        
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
