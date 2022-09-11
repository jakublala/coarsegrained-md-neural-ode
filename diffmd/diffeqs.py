import torch
import torch.nn as nn
from diffmd.utils import normalize_quat, compute_grad
from pytorch3d.transforms import quaternion_raw_multiply, quaternion_apply

import time

class ODEFunc(nn.Module):
    def __init__(self, nparticles, dim, widths, functions, dtype):
        super(ODEFunc, self).__init__()
        self.dim = dim
        self.nparticles = nparticles
        self.dtype = dtype
        self.mass = 7.0 # HACK
        
        # define neural net
        depth = len(widths) 
        layers = []
        # first layer takes in all configurational variables (xyz and quaternions)
        layers += [nn.Linear(self.dim, widths[0]), functions[0]]
        for i, width in enumerate(widths):
            if i == (depth-1):  
                # last layer outputs a single potential energy value
                layers += [nn.Linear(width, 1)]
            else:
                layers += [nn.Linear(widths[i], widths[i+1]), functions[i+1]]        
        self.net = nn.Sequential(*layers).type(self.dtype)

        print(self.net)

        assert 0 == 1

        # initialise NN parameters
        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,val=0)

    def forward(self, state):
        with torch.set_grad_enabled(True):   
            # [number of bodies, number of trajectories, number of dimensions] 
            v = state[0]
            w = state[1]
            x = state[2]
            q = state[3]

            x.requires_grad = True
            q.requires_grad = True
            
            # normalise quaternions to unit length
            q = normalize_quat(q)
            
            # get vector separation between bodies (pointing from the first atom to the second)
            r = x[:, 1, :] - x[:, 0, :]
            
            # combine NN inputs
            rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, self.dim)
            
            # assert torch.all(torch.flatten(q[:, 0, :]) == torch.swapaxes(q, 0, 1).reshape(-1, 8)[0, :]), 'incorrect resize'

            # get energy and gradients
            # TODO: check that harmonic restraint is calculated correctly
            u = self.net(rq) + self.harmonic_restraint(rq) # [number of trajectories, potential energy]
            # u = self.zero_net(rq) + self.harmonic_restraint(rq) # [number of trajectories, potential energy]

            f = -compute_grad(inputs=x, output=u)
            grad_q = compute_grad(inputs=q, output=u)

            
            # print(- self.k[0] * (torch.norm(r[0]) - self.r0[0]))
            # print(f[0])

            # assert 0 == 1
            # grad = compute_grad(inputs=rq, output=u) # [force _ torque, number of trajectories]

            # grad_r, grad_q = torch.split(grad, [3, self.dim-3], dim=1)
            # grad_q = grad_q.view(-1, self.nparticles, 4)

            # get force and update translational motion
            # TODO: do this without assigning variables to speed up computation
            # TODO: check that signs in force are correct
            # fA = torch.sign(r_vector) * grad_r # [force, number of trajectories]
            # fB = torch.sign(-r_vector) * grad_r 
            
            # f = torch.stack((fA, fB), dim=1)
            # print('=signs of r_vector==')

            # print(torch.sign(r_vector)[0, :])
            # print(torch.sign(-r_vector)[0, :])

            # print('==through x==')

            # print(f_[0, 0, :])
            # print(f_[0, 1, :])
            
            # print('==through r_vector=')
            
            # print(f[0, 0, :])
            # print(f[0, 1, :])
            # assert torch.all(torch.flatten(f) == torch.flatten(f_)), 'incorrect force calculation'
            
            # get force and update translational motion
            # TODO: do this without assigning variables to speed up computation
            # TODO: check that signs in force are correct

            # HACK: same mass for all bodies
            dvdt = f / self.mass
            dxdt = v
            
            # get torque and update rotational motion
            dqdt = 0.5 * quaternion_raw_multiply(q, torch.cat((torch.zeros(w.shape[:-1]).to(w.device).to(w.dtype).unsqueeze(-1), w), dim=2))
            # dqdt_1 = 0.5 * quatvec(q, w)

            # TODO: test out all matrix multiplications are correct
            # TODO: try assigning G to speed up computation as we re-use it once
            # TODO: if too slow to assign, do it in a single line
            
            l = w * self.inertia
            # TODO: can I help out somehow the neural net with torque calculation by knowing the physical aspect - the torques must be opposite and equal? similarly to force 
            # TODO: is dl_system / dt only the second term which swaps around the values as with dqdt?
            # TODO: G assignment is slow, speed it up
            dldt = - torch.matmul(self.Omega(q, dqdt), l.unsqueeze(-1)).squeeze(-1) - 0.5 * torch.matmul(self.G(q), grad_q.unsqueeze(-1)).squeeze(-1)
            dwdt = dldt / self.inertia

        return (dvdt, dwdt, dxdt, dqdt)
    
    def harmonic_restraint(self, rq):
        return 0.5 * self.k * torch.square(torch.norm(rq[:, 0:3], dim=1) - self.r0.squeeze()).view(-1, 1)
        # return 0.5 * self.k * torch.square(torch.norm(r, dim=1)).view(-1, 1)

    def G(self, q):
        # TODO: move this somewhere; make sure it's fast; maybe torch.stack is not ideal
        # TODO: add documentation
        G = torch.zeros(q.shape[0], q.shape[1], q.shape[2] - 1, q.shape[2]).to(q.device).type(q.type())
        G[:, :, 0, :] = torch.stack((-q[:, :, 1], q[:, :, 0], q[:, :, 3], -q[:, :, 2]), dim=-1)
        G[:, :, 1, :] = torch.stack((-q[:, :, 2], -q[:, :, 3], q[:, :, 0], q[:, :, 1]), dim=-1)
        G[:, :, 2, :] = torch.stack((-q[:, :, 3], q[:, :, 2], -q[:, :, 1], q[:, :, 0]), dim=-1)
        return G

    def Omega(self, q, dqdt):
        # TODO: move this somewhere
        return 2 * torch.matmul(self.G(q), torch.transpose(self.G(dqdt), 2, 3))