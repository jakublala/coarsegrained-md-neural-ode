import torch
import torch.nn as nn
from diffmd.utils import normalize_quat, compute_grad, quatvec

class ODEFunc(nn.Module):
    def __init__(self, nparticles, dim, width, depth):
        super(ODEFunc, self).__init__()
        self.dim = dim
        self.nparticles = nparticles
        self.mass = 7.0 # HACK
        self.inertia = torch.Tensor()
        self.k = float()

        # define neural net
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

        # initialise NN parameters
        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
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
            q = normalize_quat(q, dim=2)
            # assert torch.norm(q, dim=2).max() < 1.001, 'quaternions not normalised'
            
            # get separation between bodies
            r_vector = x[:, 0, :] - x[:, 1, :]
            r = torch.norm(r_vector, dim=-1).unsqueeze(1)
            r_vector = r_vector / r
            # assert torch.norm(r_vector, dim=1).max() < 1.1, 'separation vector not normalised'
            
            # combine NN inputs
            rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, self.dim)
            # assert torch.all(torch.flatten(q[:, 0, :]) == torch.swapaxes(q, 0, 1).reshape(-1, 8)[0, :]), 'incorrect resize'

            # get energy and gradients
            # TODO: check that harmonic restraint is calculated correctly
            u = self.net(rq) + self.harmonic_restraint(r) # [potential energy, number of trajectories]
            # TODO: check that gradient is calculated correctly
            grad = compute_grad(inputs=rq, output=u) # [force _ torque, number of trajectories]
            grad_r, grad_q = torch.split(grad, [1, self.dim-1], dim=1)
            grad_q = grad_q.view(-1, self.nparticles, 4)
            
            # get force and update translational motion
            # TODO: do this without assigning variables to speed up computation
            fA = - grad_r * r_vector # [force, number of trajectories]
            fB = grad_r * r_vector
            f = torch.stack((fA, fB), dim=1)
            # HACK: same mass for all bodies
            dvdt = f / self.mass
            dxdt = v
            
            # get torque and update rotational motion
            dqdt = 0.5 * quatvec(q, w)
            # TODO: test out all matrix multiplications are correct
            # TODO: try assigning G to speed up computation as we re-use it once
            # TODO: if too slow to assign, do it in a single line
            l = w * self.inertia[None, :, :]
            # TODO: can I help out somehow the neural net with torque calculation by knowing the physical aspect - the torques must be opposite and equal? similarly to force 
            dldt = - torch.matmul(self.Omega(q, dqdt), l.unsqueeze(-1)).squeeze(-1) - 0.5 * torch.matmul(self.G(q), grad_q.unsqueeze(-1)).squeeze(-1) / self.inertia[None, :, :]
            dwdt = dldt / self.inertia[None, :, :]
            
        return (dvdt, dwdt, dxdt, dqdt)

    def harmonic_restraint(self, diff):
        # TODO: train different ks separately, or do a batch of k spring constants, that you update with each get_batch?
        return 0.5 * self.k * torch.square(diff)

    def G(self, q):
        # TODO: move this somewhere; make sure it's fast; maybe torch.stack is not ideal
        # TODO: add documentation
        G = torch.zeros(q.shape[0], q.shape[1], q.shape[2] - 1, q.shape[2]).to(q.device).type(torch.float64)
        G[:, :, 0, :] = torch.stack((-q[:, :, 1], q[:, :, 0], q[:, :, 3], -q[:, :, 2]), dim=-1)
        G[:, :, 1, :] = torch.stack((-q[:, :, 2], -q[:, :, 3], q[:, :, 0], q[:, :, 1]), dim=-1)
        G[:, :, 2, :] = torch.stack((-q[:, :, 3], q[:, :, 2], -q[:, :, 1], q[:, :, 0]), dim=-1)
        return G

    def Omega(self, q, dqdt):
        # TODO: move this somewhere
        return 2 * torch.matmul(self.G(q), torch.transpose(self.G(dqdt), 2, 3))
