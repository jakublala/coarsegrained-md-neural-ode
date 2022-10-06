import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, nparticles, dim, widths, functions, dtype):
        super(BaselineModel, self).__init__()
        self.dim = dim
        self.nparticles = nparticles
        self.dtype = dtype
        self.mass = 7.0 # HACK
        self.kwargs = {'dim': dim, 'widths': widths, 'functions': functions, 'dtype': dtype}

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

        # initialise NN parameters
        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,val=0)

    def forward(self, state):
        _, _, x, q = state
        r = x[:, 1, :] - x[:, 0, :]
        rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, self.dim)
        return self.net(rq)
