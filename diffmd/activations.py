import torch
import torch.nn as nn

def get_activation_functions(function, nn_widths):
    def get_function(string, width):
        if string == 'relu':
            return nn.ReLU()
        elif string == 'leaky_relu':
            return nn.LeakyReLU()
        elif string == 'tanh':
            return nn.Tanh()
        elif string == 'sigmoid':
            return nn.Sigmoid()
        elif string == 'gelu':
            return nn.GELU()
        elif string == 'elu':
            return nn.ELU()
        elif string =='softexp':
            return SoftExp()
        else:
            raise Exception('activation function not implemented')

    if type(function) == str:
        return [get_function(function, width) for width in nn_widths]
    elif type(function) == list and len(function) == len(nn_widths):
        return [get_function(i) for i in function]
    else:
        raise Exception('activation function must be a string or a list of strings of the same length as the number of layers')

class SoftExp(nn.Module):
    def __init__(self, alpha = None):
        super(SoftExp,self).__init__()
        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha

