import torch

def LambdaLR(epoch, alpha):
    return lambda epoch: alpha ** epoch
