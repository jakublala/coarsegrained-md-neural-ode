import torch

def all_loss_func(pred_y, true_y):
    return torch.mean(torch.abs(pred_y - true_y))

def final_loss_func(pred_y, true_y):
    return torch.mean(torch.abs(pred_y[:, -1, :, :] - true_y[:, -1, :, :]))
