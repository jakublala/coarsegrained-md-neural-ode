import torch

def all_loss_func(pred_y, true_y):
    return torch.mean(torch.abs(pred_y - true_y))

def final_loss_func(pred_y, true_y):
    return torch.mean(torch.abs(pred_y[:, -1, :, :] - true_y[:, -1, :, :]))

def all_pos_loss_func(pred_y, true_y):
    sep = torch.abs((pred_y[:, :, 1, 6:9] - pred_y[:, :, 0, 6:9]) - (true_y[:, :, 1, 6:9] - true_y[:, :, 0, 6:9]))
    quat = torch.abs(pred_y[:, :, :, 9:] - true_y[:, :, :, 9:]).view(sep.shape[0], sep.shape[1], -1)
    return torch.mean(torch.abs(torch.cat((sep, quat), dim=-1)))

def final_pos_loss_func(pred_y, true_y):
    sep = torch.abs((pred_y[:, -1, 1, 6:9] - pred_y[:, -1, 0, 6:9]) - (true_y[:, -1, 1, 6:9] - true_y[:, -1, 0, 6:9]))
    quat = torch.abs(pred_y[:, -1, :, 9:] - true_y[:, -1, :, 9:]).view(sep.shape[0], -1)  
    return torch.mean(torch.abs(torch.cat((sep, quat), dim=-1)))
