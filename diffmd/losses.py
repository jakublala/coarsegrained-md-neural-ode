import torch

def all_mse(pred_y, true_y, stds, means, normalize=False):
    pred = list(torch.split(pred_y, [3, 3, 3, 4], dim=-1))
    true = list(torch.split(true_y, [3, 3, 3, 4], dim=-1))
    
    pred[2] = pred[2][:, :, 1, :] - pred[2][:, :, 0, :]
    true[2] = true[2][:, :, 1, :] - true[2][:, :, 0, :]

    if normalize:
        for i, mean in enumerate(means):
            pred[i] = pred[i] - mean
            pred[i] = pred[i] / stds[i]
            true[i] = true[i] - mean
            true[i] = true[i] / stds[i]

    # for i, label in enumerate(['p', 'l', 'r', 'q']):
    #     print(f'{label} mse: {torch.mean((pred[i] - true[i])**2)}')

    return torch.mean((pred[0] - true[0])**2) + torch.mean((pred[1] - true[1])**2) + torch.mean((pred[2] - true[2])**2) + torch.mean((pred[3] - true[3])**2)

def final_mse(pred_y, true_y, stds, means, normalize=False):
    pred = list(torch.split(pred_y[:, -1, :, :], [3, 3, 3, 4], dim=-1))
    true = list(torch.split(true_y[:, -1, :, :], [3, 3, 3, 4], dim=-1))

    pred[2] = pred[2][:, 1, :] - pred[2][:, 0, :]
    true[2] = true[2][:, 1, :] - true[2][:, 0, :]

    if normalize:
        for i, mean in enumerate(means):
            pred[i] = pred[i] - mean
            pred[i] = pred[i] / stds[i]
            true[i] = true[i] - mean
            true[i] = true[i] / stds[i]

    return torch.mean((pred[0] - true[0])**2) + torch.mean((pred[1] - true[1])**2) + torch.mean((pred[2] - true[2])**2) + torch.mean((pred[3] - true[3])**2)

def all_mse_pos(pred_y, true_y, stds, means, normalize=False):
    pred = list(torch.split(pred_y, [3, 3, 3, 4], dim=-1))
    true = list(torch.split(true_y, [3, 3, 3, 4], dim=-1))
    
    pred[2] = pred[2][:, :, 1, :] - pred[2][:, :, 0, :]
    true[2] = true[2][:, :, 1, :] - true[2][:, :, 0, :]
    
    if normalize:
        for i, mean in enumerate(means):
            pred[i] = pred[i] - mean
            pred[i] = pred[i] / stds[i]
            true[i] = true[i] - mean
            true[i] = true[i] / stds[i]

    # for i, label in enumerate(['p', 'l', 'r', 'q']):
    #     print(f'{label} mse: {torch.mean((pred[i] - true[i])**2)}')

    return torch.mean((pred[2] - true[2])**2) + torch.mean((pred[3] - true[3])**2)

def final_mse_pos(pred_y, true_y, stds, means, normalize=False):
    pred = list(torch.split(pred_y[:, -1, :, :], [3, 3, 3, 4], dim=-1))
    true = list(torch.split(true_y[:, -1, :, :], [3, 3, 3, 4], dim=-1))

    pred[2] = pred[2][:, 1, :] - pred[2][:, 0, :]
    true[2] = true[2][:, 1, :] - true[2][:, 0, :]

    if normalize:
        for i, mean in enumerate(means):
            pred[i] = pred[i] - mean
            pred[i] = pred[i] / stds[i]
            true[i] = true[i] - mean
            true[i] = true[i] / stds[i]

    return torch.mean((pred[2] - true[2])**2) + torch.mean((pred[3] - true[3])**2)

def energy(potential, pred_y, batch_energy):
    _, _, x, q = torch.split(pred_y[:, -1, :, :], [3, 3, 3, 4], dim=-1)
    r = x[:, 1, :] - x[:, 0, :]
    rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, 11).type(torch.float32)
    predicted_energies = potential(rq)
    return torch.mean((predicted_energies - batch_energy)**2)


# def all_loss_func(pred_y, true_y):
#     return torch.mean(torch.abs(pred_y - true_y))

# def final_loss_func(pred_y, true_y):
#     return torch.mean(torch.abs(pred_y[:, -1, :, :] - true_y[:, -1, :, :]))

# def all_pos_loss_func(pred_y, true_y):
#     sep = torch.abs((pred_y[:, :, 1, 6:9] - pred_y[:, :, 0, 6:9]) - (true_y[:, :, 1, 6:9] - true_y[:, :, 0, 6:9]))
#     quat = torch.abs(pred_y[:, :, :, 9:] - true_y[:, :, :, 9:]).view(sep.shape[0], sep.shape[1], -1)
#     return torch.mean(torch.abs(torch.cat((sep, quat), dim=-1)))

# def final_pos_loss_func(pred_y, true_y):
#     sep = torch.abs((pred_y[:, -1, 1, 6:9] - pred_y[:, -1, 0, 6:9]) - (true_y[:, -1, 1, 6:9] - true_y[:, -1, 0, 6:9]))
#     quat = torch.abs(pred_y[:, -1, :, 9:] - true_y[:, -1, :, 9:]).view(sep.shape[0], -1)  
#     return torch.mean(torch.abs(torch.cat((sep, quat), dim=-1)))

# def all_loss_func_2(pred_y, true_y):
#     return torch.mean((pred_y - true_y)**2)

# def final_loss_func_2(pred_y, true_y):
#     return torch.mean((pred_y[:, -1, :, :] - true_y[:, -1, :, :])**2)

# def all_loss_func_2(pred_y, true_y, max_p, max_l, max_x):
#     (pred_p, pred_l, pred_x, pred_q) = torch.split(pred_y, [3, 3, 3, 4], dim=-1)
#     (true_p, true_l, true_x, true_q) = torch.split(true_y, [3, 3, 3, 4], dim=-1)
    
#     # relative momentum
#     diff_p = (pred_p[:, :, 1, :] - pred_p[:, :, 0, :]) - (true_p[:, :, 1, :] - true_p[:, :, 0, :])
#     loss_p = torch.mean((diff_p / max_p)**2)
    
#     # angular momentum
#     diff_l = (pred_l - true_l)
#     loss_l = torch.mean((diff_l / max_l)**2)

#     # relative separation
#     diff_r = (pred_x[:, :, 1, :] - pred_x[:, :, 0, :]) - (true_x[:, :, 1, :] - true_x[:, :, 0, :])
#     loss_r = torch.mean((diff_r / max_x)**2)

#     # quaternions
#     diff_q = (pred_q - true_q)
#     loss_q = torch.mean(diff_q**2)

#     return loss_p + loss_l + loss_r + loss_q

# def final_loss_func_2(pred_y, true_y):
#     return torch.mean((pred_y[:, -1, :, :] - true_y[:, -1, :, :])**2)

# def all_pos_loss_func_2(pred_y, true_y, max_p, max_l, max_x):
#     (_, _, pred_x, pred_q) = torch.split(pred_y, [3, 3, 3, 4], dim=-1)
#     (_, _, true_x, true_q) = torch.split(true_y, [3, 3, 3, 4], dim=-1)

#     # relative separation
#     diff_r = (pred_x[:, :, 1, :] - pred_x[:, :, 0, :]) - (true_x[:, :, 1, :] - true_x[:, :, 0, :])
#     loss_r = torch.mean((diff_r / max_x)**2)

#     # quaternions
#     diff_q = (pred_q - true_q)
#     loss_q = torch.mean(diff_q**2)
#     return loss_r + loss_q

# def final_pos_loss_func_2(pred_y, true_y, max_p, max_l, max_x):
#     (_, _, pred_x, pred_q) = torch.split(pred_y[:, -1, :, :], [3, 3, 3, 4], dim=-1)
#     (_, _, true_x, true_q) = torch.split(true_y[:, -1, :, :], [3, 3, 3, 4], dim=-1)

#     # relative separation
#     diff_r = (pred_x[:, 1, :] - pred_x[:, 0, :]) - (true_x[:, 1, :] - true_x[:, 0, :])
#     loss_r = torch.mean((diff_r / max_x)**2)

#     # quaternions
#     diff_q = (pred_q - true_q)
#     loss_q = torch.mean(diff_q**2)

#     return loss_r + loss_q

# def final_pos_percentage_loss_func_2(pred_y, true_y, max_p, max_l, max_x):
#     (_, _, pred_x, pred_q) = torch.split(pred_y[:, -1, :, :], [3, 3, 3, 4], dim=-1)
#     (_, _, true_x, true_q) = torch.split(true_y[:, -1, :, :], [3, 3, 3, 4], dim=-1)

#     # relative separation
#     diff_r = ((pred_x[:, 1, :] - pred_x[:, 0, :]) - (true_x[:, 1, :] - true_x[:, 0, :])) / (true_x[:, 1, :] - true_x[:, 0, :])
#     loss_r = torch.mean((diff_r)**2)

#     # quaternions
#     diff_q = (pred_q - true_q)  / true_q
#     loss_q = torch.mean((diff_q)**2)

#     return loss_r + loss_q 


# def all_pos_loss_func_2(pred_y, true_y):
#     sep = torch.abs((pred_y[:, :, 1, 6:9] - pred_y[:, :, 0, 6:9]) - (true_y[:, :, 1, 6:9] - true_y[:, :, 0, 6:9]))
#     quat = torch.abs(pred_y[:, :, :, 9:] - true_y[:, :, :, 9:]).view(sep.shape[0], sep.shape[1], -1)
#     return torch.mean((torch.cat((sep, quat), dim=-1))**2)

# def final_pos_loss_func_2(pred_y, true_y):
#     sep = torch.abs((pred_y[:, -1, 1, 6:9] - pred_y[:, -1, 0, 6:9]) - (true_y[:, -1, 1, 6:9] - true_y[:, -1, 0, 6:9]))
#     quat = torch.abs(pred_y[:, -1, :, 9:] - true_y[:, -1, :, 9:]).view(sep.shape[0], -1)  
#     return torch.mean((torch.cat((sep, quat), dim=-1))**2)
