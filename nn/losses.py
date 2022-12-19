import torch

def set_loss_func(loss_func):
    if 'all-mse' == loss_func:
        raise NotImplementedError('all-mse loss function not implemented with new steps_per_dt')
        return all_mse
    elif 'final-mse' == loss_func:
        return final_mse
    elif 'all-mse-pos' == loss_func:
        raise NotImplementedError('all-mse loss function not implemented with new steps_per_dt')
        return all_mse_pos
    elif 'final-mse-pos' == loss_func:
        return final_mse_pos
    elif 'energy' == loss_func:
        return energy
    elif 'final-mse-pos-and-energy':
        return final_mse_pos_and_energy
    else:
        raise ValueError(f'loss function {loss_func} not recognised')

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

    losses = []
    for p, t in zip(pred, true):
        losses.append(torch.mean((p - t)**2))
        
    return torch.sum(torch.stack(losses)), [l.detach().cpu().item() for l in losses]

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

    losses = []
    for p, t in zip(pred, true):
        losses.append(torch.mean((p - t)**2))

    return torch.sum(torch.stack(losses)), [l.detach().cpu().item() for l in losses]

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

    losses = []
    for p, t in zip(pred, true):
        losses.append(torch.mean((p - t)**2))
        
    return torch.sum(torch.stack(losses[2:])), [0, 0] + [l.detach().cpu().item() for l in losses[2:]]

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

    losses = []
    for p, t in zip(pred, true):
        losses.append(torch.mean((p - t)**2))
        
    return torch.sum(torch.stack(losses[2:])), [0, 0] + [l.detach().cpu().item() for l in losses[2:]]

def energy(potential, pred_y, batch_energy):
    _, _, x, q = torch.split(pred_y[:, -1, :, :], [3, 3, 3, 4], dim=-1)
    r = x[:, 1, :] - x[:, 0, :]
    rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, 11).type(torch.float32)
    predicted_energies = potential(rq)
    return torch.mean((predicted_energies - batch_energy)**2)

def final_mse_pos_and_energy(potential, batch_energy, pred_y, true_y, stds, means, normalize=False, weight = 1.0):
    loss, loss_parts = final_mse_pos(pred_y, true_y, stds, means, normalize)
    energy_loss = weight * energy(potential, pred_y, batch_energy)
    return loss + energy_loss, loss_parts + [energy_loss.detach().cpu().item()]
