from re import A
import torch
import numpy as np
import os

from data.trajectory import Trajectory
import torch.utils.data
import random
from diffmd.utils import set_device, set_dtype


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_type, traj_length, dataset_fraction=None, random_dataset=False):
        self.device = set_device(config['device'])
        self.dtype = set_dtype(config['dtype'])
        self.traj_length = traj_length
        self.dataset_fraction = dataset_fraction
        self.random_dataset = random_dataset

        if dataset_type == 'train':
            # TODO: fix after implementing gradual traj length
            # HACK
            # self.max_batch_length = self.batch_length + int(config['epochs'] / config['batch_length_freq']) * config['batch_length_step']
            self.max_traj_length = traj_length
            self.eval_init_skip = None
        else:
            # TODO: fix after implementing gradual traj length
            # HACK
            # self.max_batch_length = self.batch_length
            self.max_traj_length = traj_length
            self.eval_init_skip = config['eval_init_skip']
            
        self.folder = self.set_folder(config, dataset_type)
        self.filenames = self.get_filenames()
        self.trajs = self.get_trajectories()
        
        self.data = self.get_data()
        self.energies = self.get_energies()

        self.init_IDS = self.get_init_IDS()
        
        if self.dataset_fraction != None:
            self.init_IDS = self.get_fraction_IDS()

        self.stds = self.find_stds()
        self.means = self.find_means()

    def __len__(self):
        return len(self.init_IDS)

    def __getitem__(self, index):
        init_id = self.init_IDS[index]
        traj_id, timestep_id = [int(i) for i in init_id.split('-')]
        dt = self.trajs[traj_id].dt
        k = self.trajs[traj_id].k
        r0 = self.trajs[traj_id].r0
        inertia = self.trajs[traj_id].inertia
        
        # get initial condition
        init = (self.data[traj_id, timestep_id], dt, k, r0, inertia)
        
        # get true trajectory
        batch_y = self.data[traj_id, timestep_id:(timestep_id+self.traj_length+1)]
        final_energy = self.energies[traj_id, timestep_id+self.traj_length]
        return init, batch_y, final_energy

    def get_filenames(self):
        filenames = [f for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f))]
        filenames = [f.replace('-info.dat', '') for f in filenames if 'NVE' in f and '.dat' in f]
        return filenames

    def get_trajectories(self):
        return [Trajectory(self.folder+filename, self.device, self.dtype) for filename in self.filenames]

    def get_energies(self):
        return torch.stack([torch.from_numpy(t.energies).to(self.device).type(self.dtype) for t in self.trajs])

    def set_folder(self, config, dataset_type):
        if dataset_type in ['train', 'validation', 'test']:
            return config['folder']+f'/{dataset_type}/'
        else:
            raise ValueError('dataset_type must be either train, test or validation')

    def get_data(self):
        data = [torch.cat(t.traj, dim=-1) for t in self.trajs]
        data = torch.cat(data, dim=0)
        return data.to(self.device)

    def get_init_IDS(self):
        init_IDS = []
        for traj_id, traj in enumerate(self.trajs):
            ids = list(range(traj.reader.n_logged_timesteps))
            ids = [f'{traj_id}-{i}' for i in ids]
            if self.eval_init_skip == None:
                init_IDS += ids[:-self.max_traj_length]
            else:
                init_IDS += ids[:-self.max_traj_length:self.eval_init_skip]

        return init_IDS

    def get_fraction_IDS(self):
        num_inits = int(len(self.init_IDS) * self.dataset_fraction)
        if self.random_dataset:
            random.shuffle(self.init_IDS)
        return self.init_IDS[:num_inits]

    def find_stds(self):
        p, l, x, q = torch.split(self.data, [3, 3, 3, 4], dim=-1)
        r = x[:, :, 1, :] - x[:, :, 0, :]
        p_std = torch.std(p, dim=1).mean().item()
        l_std = torch.std(l, dim=1).mean().item()
        r_std = torch.std(r, dim=1).mean().item()
        q_std = torch.std(q, dim=1).mean().item()
        return p_std, l_std, r_std, q_std

    def find_means(self):
        p, l, x, q = torch.split(self.data, [3, 3, 3, 4], dim=-1)
        r = x[:, :, 1, :] - x[:, :, 0, :]
        p_mean = torch.mean(p, dim=1).mean().item()
        l_mean = torch.mean(l, dim=1).mean().item()
        r_mean = torch.mean(r, dim=1).mean().item()
        q_mean = torch.mean(q, dim=1).mean().item()
        return p_mean, l_mean, r_mean, q_mean

    def update(self, traj_length=None):
        if traj_length != None:
            self.traj_length = traj_length
