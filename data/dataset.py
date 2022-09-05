from re import A
import torch
import numpy as np
import os

from data.trajectory import Trajectory
import torch.utils.data
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_type, batch_length, dataset_fraction=None):
        self.device = config['device']
        self.dtype = config['dtype']
        self.batch_length = batch_length
        self.dataset_fraction = dataset_fraction

        if dataset_type == 'train':
            self.max_batch_length = self.batch_length + int(config['epochs'] / config['batch_length_freq']) * config['batch_length_step']
            self.traj_step = config['traj_step']
            self.eval_init_skip = None
        else:
            self.max_batch_length = self.batch_length
            self.traj_step = config['eval_traj_step']
            self.eval_init_skip = config['eval_init_skip']
            
        self.folder = self.set_folder(config, dataset_type)
        self.filenames = self.get_filenames()
        self.trajs = self.get_trajectories()
        
        self.data = self.get_data()
        self.init_IDS = self.get_init_IDS()
        
        if self.dataset_fraction != None:
            self.init_IDS = self.get_fraction_IDS()

        self.stds = self.find_stds()
        self.means = self.find_means()

        # add logging in trajectory names of the used trajectories
        
    def __len__(self):
        return len(self.init_IDS)

    def __getitem__(self, index):
        init_id = self.init_IDS[index]
        traj_id, timestep_id = [int(i) for i in init_id.split('-')]
        dt = self.trajs[traj_id].dt * self.traj_step
        k = self.trajs[traj_id].k
        r0 = self.trajs[traj_id].r0
        inertia = self.trajs[traj_id].inertia
        
        # get initial condition
        init = (self.data[traj_id, timestep_id], dt, k, r0, inertia)
        
        # get true trajectory
        true_traj = self.data[traj_id, timestep_id:(timestep_id+(self.batch_length+1)*self.traj_step):self.traj_step]
        return init, true_traj

    def get_filenames(self):
        filenames = [f for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f))]
        filenames = [f.replace('-info.dat', '') for f in filenames if 'NVE' in f and '.dat' in f]
        return filenames

    def get_trajectories(self):
        return [Trajectory(self.folder+filename, self.device, self.dtype) for filename in self.filenames]

    def get_batch(self, batch_size, batch_length):
        # OBSOLUTE
        """
        Get a batch of data from a trajectory.
            
        Args:
            batch_size (int): number of trajectories in the batch
            batch_length (int): length of each trajectory in the batch
        
        Returns:
            batch_t (torch.Tensor): time steps for the batch
            pos_init (torch.Tensor): initial positions for the batch
            batch_trajs (tuple): batch of torch tensors containing the data (velocities, angular velocities, centre of masses, quaternions - in the form (nparticles, batch_size, batch_length, dim))
        """
        # choose a trajectory randomly
        trajectory = np.random.choice(self.trajs)
        dt = trajectory.dt
        traj = trajectory.traj

        nparticles = traj[0].shape[2]
        vel_dim = traj[0].size()[3]
        angvel_dim = traj[1].size()[3]
        com_dim = traj[2].size()[3]
        quat_dim = traj[3].size()[3]

        # assert vel_dim == 3, 'velocity dimension must be 3'
        # assert angvel_dim == 3, 'angular velocity dimension must be 3'
        # assert com_dim == 3, 'centre of mass dimension must be 3'
        # assert quat_dim == 4, 'quaternion dimension must be 4'

        sampled_is = torch.randint(traj[0].shape[0],size = (batch_size,)).to(self.device)
        sampled_js = torch.randint(traj[0].shape[1]-batch_length,size = (batch_size,)).to(self.device)
        initial_time = sampled_js*dt
    
        batch_t = torch.linspace(0.0,dt*(batch_length-1),batch_length).to(self.device)
        
        vels = traj[0][sampled_is,sampled_js,:,:].type(self.dtype)
        ang_vels = traj[1][sampled_is,sampled_js,:,:].type(self.dtype)
        coms = traj[2][sampled_is,sampled_js,:,:].type(self.dtype)
        quats = traj[3][sampled_is,sampled_js,:,:].type(self.dtype)
        
        pos_init = (vels, ang_vels, coms, quats)

        sampled_vels = []
        sampled_ang_vels = []
        sampled_coms = []
        sampled_quats = []
        for i in range(batch_size):
            vels = traj[0][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:]
            ang_vels = traj[1][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:]
            coms = traj[2][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:]
            quats = traj[3][sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:]
            
            sampled_vels.append(vels)
            sampled_ang_vels.append(ang_vels)
            sampled_coms.append(coms)
            sampled_quats.append(quats)
            
        sampled_vels = torch.stack(sampled_vels, dim=0).type(self.dtype)
        sampled_ang_vels = torch.stack(sampled_ang_vels, dim=0).type(self.dtype)
        sampled_coms = torch.stack(sampled_coms, dim=0).type(self.dtype)
        sampled_quats = torch.stack(sampled_quats, dim=0).type(self.dtype)
        batch_trajs = (sampled_vels, sampled_ang_vels, sampled_coms, sampled_quats)
        
        
        return batch_t, pos_init, batch_trajs, trajectory.k, trajectory.inertia.type(self.dtype), trajectory.file_path

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
                init_IDS += ids[:-self.max_batch_length*self.traj_step]
            else:
                init_IDS += ids[:-self.max_batch_length*self.traj_step:self.eval_init_skip]

        return init_IDS

    def get_fraction_IDS(self):
        num_inits = int(len(self.init_IDS) * self.dataset_fraction)
        random.shuffle(self.init_IDS)
        return self.init_IDS[:num_inits]

    def find_stds(self):
        p, l, x, q = torch.split(self.data, [3, 3, 3, 4], dim=-1)
        r = x[:, :, 1, :] - x[:, :, 0, :]
        p_std = torch.std(p, dim=1).mean()
        l_std = torch.std(l, dim=1).mean()
        r_std = torch.std(r, dim=1).mean()
        q_std = torch.std(q, dim=1).mean()
        return p_std, l_std, r_std, q_std

    def find_means(self):
        p, l, x, q = torch.split(self.data, [3, 3, 3, 4], dim=-1)
        r = x[:, :, 1, :] - x[:, :, 0, :]
        p_mean = torch.mean(p, dim=1).mean()
        l_mean = torch.mean(l, dim=1).mean()
        r_mean = torch.mean(r, dim=1).mean()
        q_mean = torch.mean(q, dim=1).mean()
        return p_mean, l_mean, r_mean, q_mean

    def update(self, batch_length=None, traj_step=None):
        if batch_length != None:
            self.batch_length = batch_length
        if traj_step != None:
            self.traj_step = traj_step

        assert (self.batch_length+1) * self.traj_step < self.data.shape[1], 'batch_length * traj_step must be less than the number of timesteps in the data'
        
