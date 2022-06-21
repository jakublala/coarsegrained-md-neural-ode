import torch
import numpy as np
import os

from data.trajectory import Trajectory
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_type):
        self.device = config['device']
        self.dtype = config['dtype']
        self.batch_length = config['batch_length']

        self.folder = self.set_folder(config, dataset_type)
        self.filenames = self.get_filenames()
        self.trajs = self.get_trajectories()
        self.data = self.get_data()
        self.init_IDS = self.get_init_IDS()
        
    def __len__(self):
        return len(self.init_IDS)

    def __getitem__(self, index):
        return

    def get_filenames(self):
        filenames = [f for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f))]
        filenames = [f.replace('-info.dat', '') for f in filenames if 'NVE' in f and '.dat' in f]
        return filenames

    def get_trajectories(self):
        return [Trajectory(self.folder+filename, self.device) for filename in self.filenames]

    def get_batch(self, batch_size, batch_length):
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
        for t in self.trajs:
            print(t.traj.shape)
            


    # def get_init_IDS(self):
    #     init_IDS = []
    #     for traj in self.trajs:
    #         init_IDS.append(traj.reader.init_IDS)
        
