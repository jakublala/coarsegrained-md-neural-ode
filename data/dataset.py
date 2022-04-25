from data.reader import Reader
import torch
import quaternion
import pandas as pd
import numpy as np


class Dataset():

    def __init__(self, filename):
        self.filename = filename
        # TODO: add the below into Reader
        self.temp, self.k, self.r0, self.seed = self.get_metadata_from_filename(self.filename)
        self.reader = Reader(filename)
        self.traj, self.inertia = self.get_traj(self.reader)
        self.dt = self.get_dt(self.reader)

    def get_traj(self, reader):
        # TODO: add options to train on just a part of trajectory
        # TODO: documentation
        log_labels, log_lines = reader.read_simulation_log()
        traj_labels, traj_lines = reader.read_reduced_traj(save=True)
        orig_labels, orig_lines = reader.read_original_traj(save=True)

        centre_of_masses, quaternions, velocities, ang_velocities, ang_momenta, inertia = get_data(filename)

        inertia = self.process_inertia(inertia)
        trajectory = self.process_data(centre_of_masses, quaternions, velocities, ang_velocities, inertia)
        return trajectory, inertia

    def get_metadata_from_filename(self, filename):
        # TODO: documentation
        filename = filename.split('_')
        temp = filename[0][filename[0].find('temp') + (4+1):]
        k = filename[1][filename[1].find('k') + (1+1):]
        r0 = filename[2][filename[2].find('r0') + (2+1):]
        seed = filename[3][filename[3].find('s') + (1+1):]
        return temp, k, r0, seed
    
    def get_dt(self, reader):
        log_freq = reader.log_frequency
        time_step = reader.timestep
        return time_step * log_freq

    def get_data(file_name):
        # train_split = 0.9
        # test_split = 1 - train_split
        df = pd.read_csv(file_name+'-reduced_traj.csv')
        # HACK: do this based on the column names, not explicitly
        com = ['c_com_1[1]', 'c_com_1[2]', 'c_com_1[3]', 'c_com_2[1]', 'c_com_2[2]', 'c_com_2[3]']
        q = ['c_q_1[1]', 'c_q_1[2]', 'c_q_1[3]', 'c_q_1[4]', 'c_q_2[1]', 'c_q_2[2]', 'c_q_2[3]', 'c_q_2[4]']
        vel = ['c_vel_1[1]', 'c_vel_1[2]', 'c_vel_1[3]', 'c_vel_2[1]', 'c_vel_2[2]', 'c_vel_2[3]'] 
        av = ['c_av_1[1]', 'c_av_1[2]', 'c_av_1[3]', 'c_av_2[1]', 'c_av_2[2]', 'c_av_2[3]']
        am = ['c_am_1[1]', 'c_am_1[2]', 'c_am_1[3]', 'c_am_2[1]', 'c_am_2[2]', 'c_am_2[3]']
        inertia = ['c_i_1[1]', 'c_i_1[2]', 'c_i_1[3]', 'c_i_2[1]', 'c_i_2[2]', 'c_i_2[3]']
        
        centre_of_masses = df.loc[:, ['timestep', *com]]
        quaternions = df.loc[:, ['timestep', *q]]
        velocities = df.loc[:, ['timestep', *vel]]
        ang_velocities = df.loc[:, ['timestep', *av]]
        ang_momenta = df.loc[:, ['timestep', *am]]
        # inertia = df.loc[0, ['timestep', *inertia]]
        inertia = df.loc[:, ['timestep', *inertia]]
        # TODO: use DataLoaders?
        
        # trajs = np.load('data/trajectories/diatomic_spring_narrow.npy')
        # split_index = int(trajs.shape[0] * train_split)

        # np.random.shuffle(trajs)
        # training_trajs = torch.Tensor(trajs[:split_index, :, :]).to(device)
        # testing_trajs = torch.Tensor(trajs[split_index:, :, :]).to(device)
        
        return centre_of_masses, quaternions, velocities, ang_velocities, ang_momenta, inertia

    def process_inertia(inertia):
        assert np.all(inertia.std().iloc[1:].to_numpy() == 0), 'inertia is not constant'
        inertia = inertia.iloc[0, 1:].to_numpy().reshape(2, 3)
        return torch.from_numpy(inertia).to(device)

    def process_data(centre_of_masses, quaternions, velocities, ang_velocities, inertia):
        # TODO: swap -1 and nparticles in view to avoid swapping axes later on
        # HACK: send a single trajectory
        ntraj = 1

        # HACK: pair potential
        nparticles = 2
        vel_dim = 3
        angvel_dim = 3
        com_dim = 3
        quat_dim = 4

        # Get centre of masses
        com1 = centre_of_masses.loc[:, ['c_com_1[1]', 'c_com_1[2]', 'c_com_1[3]']].to_numpy()
        com2 = centre_of_masses.loc[:, ['c_com_2[1]', 'c_com_2[2]', 'c_com_2[3]']].to_numpy()
        # separation = np.linalg.norm(com1-com2, axis=1).reshape(-1, 1)
        coms = torch.from_numpy(np.hstack((com1, com2))).to(device).view(ntraj, -1, nparticles, com_dim)
        
        # Get quaternion rotations (swap axes to put real part first)
        quat1 = quaternions.loc[:, ['c_q_1[4]', 'c_q_1[1]', 'c_q_1[2]', 'c_q_1[3]']].to_numpy()
        quat2 = quaternions.loc[:, ['c_q_2[4]', 'c_q_2[1]', 'c_q_2[2]', 'c_q_2[3]']].to_numpy()
        quats = torch.from_numpy(np.hstack((quat1, quat2))).to(device).view(ntraj, -1, nparticles, quat_dim)
        
        # Get translation velocities
        vel1 = velocities.loc[:, ['c_vel_1[1]', 'c_vel_1[2]', 'c_vel_1[3]']].to_numpy()
        vel2 = velocities.loc[:, ['c_vel_2[1]', 'c_vel_2[2]', 'c_vel_2[3]']].to_numpy() 
        # hexagon_mass = 7.0
        # mom = torch.from_numpy(np.hstack((vel1 * hexagon_mass, vel2 * hexagon_mass))).to(device).view(ntraj, -1, nparticles, vel_dim)
        vel = torch.from_numpy(np.hstack((vel1, vel2))).to(device).view(ntraj, -1, nparticles, vel_dim)

        # Get angular velocities
        ang_vel_1 = ang_velocities.loc[:, ['c_av_1[1]', 'c_av_1[2]', 'c_av_1[3]']].to_numpy()
        ang_vel_2 = ang_velocities.loc[:, ['c_av_2[1]', 'c_av_2[2]', 'c_av_2[3]']].to_numpy()
        # Convert system to body coords
        # TODO: we could do this when writing to CSV to speed up training
        quat1 = quaternion.from_float_array(quat1)
        quat2 = quaternion.from_float_array(quat2)
        ang_vel_1 = quaternion.from_vector_part(ang_vel_1)
        ang_vel_2 = quaternion.from_vector_part(ang_vel_2)
        ang_vel_1 = quat1.conj() * ang_vel_1 * quat1
        ang_vel_2 = quat2.conj() * ang_vel_2 * quat2
        ang_vel_1 = quaternion.as_vector_part(ang_vel_1)
        ang_vel_2 = quaternion.as_vector_part(ang_vel_2)
        ang_vel = torch.from_numpy(np.hstack((ang_vel_1, ang_vel_2))).to(device).view(ntraj, -1, nparticles, angvel_dim)
        return (vel, ang_vel, coms, quats)
