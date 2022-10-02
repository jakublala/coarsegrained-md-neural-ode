import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from matplotlib.offsetbox import AnchoredText

class Plotter():

    def __init__(self, trainer, dataset_steps = 10):
        self.trainer = trainer
        self.folder = self.trainer.load_folder
        
        self.create_folders()
        
        self.natoms = 14
        self.trajectory_index = 0

        self.df = self.get_dataframe()
        self.v, self.w, self.x, self.q, self.r, self.rq = self.get_trajectory()
        self.pred_v, self.pred_w, self.pred_x, self.pred_q, self.pred_r, self.pred_rq, self.pred_t = self.get_predicted_trajectories(dataset_steps)
        
        self.true_energies, self.predicted_energies = self.get_energies()
        self.harmonic_energy = self.get_harmonic_energy(self.r)
        self.kinetic_energy = self.get_kinetic_energy(self.v, self.w)
        
        try:
            self.LAMMPS_potential, self.LAMMPS_pot_start, self.LAMMPS_pot_end, self.LAMMPS_pot_steps = self.read_pair_potential()
        except:
            print('no LAMMPS pair potential file found')
            
    def create_folders(self):
        folders = ["figures/energies/lammps", "figures/energies/nn"]
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)

    def get_dataframe(self):
        df = pd.read_csv(self.trainer.training_dataset.trajs[self.trajectory_index].file_path+'.csv')
        return df

    def get_trajectory(self):
        data = self.trainer.training_dataset.trajs[self.trajectory_index].traj
        data = [i.squeeze() for i in data]
        v, w, x, q = data
        r = x[:, 1, :] - x[:, 0, :]
        rq = torch.cat((r, q.reshape(-1, 8)), dim=1).reshape(-1, 11).type(torch.float32)
        return v, w, x, q, r, rq

    def get_energies(self):
        potential = self.trainer.func.net
        
        true_energies = self.df['potential_energy'].to_numpy()
        predicted_energies = potential(self.rq).detach().cpu().numpy()  

        return true_energies, predicted_energies

        
    def read_pair_potential(self):
        lines = []
        skip = True
        with open(f'{self.trainer.folder}/train/pair_potential.txt', 'r') as f:
            for line in f.readlines():
                if line[0] == "1":
                    skip = False
                if skip:
                    if line[0] == 'N':
                        num_steps = int(line.split()[1])
                        start, end = [float(i) for i in line.split()[3:]]
                    continue
                else:
                    if line[0] == "#":
                        break
                    line = line.replace("\n", "")
                    line = line.split()
                    line = [float(i) for i in line]
                    lines.append(line)

        data = np.array(lines, dtype=np.float32)
        return data[:, [1, 2]], start, end, num_steps

    def get_harmonic_energy(self, r):
        # harmonic energy
        k = self.trainer.training_dataset.trajs[self.trajectory_index].k
        r0 = self.trainer.training_dataset.trajs[self.trajectory_index].r0
        harmonic_energy = (0.5 * k * torch.square(torch.norm(r, dim=1) - r0)).detach().cpu().numpy()
        return harmonic_energy

    def get_kinetic_energy(self, v, w):
        M = 7.0
        kinetic_energy_trans = torch.sum(torch.sum(0.5 * M * v**2, dim=-1), dim=-1)
        batch_input, _, _ = self.trainer.training_dataset[0]
        inertia = batch_input[-1]
        kinetic_energy_rot = torch.sum(torch.sum(0.5 * inertia * w**2, dim=-1), dim=-1)
        kinetic_energy = (kinetic_energy_trans + kinetic_energy_rot).detach().cpu().numpy()

        plt.plot(kinetic_energy_rot.detach().cpu(), label='rot')
        plt.plot(kinetic_energy_trans.detach().cpu(), label='trans')
        plt.plot(kinetic_energy, label='tot')
        plt.legend()
        plt.savefig('figures/test.png')
        plt.close()

        return kinetic_energy


    def LAMMPS_energy_plot(self, num_steps):
        plt.title('Logged Energies from LAMMPS')
        plt.plot(self.df['total_energy'].to_numpy()[:num_steps] + self.harmonic_energy[:num_steps], 'k', label='Total')
        plt.plot(self.df['potential_energy'].to_numpy()[:num_steps], 'b', label='Potential')
        plt.plot(self.df['kinetic_energy'].to_numpy()[:num_steps], 'r', label='Kinetic')
        plt.plot(self.harmonic_energy[:num_steps], 'g', label='Harmonic')
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Energy')
        plt.savefig('figures/energies/LAMMPS_energy.png')
        plt.close()

    def plot_parity(self):
        predicted_energies = self.predicted_energies - self.predicted_energies[0]
        indices = np.random.randint(1000, size=self.rq.shape[0])
        fig, ax = plt.subplots()
        ax.plot(self.true_energies[indices], predicted_energies[indices], 'bo', markersize=1)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        ax.set_aspect('equal')

        min_y = predicted_energies[indices].min()
        max_y = predicted_energies[indices].max()

        min_x = self.true_energies[indices].min()
        max_x = self.true_energies[indices].max()
        
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.plot([min_x, min_x], [max_x, max_x], 'k-')
        fig.savefig('figures/energies/energy_matrix.png')
        plt.close(fig)


    def plot_pair_potential(self):
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1], label='energy')
        plt.legend()
        plt.savefig('figures/energies/pair_potential.png')
        plt.close()

    def plot_hexagon_potential(self):
        potential = self.trainer.func.net

        inp = torch.zeros((self.LAMMPS_pot_steps, 11)).to(self.trainer.device)
        # HACK
        extrapolation = 5
        interval = torch.linspace(self.LAMMPS_pot_start, self.LAMMPS_pot_end+extrapolation, self.LAMMPS_pot_steps)
        interval = torch.linspace(-5, 5, 500)
        plot_interval = interval.cpu().numpy()

        # case 1: hexagons facing each other (i.e. varying z-axis)
        temp_inp = inp.clone()
        temp_inp[:, 2] = interval
        temp_inp[:, 6] = 1.0
        temp_inp[:, -1] = 1.0
        predicted = potential(temp_inp).detach().cpu().numpy()
        
        plt.title('Hexagon Potential') 
        plt.plot(plot_interval, predicted, 'r', alpha=0.5, label='NN original')
        plt.plot(plot_interval, predicted - predicted[-1], 'r', label='NN shifted')
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1], 'k-', label='LAMMPS')
        plt.legend()
        plt.xlabel('z-axis')
        plt.ylabel('Energy')
        plt.ylim(-5, predicted.max() + 0.01 * predicted.max())
        plt.savefig('figures/energies/hexagon_potential_1Dface.png')
        plt.close()

        # case 2: hexagons in a single plane (i.e. varying x-axis)
        temp_inp = inp.clone()
        temp_inp[:, 0] = interval
        temp_inp[:, 6] = 1.0
        temp_inp[:, -1] = 1.0
        predicted = potential(temp_inp).detach().cpu().numpy()
        
        plt.title('Hexagon Potential')
        plt.plot(plot_interval, predicted, 'r', alpha=0.5, label='NN original')
        plt.plot(plot_interval, predicted - predicted[-1], 'r', label='NN shifted')
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1], 'k-', alpha=0.5, label='LAMMPS')
        plt.plot(self.LAMMPS_potential[:, 0] + 2, self.LAMMPS_potential[:, 1], 'k-', label='LAMMPS + 2')
        plt.legend()
        plt.xlabel('x-axis')
        plt.ylabel('Energy')
        plt.ylim(-5, predicted.max() + 0.01 * predicted.max())
    
        plt.savefig('figures/energies/hexagon_potential_1DplaneX.png')
        plt.close()

        # case 3: hexagons in a single plane (i.e. varying y-axis)
        temp_inp = inp.clone()
        temp_inp[:, 1] = interval
        temp_inp[:, 6] = 1.0
        temp_inp[:, -1] = 1.0
        predicted = potential(temp_inp).detach().cpu().numpy()
        
        plt.title('Hexagon Potential')
        plt.plot(plot_interval, predicted, 'r', alpha=0.5, label='NN original')
        plt.plot(plot_interval, predicted - predicted[-1], 'r', label='NN shifted')
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1], 'k-', alpha=0.5, label='LAMMPS')
        plt.plot(self.LAMMPS_potential[:, 0] + 2, self.LAMMPS_potential[:, 1], 'k-', label='LAMMPS + 2')
        plt.legend()
        plt.xlabel('y-axis')
        plt.ylabel('Energy')
        plt.ylim(-5, predicted.max() + 0.01 * predicted.max())
    
        plt.savefig('figures/energies/hexagon_potential_1DplaneY.png')
        plt.close()

    def plot_traj_potential(self, num_steps):
        plt.plot(self.true_energies[:num_steps], 'k', label='Actual')
        plt.plot(self.predicted_energies[:num_steps] - self.predicted_energies[0], 'r', label='Predicted')
        plt.ylabel('Energy')
        plt.xlabel('Time step')
        plt.legend()
        plt.savefig('figures/energies/trajectory_potential.png')
        plt.close()

    def get_predicted_trajectories(self, dataset_steps):
        batch_input, _, _ = self.trainer.training_dataset[0]
        batch_input = list(batch_input)
        batch_input[0] = batch_input[0].unsqueeze(0)
        batch_input = tuple(batch_input)

        steps_per_dt = 100
        traj_steps = dataset_steps * steps_per_dt

        with torch.no_grad():
            pred_y = self.trainer.forward_pass(batch_input, traj_steps, steps_per_dt=steps_per_dt)
            effective_dt = batch_input[1] / steps_per_dt
            pred_t = self.trainer.get_batch_t(effective_dt, traj_steps).detach().cpu().numpy()
        pred_y = list(torch.split(pred_y, [3, 3, 3, 4], dim=-1))
        pred_y = [x.squeeze(0) for x in pred_y]
        pred_y.append(pred_y[2][:, 1, :] - pred_y[2][:, 0, :])
        pred_y.append(torch.cat((pred_y[-1], pred_y[-2].reshape(-1, 8)), dim=1).reshape(-1, 11).type(torch.float32))        
        return *[i.detach() for i in pred_y], pred_t


    def traj_energies(self):
        potential_energy = self.trainer.func.net(self.pred_rq).squeeze().detach().cpu().numpy()
        kinetic_energy = self.get_kinetic_energy(self.pred_v, self.pred_w)
        harmonic_energy = self.get_harmonic_energy(self.pred_r)

        plt.plot(self.pred_t, potential_energy, 'b', label='Potential')
        plt.plot(self.pred_t, kinetic_energy, 'r', label='Kinetic')
        plt.plot(self.pred_t, harmonic_energy, 'g', label='Harmonic')

        total_energy = potential_energy + kinetic_energy + harmonic_energy
        plt.plot(self.pred_t, total_energy, 'k', label='Total')
      
        plt.title('Energies of Predicted Trajectory Using NN Potential')
        plt.ylabel('Energy')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.savefig('figures/energies/traj_energies.png')
        plt.close()
    



    def NN_energy(self, num_steps):
        plt.plot(self.predicted_energies[:num_steps], 'b-', label='Potential')
        plt.plot(self.harmonic_energy[:num_steps], 'g-', label='Harmonic')
        plt.plot(self.kinetic_energy[:num_steps], 'r-', label='Kinetic')
        total_energy = self.predicted_energies[:num_steps].squeeze() + self.harmonic_energy[:num_steps] + self.kinetic_energy[:num_steps]
        plt.plot(total_energy, 'k', alpha=0.8, label='Total ')        
        plt.legend()
        plt.title('NN Energy based on LAMMPS trajectory')
        plt.xlabel('Time step')
        plt.ylabel('Energy')
        plt.savefig('figures/energies/NN_energy.png')
        plt.close()

    def traj_distribution(self, var='r'):
        r_norm = torch.norm(self.__dict__[var], dim=-1)
        plt.hist(r_norm.flatten().detach().cpu().numpy(), bins=100)
        plt.ylabel('Occurences')
        plt.xlabel(f'{var}')
        plt.savefig(f'figures/hist_{var}.png')
        plt.close()


    def plot_traj(self, dataset, num_steps=100, checkpoint=False):
        # TODO: finish this and make it work, so that we can call it from Trainer
        def get_anchored_text():
            at = AnchoredText(f'epoch: Final', prop=dict(size=10), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            return at

        def plot(indices, body_id, title, filename):
            colours = ['r-', 'b-', 'g-', 'm-']
            fig, ax = plt.subplots()
            for c, i in enumerate(indices):
                ax.set_title(title)
                ax.plot(batch_t, batch_y[:,body_id,i], 'k--', alpha=0.3, label=f'true {i}')
                ax.plot(pred_t, pred_y[:,body_id,i], colours[c], alpha=0.5, label=f'pred {i}')
            ax.add_artist(get_anchored_text())
            fig.savefig(f'{subfolder}/{filename}.png')
            plt.close(fig)            


        subfolder = f'figures/trajs/{dataset}'
        if dataset == 'train':
            dataset = self.trainer.training_dataset
        elif dataset == 'test':
            dataset = self.trainer.test_dataset
        elif dataset == 'validate':
            dataset = self.trainer.validation_dataset
        else:
            raise ValueError('dataset must be train, test or validate')

        dataset_steps = num_steps

        traj_steps = dataset_steps * self.trainer.steps_per_dt
        
        # self.training_dataset.update(dataset_steps)
        with torch.no_grad():
            # get the earliest init conditions to ensure trajectories are long enough
            init_index = dataset.init_IDS.index(min(dataset.init_IDS, key=len))
            batch_input, batch_y, _ = dataset[init_index]
            batch_input = list(batch_input)
            batch_input[0] = batch_input[0].unsqueeze(0)
            batch_input = tuple(batch_input)

            pred_y = self.trainer.forward_pass(batch_input, traj_steps=traj_steps, steps_per_dt=self.trainer.steps_per_dt).squeeze().cpu().numpy()
            batch_y = batch_y.cpu().numpy()
            
            effective_dt = batch_input[1] / self.trainer.steps_per_dt
            pred_t = self.trainer.get_batch_t(effective_dt, traj_steps).cpu().numpy()
            batch_t = self.trainer.get_batch_t(batch_input[1], dataset_steps).cpu().numpy()

            print(pred_t.shape)
            print(batch_t.shape)

            print(pred_y.shape)
            print(batch_y.shape)

            ind_vel = [0, 1, 2]
            ind_ang = [3, 4, 5]
            ind_quat = [9, 10, 11, 12]

            for i in [0, 1]:
                plot(ind_vel, i, f'velocities {i+1}', f'vel{i+1}')
                plot(ind_ang, i, f'angular velocities {i+1}', f'angvel{i+1}')
                plot(ind_quat, i, f'quaternions {i+1}', f'quat{i+1}')
            
            # centre of mass positions (set initial position of first COM to zero)
            batch_y[:,:,6:9] = batch_y[:,:,6:9] - batch_y[:,[0],6:9]
            pred_y[:,:,6:9] = pred_y[:,:,6:9] - pred_y[:,[0],6:9]
            
            # centre of mass separation
            batch_y_sep = np.linalg.norm(batch_y[:,1,6:9] - batch_y[:,0,6:9], axis=-1)
            pred_y_sep = np.linalg.norm(pred_y[:,1,6:9] - pred_y[:,0,6:9], axis=-1)

            fig, ax = plt.subplots()
            ax.set_title('separation')
            ax.plot(batch_t, batch_y_sep, 'k--', alpha=0.3, label=f'true')
            ax.plot(pred_t, pred_y_sep, 'r-', alpha=0.5, label=f'pred')
            ax.add_artist(get_anchored_text())
            fig.savefig(f'{subfolder}/sep.png')
            plt.close(fig)

        # revert changes to traj length
        # self.training_dataset.update(self.dataset_steps)
    


