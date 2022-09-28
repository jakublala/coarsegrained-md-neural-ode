import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

class Plotter():

    def __init__(self, trainer):
        self.trainer = trainer
        self.natoms = 14
        self.trajectory_index = 0

        self.df = self.get_dataframe()
        self.v, self.w, self.x, self.q, self.r, self.rq = self.get_trajectory()
        self.true_energies, self.predicted_energies = self.get_energies()
        self.harmonic_energy = self.get_harmonic_energy()
        self.LAMMPS_potential, self.LAMMPS_pot_start, self.LAMMPS_pot_end, self.LAMMPS_pot_steps = self.read_pair_potential()

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

    def get_harmonic_energy(self):
        # harmonic energy
        k = self.trainer.training_dataset.trajs[self.trajectory_index].k
        r0 = self.trainer.training_dataset.trajs[self.trajectory_index].r0
        harmonic_energy = (0.5 * k * torch.square(torch.norm(self.r, dim=1) - r0)).detach().cpu().numpy()
        return harmonic_energy

    def LAMMPS_energy_plot(self, num_steps):
        
        plt.title('Logged Energies from LAMMPS')
        plt.plot(self.df['total_energy'].to_numpy()[:num_steps] * self.natoms + self.harmonic_energy[:num_steps], 'k', label='Total')
        plt.plot(self.df['potential_energy'].to_numpy()[:num_steps] * self.natoms, 'b', label='Potential')
        plt.plot(self.df['kinetic_energy'].to_numpy()[:num_steps] * self.natoms, 'r', label='Kinetic')
        plt.plot(self.harmonic_energy[:num_steps], 'g', label='Harmonic')
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Energy')
        plt.savefig('figures/energies/LAMMPS_energy.png')
        plt.close()

    def plot_parity(self):
        indices = np.random.randint(500, size=self.rq.shape[0])
        fig, ax = plt.subplots()
        ax.plot(self.true_energies[indices] * self.natoms, self.predicted_energies[indices], 'bo', markersize=1)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        # set equal aspect ratio
        # ax.set_aspect('equal')

        min_y = self.predicted_energies[indices].min()
        max_y = self.predicted_energies[indices].max()

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
        plot_interval = interval.cpu().numpy()

        # case 1: hexagons facing each other (i.e. varying z-axis)
        temp_inp = inp.clone()
        temp_inp[:, 2] = interval
        temp_inp[:, 6] = 1.0
        temp_inp[:, -1] = 1.0
        predicted = potential(temp_inp).detach().cpu().numpy()

        plt.title('Hexagon Potential')
        plt.plot(plot_interval, predicted, 'r', label='NN')
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1], 'k-', label='LAMMPS')
        plt.legend()
        plt.xlabel('z-axis')
        plt.ylabel('Energy')
        plt.ylim(predicted.min() - 0.01 * predicted.min(), predicted.max() + 0.01 * predicted.max())
        plt.savefig('figures/energies/hexagon_potential_1Dface.png')
        plt.close()

        # case 2: hexagons in a single plane (i.e. varying x-axis)
        temp_inp = inp.clone()
        temp_inp[:, 0] = interval
        temp_inp[:, 6] = 1.0
        temp_inp[:, -1] = 1.0
        predicted = potential(temp_inp).detach().cpu().numpy()
        
        plt.title('Hexagon Potential')
        plt.plot(plot_interval, predicted, 'r', label='NN')
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1] , 'k-', label='LAMMPS')
        plt.legend()
        plt.xlabel('x-axis')
        plt.ylabel('Energy')
        plt.ylim(predicted.min() - 0.01 * predicted.min(), predicted.max() + 0.01 * predicted.max())
    
        plt.savefig('figures/energies/hexagon_potential_1DplaneX.png')
        plt.close()

        # case 3: hexagons in a single plane (i.e. varying y-axis)
        temp_inp = inp.clone()
        temp_inp[:, 1] = interval
        temp_inp[:, 6] = 1.0
        temp_inp[:, -1] = 1.0
        predicted = potential(temp_inp).detach().cpu().numpy()
        
        plt.title('Hexagon Potential')
        plt.plot(plot_interval, predicted, 'r', label='NN')
        plt.plot(self.LAMMPS_potential[:, 0], self.LAMMPS_potential[:, 1], 'k-', label='LAMMPS')
        plt.legend()
        plt.xlabel('y-axis')
        plt.ylabel('Energy')
        plt.ylim(predicted.min() - 0.01 * predicted.min(), predicted.max() + 0.01 * predicted.max())
    
        plt.savefig('figures/energies/hexagon_potential_1DplaneY.png')
        plt.close()



    def plot_traj_potential(self, num_steps):
        plt.plot(self.true_energies[:num_steps] * self.natoms, 'k', label='Actual')
        plt.plot(self.predicted_energies[:num_steps] - self.predicted_energies[0], 'r', label='Predicted')

        plt.ylabel('Energy')
        plt.xlabel('Time step')
        plt.legend()
        plt.savefig('figures/energies/trajectory_potential.png')
        plt.close()

    
            
