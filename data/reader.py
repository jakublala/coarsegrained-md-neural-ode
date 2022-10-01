import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Reader():

    def __init__(self, file_path):
        self.file_path = file_path
        # HACK: this is a hack to get the number of particles
        self.nparticles = 14
        self.timestep, self.runsteps, self.log_frequency = self.get_metadata()
        self.n_logged_timesteps = int(self.runsteps/self.log_frequency + 1)
        self.logged_timesteps = np.arange(0, self.runsteps+1, self.log_frequency)
        print(f"Found timestep: {self.timestep}, n of run steps: {self.runsteps}, and dump log freq: {self.log_frequency}, n of logged timesteps: {self.n_logged_timesteps}")
        
        # create processed csv file
        if not os.path.exists(self.file_path+'.csv'):
            self.create_dataframe()

    def create_dataframe(self):
        # get trajectories
        labels, lines = self.read_reduced_traj()
        df = pd.DataFrame(lines, columns=labels)
        
        # get energies (multiply to get system energy)
        labels, lines = self.read_simulation_log()
        data = np.array(lines)
        df['potential_energy'] = data[:, labels.index('PotEng')] * self.nparticles
        df['kinetic_energy'] = data[:, labels.index('KinEng')] * self.nparticles
        df['total_energy'] = data[:, labels.index('TotEng')] * self.nparticles

        # save to csv
        df.to_csv(self.file_path+'.csv', index=False)
        return


    def read_original_traj(self):
        subfix = '-traj.dump'
        lines = []
        with open(self.file_path+subfix, 'r') as f:
            particles_counter = 0
            particles_temp = []
            log_count = 0
            for i, line in enumerate(f):   
                if particles_counter == self.nparticles:
                    particles_temp = np.array(particles_temp, dtype=float)
                    particles_temp = particles_temp[particles_temp[:, 2].argsort()]
                    lines.append(particles_temp)
                    particles_counter = 0
                    particles_temp = []
                    log_count += 1
                    continue
                elif particles_counter > 0:
                    line = line.rstrip(' \n')
                    line = line.split()
                    line.pop(1) # remove 'type' column
                    particles_temp.append([str(self.logged_timesteps[log_count]), self.body_id(line[0])]+line)
                    particles_counter += 1
                else:
                    if i == 8:
                        line = line.lstrip('ITEM: ATOMS ')
                        labels = line.split()
                        labels.remove('type')
                        labels = ['timestep', 'body_id'] + labels
                    if (i-9) % (self.nparticles + 9) == 0:
                        line = line.rstrip(' \n')
                        line = line.split()
                        line.pop(1) # remove 'type' column
                        particles_temp.append([str(self.logged_timesteps[log_count]), self.body_id(line[0])]+line)
                        particles_counter += 1
        
        lines = np.vstack(lines)
            
        return labels, lines

    def read_reduced_traj(self):
        subfix = '-info.dat'
        lines = []
        log_count = 0
        with open(self.file_path+subfix, 'r') as f:
            for i, line in enumerate(f):
                if i == 8:
                    line = line.lstrip('ITEM: ENTRIES ')
                    labels = ['timestep'] + line.split()
                if (i+1) % 10 == 0:
                    line = line.rstrip(' \n')
                    line = [str(self.logged_timesteps[log_count])] + line.split()
                    lines.append(line)
                    log_count += 1
        lines = np.array(lines, dtype=float)
        return labels, lines

    def read_simulation_log(self):
        subfix = '-sim.log'
        with open(self.file_path+subfix, 'r') as f:
            for i, line in enumerate(f):
                if i < 6:
                    continue
                if i == 6:
                    label = line.split()
                    lines = np.zeros([self.n_logged_timesteps, len(label)])
                    step_no = 0
                else:
                    line = line.rstrip(' \n')
                    line = line.split()
                    try:
                        line = np.array(line, dtype=float)
                    except ValueError:
                        print(f"Breaking - likely reached last line of sim log: {line}")
                        break
                    lines[step_no] = line
                    step_no += 1
        return label, lines

    def get_metadata(self):
        subfix = '-input.log'
        with open(self.file_path+subfix, 'r') as f:
            counter = 0
            for i, line in enumerate(f):
                if line == '\n':
                    continue
                else:
                    counter += 1
                    if counter == 2:
                        log_frequency = int(line.split()[3])
                    elif counter == 3:
                        runsteps = int(line.split()[3])
                    elif counter == 4:
                        timestep = float(line.split()[3])
                        break
        return timestep, runsteps, log_frequency

    def body_id(self, atom_id):
        if (int(atom_id)) <= (self.nparticles // 2):
            return 1
        else:
            return 2 