import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Reader():

    def __init__(self, file_name):
        self.file_name = file_name
        self.nparticles = 14
        self.timestep, self.runsteps, self.log_frequency = self.get_metadata()
        self.n_logged_timesteps = int(self.runsteps/self.log_frequency + 1)
        self.logged_timesteps = np.arange(0, self.runsteps+1, self.log_frequency)
       
    def read_original_traj(self, save=True):
        subfix = '-traj.dump'
        lines = []
        with open(self.file_name+subfix, 'r') as f:
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
        if save:
            df = pd.DataFrame(lines, columns=labels)
            df.to_csv(self.file_name+'-original_traj.csv', index=False)
            
        return labels, lines

    def read_reduced_traj(self, save=True):
        subfix = '-info.dat'
        lines = []
        log_count = 0
        with open(self.file_name+subfix, 'r') as f:
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

        # TODO: add a column for body_id rather than have so many columns
        if save:
            df = pd.DataFrame(lines, columns=labels)
            df.to_csv(self.file_name+'-reduced_traj.csv', index=False)
        
        return labels, lines

    def read_simulation_log(self):
        # TODO: is this useful for anything? make this useful
        # TODO: it does not even work
        subfix = '-sim.log'
        with open(self.file_name+subfix, 'r') as f:
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
        with open(self.file_name+subfix, 'r') as f:
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

        print(f"Found timestep: {timestep}, n of run steps: {runsteps}, and dump log freq: {log_frequency}")
        return timestep, runsteps, log_frequency

    def body_id(self, atom_id):
        if (int(atom_id)) <= (self.nparticles // 2):
            return 1
        else:
            return 2 