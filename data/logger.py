import numpy as np
import pandas as pd

class Logger():
    def __init__(self):
        self.epoch = []
        self.itr = []
        self.lr = []
        self.traj_steps = []
        self.steps_per_dt = []
        self.train_loss = []
        self.avg_train_loss = []
        self.train_loss_vel = []
        self.train_loss_angvel = []
        self.train_loss_pos = []
        self.train_loss_quat = []
        self.train_loss_energy = []
        self.test_loss = []
        self.time = []

    def update(self, entry):
        for key, value in entry.items():
            self.__dict__[key].append(value)

    def save_csv(self, folder):
        df = pd.DataFrame(self.__dict__)
        df.to_csv(f'{folder}/log.csv', index=False)

    def load_previous(self, folder):
        df = pd.read_csv(f'{folder}/log.csv')
        for key in df.keys():
            self.__dict__[key] = df[key].to_list()