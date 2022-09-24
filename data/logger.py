import numpy as np
import pandas as pd

class Logger():
    def __init__(self):
        self.header = ['epoch', 'itr', 'lr', 'traj_steps', 'steps_per_dt', 'train_loss-vel', 'train_loss-angvel', 'train_loss-pos', 'train_loss-quat', 'test_loss', 'time']
        self.log = []

    def update(self, update):
        # update is a list of all metered values
        # ['epoch', 'itr', 'lr', 'traj_length', 'train_loss-vel', 'train_loss-angvel', 'train_loss-pos', 'train_loss-quat', 'test_loss', 'time']
        self.log.append(update)

    def save_csv(self, folder):
        pd.DataFrame(np.array(self.log), columns=self.header).to_csv(f'{folder}/log.csv', index=None)

    # TODO: add loading from previous CSV