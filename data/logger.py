import numpy as np
import pandas as pd

class Logger():
    def __init__(self):
        self.header = ['epoch', 'itr', 'lr', 'traj_length', 'train_loss', 'test_loss', 'time']
        self.log = []

    def update(self, update):
        # update is a list of all metered values
        # [epoch, itr, lr, traj length, training loss, test loss, itr time]
        self.log.append(update)

    def save_csv(self, folder):
        pd.DataFrame(np.array(self.log), columns=self.header).to_csv(f'{folder}/log.csv', index=None)