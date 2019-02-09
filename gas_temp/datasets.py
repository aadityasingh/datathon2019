from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GasTempDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv, ins=[6,7,8,9,10,11,12,13,14,15,16], thresh=None):
        self.data = pd.read_csv(csv)
        self.thresh = thresh # FOR NOW WE ASSUME THRESH IS NONE
        self.gas_nums = ins
        self.gas_names = self.gases.columns[ins]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input_vec': self.data.iloc[idx, self.gas_nums].values, 'output_num': self.data[idx]['avg_temp']}

        return sample

# Example use:
# gtd = GasTempDataset("ReadFromMe.csv")
# dataloader = DataLoader(gtd, batch_size=4, shuffle=True)