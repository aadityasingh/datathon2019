from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class GasTempDataset(Dataset):

    def __init__(self, csv, ins, thresh=None):
        self.data = pd.read_csv(csv)
        self.thresh = thresh # FOR NOW WE ASSUME THRESH IS NONE
        self.gases = ins

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input_vec': torch.from_numpy(self.data.iloc[idx][self.gases].values.astype(float)), 'output_num': self.data[idx]['avg_temp']}

        return sample

def make_data_loaders(path, ins, batch_size=16, val_split=0.2, thresh=None):
    dataset = GasTempDataset(path, ins)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

