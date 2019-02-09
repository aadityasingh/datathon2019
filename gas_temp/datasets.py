from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PlantGasDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_plant, csv_gas, ins, outs, thresh=None):
        self.gases = pd.read_csv(csv_gas)
        self.plants = pd.read_csv(csv_plant)
        self.thresh = thresh
        self.inputs = dict()
        self.outputs = dict()
        for index, row in self.plants.iterrows():
            to_inc = self.inputs.setdefault(row['city']+', '+row['state_name'], [0]*len(ins))
            for i, source in enumerate(ins):
                to_inc[i] += row[source]
        for index, row in self.gases.iterrows():
            


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample