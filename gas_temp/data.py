import os
import copy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd

def is_valid_split(data_split):
    if len(data_split) == 2:
        for ratio in data_split:
            if ratio < 0 or ratio > 1:
                return False
        return data_split[0]+data_split[1] <= 1
    if len(data_split) == 3:
        for ratio in data_split:
            if ratio < 0 or ratio > 1:
                return False
        return data_split[0] + data_split[1] + data_split[2] == 1
    return False

def load_data(df, data_split, num_workers=4, pin_memory=False):
    if is_valid_split(data_split):
        train_size = data_split[0]
        valid_size = data_split[1]
    else:
        print("value error")
        raise ValueError("data_split is not correctly formatted")

    # Pandas dataframe to PyTorch tensor
    train_dataset = torch.tensor(df.values)
    valid_dataset = torch.tensor(copy.deepcopy(df.values))
    test_dataset = torch.tensor(copy.deepcopy(df.values))

    # shuffles and splits indices into the train, validation, and test data sets
    num_imgs = len(train_dataset)
    indices = range(num_imgs)
    split1 = int(np.floor(train_size * num_imgs))
    split2 = int(split1 + np.floor(valid_size * num_imgs))
    indices_train, indices_valid, indices_test = indices[:split1], indices[split1:split2], indices[split2:]
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_valid = SubsetRandomSampler(indices_valid)
    sampler_test = SubsetRandomSampler(indices_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, sampler=sampler_train, 
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, sampler=sampler_valid, 
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, sampler=sampler_test, 
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader
