import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

def is_valid_split(data_split, data_split, random_seed):
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

def load_data(df):
    if is_valid_split(data_split):
        train_size = data_split[0]
        valid_size = data_split[1]
    else:
        print("value error")
        raise ValueError("data_split is not correctly formatted")

    # shuffles and splits indices into the train, validation, and test data sets
    num_imgs = len(train_dataset)
    indices = range(num_imgs)
    split1 = int(np.floor(train_size * num_imgs))
    split2 = int(split1 + np.floor(valid_size * num_imgs))
    indices_train, indices_valid, indices_test = indices[:split1], indices[split1:split2], indices[split2:]
    sampler_train = SubsetRandomSampler(indices_train)
    sampler_valid = SubsetRandomSampler(indices_valid)
    sampler_test = SubsetRandomSampler(indices_test)

    # Pandas dataframe to PyTorch tensor
    dataset = torch.tensor(df.values)

    # Create DataLoaders
    self.train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler_train, 
        num_workers=num_workers, pin_memory=pin_memory,
    )
    self.valid_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler_valid, 
        num_workers=num_workers, pin_memory=pin_memory,
    )
    self.test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler_test, 
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader
