# PLANT TO GAS NEURAL NETWORK MODEL HERE:
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm
import numpy as np
import utils

import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

from datasets import PlantGasDataset

# Input plant types we care about (listed for easy tuning)
sources = ['nuclear', 'wind', 'hydroelectric', 'natural gas', 'coal', 'pumped storage', 'biomass', 'solar', 'geothermal']
# Feature gases that we care about
gases = ['TSO4','TNH4','CA','MG','NA','K','CL','NSO4','NHNO3','WSO2','WNO3']
gases.remove('MG')
gases.remove('NA')
gases.remove('K')
gases.remove('CL')

pgd = PlantGasDataset(plantCSV, gasCSV, sources, gases)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 10000)
    parser.add_argument('--test_every', type = int, default = 100)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_step', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--decay', type=float, default=0.995)

    return parser

def train(opts):
    training_params = {'num_epochs' : opts.epochs, 'learning_rate' : opts.lr, 'weight_decay' : 0.3, 
        'learning_rate_decay' : opts.decay, 'learning_rate_step': opts.lr_step, 'cuda' : False, 
        'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/', 'hidden_dim': opts.hidden_dim}

    if opts.checkpoint == 1:
        checkpoint = torch.load('./runs/checkpoints/checkpoint2.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

    loss = Loss()

    trainer = Trainer(model, loss, train_loader, test_loader, training_params)

    # print(trainer)

    trainer.train(opts)

if __name__ == "__main__":
    parser = create_parser()
    opts = parser.parse_args()
    
    train(opts)