import pdb
import argparse

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from models import GasTempNet
from data import load_data
from train import Trainer

# gases = ['TS04', 'TNH4', 'CA', 'MG', 'NA', 'K', 'CL', 'NSO4', 'NHNO3', 'WSO2', 'WNO3']
gases = ['a', 'b']

input_size = len(gases)
output_size = 1

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--test_every', type = int, default = 100)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_step', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--decay', type=float, default=0.995)

    return parser

def train(df, opts):
    training_params = {'num_epochs' : opts.epochs, 'learning_rate' : opts.lr, 'weight_decay' : 0.3, 
        'learning_rate_decay' : opts.decay, 'learning_rate_step': opts.lr_step, 'cuda' : False, 
        'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/', 'hidden_dim': opts.hidden_dim}

    model = GasTempNet(training_params['hidden_dim'], input_size, output_size)

    if opts.checkpoint == 1:
        checkpoint = torch.load('./runs/checkpoints/checkpoint2.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

    train_loader, val_loader, test_loader = load_data(df, (.8,.1,.1))

    trainer = Trainer(model, train_loader, test_loader, training_params)

    # print(trainer)

    trainer.train(opts)

if __name__ == "__main__":
    parser = create_parser()
    opts = parser.parse_args()
    
    import random
    dummy_data = []
    for i in range(10):
        a = random.random()
        b = random.random()
        dummy_data.append([a,b, a+b])
    df = pd.DataFrame(data=np.array(dummy_data))

    train(df, opts)