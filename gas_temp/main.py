import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from models import GasTempNet
from data import load_data

def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = utils.to_data(fixed_X), utils.to_data(fake_X)
    Y, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))

def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_every', type=int, default=6)
    opts = parser.parse_args()

# gases = ['TS04', 'TNH4', 'CA', 'MG', 'NA', 'K', 'CL', 'NSO4', 'NHNO3', 'WSO2', 'WNO3']
gases = ['a', 'b']

def run(df):
    # Hyper-parameters
    hidden_size = 100
    input_size = len(gases)
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Linear regression model
    model = GasTempNet(hidden_size, input_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    train_loader, val_loader, test_loader = load_data(df, (.8,.1,.1))

    for epoch in range(num_epochs):
        for i, row in enumerate(train_loader):
            inputs = row[:, :-1]
            labels = row[:, -1]
            print(inputs, labels)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, i+1, total_step, loss.item()
                ))

            # # Save the generated samples
            # if iteration % opts.sample_every == 0:
            #     save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

            # # Save the model parameters
            # if iteration % opts.checkpoint_every == 0:
            #     checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)

    # Test
    with torch.no_grad():
        total = len(test_loader)
        correct = 0
        for inputs, labels in test_loader:
            predicted = model(inputs)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    # dummy
    dummy_data = []
    for i in range(10):
        a = random.random()
        b = random.random()
        dummy_data.append([a,b, a+b])
    df = pd.DataFrame(data=np.array(dummy_data))
    
    run(df)
