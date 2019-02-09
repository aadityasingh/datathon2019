import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import GasTempNet
from data import get_data

gases = ['TS04', 'TNH4', 'CA', 'MG', 'NA', 'K', 'CL', 'NSO4', 'NHNO3', 'WSO2', 'WNO3']

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

train_loader, test_loader = get_data() #TODO params

for epoch in range(num_epochs):
    for i, (images, labels)  in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backprop
        optimzer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, total_step, loss.item()
            ))

# Test
with torch.no_grad():
    total = len(test_loader)
    correct = 0
    for inputs, labels in test_loader:
        predicted = model(inputs)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))
