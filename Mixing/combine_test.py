#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:46:33 2024

@author: hossein
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import os
import time
from timeit import default_timer
from utilities3 import *
from Adam import Adam

# Import the CombinedModel
from CombinedModel import CombinedModel  # Replace 'untitled4' with your actual model filename
torch.cuda.empty_cache()

# ====================================
# Hyperparameters
# ====================================
ntrain = 200
nvali = 50
ntest = 130

batch_size_train = 10
batch_size_vali = 10
learning_rate = 0.002
epochs = 1000
step_size = 100
gamma = 0.5

# Model Parameters
fno_modes1, fno_modes2 = 4, 4  # Modes for FNO
lno_modes1, lno_modes2 = 5, 5    # Modes for LNO
layer_types = ['FNO', 'LNO', 'Combined']  # Choose the layer configuration
width = 16  # Number of channels

# ====================================
# Load Data
# ====================================
reader = MatReader('data_beam.mat')
x_train = reader.read_field('f_train')
y_train = reader.read_field('u_train')
x_vali = reader.read_field('f_vali')
y_vali = reader.read_field('u_vali')
x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')

# Reshape Data
x_train = x_train.reshape(ntrain, x_train.shape[1], x_train.shape[2], 1)
x_vali = x_vali.reshape(nvali, x_vali.shape[1], x_vali.shape[2], 1)
x_test = x_test.reshape(ntest, x_test.shape[1], x_test.shape[2], 1)

# Add Grid Information (Ensure 3 input channels for the model)
def add_grid(x):
    gridx = torch.linspace(0, 1, x.shape[1]).reshape(1, -1, 1, 1).repeat([x.shape[0], 1, x.shape[2], 1])
    gridy = torch.linspace(0, 1, x.shape[2]).reshape(1, 1, -1, 1).repeat([x.shape[0], x.shape[1], 1, 1])
    return torch.cat((x, gridx, gridy), dim=-1)

x_train = add_grid(x_train)
x_vali = add_grid(x_vali)
x_test = add_grid(x_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=False)

# ====================================
# Model Initialization
# ====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel(
    in_channels=3, 
    out_channels=width, 
    fno_modes1=fno_modes1, fno_modes2=fno_modes2, 
    lno_modes1=lno_modes1, lno_modes2=lno_modes2, 
    layer_types=layer_types
).to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=True)

# ====================================
# Training Loop
# ====================================
train_loss = np.zeros(epochs)
vali_loss = np.zeros(epochs)

for ep in range(epochs):
    model.train()
    train_l2 = 0.0
    t1 = default_timer()
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = myloss(out.view(batch_size_train, -1), y.view(batch_size_train, -1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()
    train_loss[ep] = train_l2 / len(train_loader)
    
    # Validation
    model.eval()
    vali_l2 = 0.0
    with torch.no_grad():
        for x, y in vali_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            vali_l2 += myloss(out.view(-1, x_vali.shape[1], x_vali.shape[2]), y).item()

    vali_loss[ep] = vali_l2 / len(vali_loader)
    t2 = default_timer()
    
    print(f"Epoch {ep+1}/{epochs}, Time: {t2-t1:.2f}s, Train Loss: {train_loss[ep]:.4f}, Val Loss: {vali_loss[ep]:.4f}")

# ====================================
# Save Results
# ====================================
torch.save(model, 'combined_model.pt')
print("Model saved successfully!")

# Plot Loss
import matplotlib.pyplot as plt
plt.plot(train_loss, label='Train Loss')
plt.plot(vali_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.title('Training and Validation Loss')
plt.show()
