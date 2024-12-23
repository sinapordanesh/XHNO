#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:01:14 2024

@author: hossein
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io
from timeit import default_timer
from utilities3 import MatReader, LpLoss
from Adam import Adam

# Spectral FNO Layer
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.weights = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))).real

# Laplace LNO Layer
class PR2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(PR2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.weights_pole1 = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        alpha = torch.fft.fft2(x, dim=[-2, -1])
        height, width = x.shape[-2], x.shape[-1]
        device = x.device
        lambda1 = torch.fft.fftfreq(height, d=1, device=device).reshape(1, 1, height, 1, 1) * 2 * np.pi * 1j
        lambda2 = torch.fft.fftfreq(width, d=1, device=device).reshape(1, 1, 1, width, 1) * 2 * np.pi * 1j

        weights_pole1 = self.weights_pole1.unsqueeze(2).unsqueeze(3)
        weights_pole2 = self.weights_pole2.unsqueeze(2).unsqueeze(3)

        term1 = 1.0 / (lambda1 - weights_pole1)
        term2 = 1.0 / (lambda2 - weights_pole2)

        residue = torch.einsum("iohwm, ioawm -> iohwa", term1, term2).squeeze(-1).unsqueeze(0)
        alpha_res = alpha.unsqueeze(1) * residue
        alpha_res = alpha_res.sum(dim=2)

        return torch.fft.ifft2(alpha_res, s=(height, width)).real

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, in_channels, out_channels, modes1_fno, modes2_fno, modes1_lno, modes2_lno, num_layers, use_fno, use_lno):
        super(CombinedModel, self).__init__()
        self.fc0 = nn.Linear(5, out_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_fno:
                self.layers.append(SpectralConv2d(out_channels, out_channels, modes1_fno, modes2_fno))
            if use_lno:
                self.layers.append(PR2d(out_channels, out_channels, modes1_lno, modes2_lno))
            self.layers.append(nn.Conv2d(out_channels, out_channels, 1))
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = torch.relu(layer(x) + x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

# Training Function
def train_model(model, epochs, train_loader, vali_loader, device):
    optimizer = Adam(model.parameters(), lr=0.002)
    myloss = LpLoss(size_average=True)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = myloss(out.view(y.shape), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Load Data
reader = MatReader('data_beam.mat')
x_train = reader.read_field('f_train').reshape(-1, 64, 64, 1)
y_train = reader.read_field('u_train')
x_val = reader.read_field('f_vali').reshape(-1, 64, 64, 1)
y_val = reader.read_field('u_vali')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=8, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=8, shuffle=False)

# Run Experiments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_channels = 16
modes1_fno, modes2_fno = 12, 12
modes1_lno, modes2_lno = 4, 4  # Different modes for LNO

results = {}
for num_layers in range(1, 6):
    for config, use_fno, use_lno in [("FNO", True, False), ("LNO", False, True), ("FLNO", True, True)]:
        print(f"\nTraining {config} with {num_layers} layers")
        model = CombinedModel(1, out_channels, modes1_fno, modes2_fno, modes1_lno, modes2_lno, num_layers, use_fno, use_lno).to(device)
        train_model(model, epochs=10, train_loader=train_loader, vali_loader=vali_loader, device=device)
        results[f"{config}_{num_layers}"] = model

# Save Results
scipy.io.savemat('results.mat', {'results': results})
print("Results saved to 'results.mat'.")
