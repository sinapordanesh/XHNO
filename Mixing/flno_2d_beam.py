#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:09:03 2024

@author: hossein
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io
from timeit import default_timer
from utilities3 import *
from Adam import Adam

# ========================
# FNO Layer
# ========================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.weights = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize, channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights)
        return torch.fft.irfft2(out_ft, s=(height, width)).real


# ========================
# LNO Layer
# ========================
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

        residue = torch.einsum("iohwm, ioawm -> iohwa", term1, term2).squeeze(-1)
        residue = residue.unsqueeze(0)
        alpha_res = alpha.unsqueeze(1) * residue
        alpha_res = alpha_res.sum(dim=2)

        return torch.fft.ifft2(alpha_res, s=(height, width)).real


# ========================
# Combined Model
# ========================
class CombinedModel(nn.Module):
    def __init__(self, in_channels, out_channels, modes1_fno, modes2_fno, modes1_lno, modes2_lno, num_layers, model_type):
        super(CombinedModel, self).__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.fc0 = nn.Linear(1, out_channels)

        # Define layers
        self.fno_layers = nn.ModuleList([SpectralConv2d(out_channels, out_channels, modes1_fno, modes2_fno) for _ in range(num_layers)])
        self.lno_layers = nn.ModuleList([PR2d(out_channels, out_channels, modes1_lno, modes2_lno) for _ in range(num_layers)])
        self.linear_layers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1) for _ in range(num_layers)])
        
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        for i in range(self.num_layers):
            if self.model_type == 'FNO':
                x = x + self.fno_layers[i](x)
            elif self.model_type == 'LNO':
                x = x + self.lno_layers[i](x)
            elif self.model_type == 'FLNO':
                x = x + self.fno_layers[i](x) + self.lno_layers[i](x)
            x = x + self.linear_layers[i](x)
            x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ========================
# Data Preparation
# ========================
ntrain = 200
nvali = 50
ntest = 130
batch_size = 10
modes1_fno, modes2_fno = 12, 12
modes1_lno, modes2_lno = 4, 4
width = 16

reader = MatReader('data_beam.mat')
x_train = reader.read_field('f_train').reshape(ntrain, -1, -1, 1)
y_train = reader.read_field('u_train')
x_val = reader.read_field('f_vali').reshape(nvali, -1, -1, 1)
y_val = reader.read_field('u_vali')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

# ========================
# Train and Save Results
# ========================
results = {}
def train_and_evaluate(model, train_loader, vali_loader, epochs=10, learning_rate=0.001, key_prefix=""):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    myloss = LpLoss(size_average=True)
    best_val_loss = float('inf')

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()
            optimizer.step()

        model.eval()
        vali_l2 = 0
        with torch.no_grad():
            for x, y in vali_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                vali_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        vali_l2 /= len(vali_loader)

        if vali_l2 < best_val_loss:
            best_val_loss = vali_l2
            best_predictions = out.cpu().numpy()

    results[f"pred_{key_prefix}"] = best_predictions
    results[f"loss_{key_prefix}"] = best_val_loss

# Run for each model
for model_type in ['FNO', 'LNO', 'FLNO']:
    print(f"\nTraining {model_type} Model")
    model = CombinedModel(1, width, modes1_fno, modes2_fno, modes1_lno, modes2_lno, num_layers=5, model_type=model_type).cuda()
    train_and_evaluate(model, train_loader, vali_loader, epochs=10, learning_rate=0.001, key_prefix=model_type)

# Save all results into one .mat file
scipy.io.savemat("combined_results.mat", results)
print("All results saved to combined_results.mat")
