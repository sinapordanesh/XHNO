#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:42:21 2024

@author: hossein
"""

import torch
import torch.nn as nn
import numpy as np
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

# Spectral FNO Layer
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

        residue = torch.einsum("iohwm, ioawm -> iohwa", term1, term2).squeeze(-1)
        residue = residue.unsqueeze(0)
        alpha_res = alpha.unsqueeze(1) * residue
        alpha_res = alpha_res.sum(dim=2)

        return torch.fft.ifft2(alpha_res, s=(height, width)).real

# Combined Model with Configurable Layers
class CombinedModel(nn.Module):
    def __init__(self, in_channels, out_channels, fno_modes1, fno_modes2, lno_modes1, lno_modes2, layer_types):
        """
        layer_types: List of strings, each representing the type of layer:
                     'FNO'      -> FNO-only layer
                     'LNO'      -> LNO-only layer
                     'Combined' -> Combined FNO + LNO + Linear
        """
        super(CombinedModel, self).__init__()
        self.layer_types = layer_types

        # Input transformation
        self.fc0 = nn.Linear(5, out_channels)

        # Define layers based on configuration
        self.layers = nn.ModuleList()
        for layer_type in layer_types:
            if layer_type == 'FNO':
                self.layers.append(SpectralConv2d(out_channels, out_channels, fno_modes1, fno_modes2))
            elif layer_type == 'LNO':
                self.layers.append(PR2d(out_channels, out_channels, lno_modes1, lno_modes2))
            elif layer_type == 'Combined':
                self.layers.append(nn.ModuleDict({
                    "fno": SpectralConv2d(out_channels, out_channels, fno_modes1, fno_modes2),
                    "lno": PR2d(out_channels, out_channels, lno_modes1, lno_modes2),
                    "linear": nn.Conv2d(out_channels, out_channels, 1)
                }))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # Output transformation
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Prepare the input grid
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)  # Shape: [batch, channels, height, width]

        # Process each layer based on its type
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == 'FNO':
                x = x + layer(x)  # FNO-only
            elif self.layer_types[i] == 'LNO':
                x = x + layer(x)  # LNO-only
            elif self.layer_types[i] == 'Combined':
                fno_out = layer['fno'](x)
                lno_out = layer['lno'](x)
                linear_out = layer['linear'](x)
                x = x + fno_out + lno_out + linear_out  # Combined output
            x = torch.relu(x)  # Activation

        x = x.permute(0, 2, 3, 1)  # Shape: [batch, height, width, channels]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

# Test the Model
batch_size = 2
in_channels = 3
out_channels = 16
fno_modes1, fno_modes2 = 12, 12  # Modes for FNO layers
lno_modes1, lno_modes2 = 8, 8    # Modes for LNO layers
layer_types = ['FNO', 'LNO', 'Combined']  # Example: one FNO layer, one LNO layer, one Combined layer

x = torch.randn(batch_size, 64, 64, in_channels).cuda()
model = CombinedModel(in_channels, out_channels, fno_modes1, fno_modes2, lno_modes1, lno_modes2, layer_types).cuda()

output = model(x)
print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
