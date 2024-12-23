import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from utilities3 import *  # Utility functions for data handling
from Adam import Adam     # Custom optimizer

# ====================================
# PR and SpectralConv Layers
# ====================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.modes1 = modes1
        self.weights1 = nn.Parameter(torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))

class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()
        self.weights_pole = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x):
        t = grid_x_train.cuda()
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0 = torch.fft.fftfreq(t.shape[0], dt) * 2 * np.pi * 1j
        lambda1 = lambda0.reshape(1, 1, -1, 1).cuda()
        weights_pole = self.weights_pole.unsqueeze(2)
        term = torch.div(1, lambda1 - weights_pole)
        Hw = self.weights_residue.unsqueeze(2) * term
        Hw = Hw.sum(dim=-1)
        output_residue = torch.einsum("bix,iox->box", alpha, Hw.permute(1, 0, 2))
        return torch.real(torch.fft.ifft(output_residue, n=x.size(-1)))

# ====================================
# FLNO Model
# ====================================
class FLNO1d(nn.Module):
    def __init__(self, modes, width, num_layers, laplace_first_only=False):
        super(FLNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers
        self.laplace_first_only = laplace_first_only

        self.fc0 = nn.Linear(2, self.width)
        self.pr_layers = nn.ModuleList()
        self.fconv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0 or not laplace_first_only:
                self.pr_layers.append(PR(self.width, self.width, self.modes1))
            else:
                self.pr_layers.append(None)
            self.fconv_layers.append(SpectralConv1d(self.width, self.width, self.modes1))
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        t = self.get_grid(x.shape, x.device)
        x = torch.cat((x, t), dim=-1)
        x = F.gelu(self.fc0(x)).permute(0, 2, 1)
        for i in range(self.num_layers):
            if self.pr_layers[i] is not None:
                x = F.gelu(self.pr_layers[i](x) + self.fconv_layers[i](x) + self.w_layers[i](x))
            else:
                x = F.gelu(self.fconv_layers[i](x) + self.w_layers[i](x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1)
        return gridx.repeat(batchsize, 1, 1)

# ====================================
# Training and Saving Results
# ====================================
modes, width, epochs = 16, 4, 1000
step_size, gamma, learning_rate = 100, 0.5, 0.002
batch_size_train, batch_size_vali = 20, 20

reader = MatReader('data.mat')
x_train = reader.read_field('f_train').reshape(-1, 2048, 1)
y_train = reader.read_field('u_train')
x_vali = reader.read_field('f_vali').reshape(-1, 2048, 1)
y_vali = reader.read_field('u_vali')
grid_x_train = reader.read_field('x_train')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)

# Storage for results
results = {"train_loss_laplace": [], "vali_loss_laplace": [], 
           "train_loss_no_laplace": [], "vali_loss_no_laplace": [], "num_layers": []}

# Run for 1 to 10 layers
for scenario in ["laplace", "no_laplace"]:
    laplace_first_only = True if scenario == "laplace" else False
    for num_layers in range(1, 11):
        print(f"Running {scenario} with {num_layers} layers...")
        model = FLNO1d(modes, width, num_layers, laplace_first_only).cuda()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        train_loss, vali_loss = [], []

        for ep in range(epochs):
            model.train()
            batch_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = F.mse_loss(model(x.cuda()).view(batch_size_train, -1), y.cuda().view(batch_size_train, -1))
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            train_loss.append(batch_loss / len(train_loader))

            model.eval()
            with torch.no_grad():
                val_loss = np.mean([F.mse_loss(model(x.cuda()).view(batch_size_vali, -1), y.cuda().view(batch_size_vali, -1)).item() 
                                    for x, y in vali_loader])
            vali_loss.append(val_loss)
            scheduler.step()

        if scenario == "laplace":
            results["train_loss_laplace"].append(train_loss)
            results["vali_loss_laplace"].append(vali_loss)
        else:
            results["train_loss_no_laplace"].append(train_loss)
            results["vali_loss_no_laplace"].append(vali_loss)
        results["num_layers"].append(num_layers)

# Save results to a single .mat file
output_file = "training_results_laplace_vs_no_laplace.mat"
sio.savemat(output_file, results)
print(f"Results saved to {output_file}")

