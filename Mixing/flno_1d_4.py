import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
from utilities3 import *
from Adam import Adam


# ====================================
# Directory and Saving Setup
# ====================================
save_index = 1
current_directory = os.getcwd()
case = "Case_FLNO_4Layers"
folder_index = str(save_index)

results_dir = "/" + case + folder_index + "/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)


# ====================================
# PR and SpectralConv Layers
# ====================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        t = grid_x_train.cuda()
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0 = torch.fft.fftfreq(t.shape[0], dt) * 2 * np.pi * 1j
        lambda1 = lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()

        term1 = torch.div(1, torch.sub(lambda1, self.weights_pole))
        Hw = self.weights_residue * term1
        output_residue1 = torch.einsum("bix,xiok->box", alpha, Hw)
        output_residue2 = torch.einsum("bix,xiok->bok", alpha, -Hw)

        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)

        x2 = torch.zeros(output_residue2.shape[0], output_residue2.shape[1], t.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1 = torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1, -1))
        term2 = torch.exp(term1)
        x2 = torch.einsum("bix,ioxz->boz", output_residue2, term2)
        x2 = torch.real(x2)
        x2 = x2 / x.size(-1)
        return x1 + x2


# ====================================
# FLNO Model with 4 Layers
# ====================================
class FLNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FLNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(1, self.width)

        # Layers
        self.pr0 = PR(self.width, self.width, self.modes1)
        self.pr1 = PR(self.width, self.width, self.modes1)
        self.pr2 = PR(self.width, self.width, self.modes1)
        self.pr3 = PR(self.width, self.width, self.modes1)

        self.fconv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.fconv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.fconv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.fconv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        t = self.get_grid(x.shape, x.device)
        x = torch.cat((x, t), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Layer 1
        x = F.gelu(self.pr0(x) + self.fconv0(x) + self.w0(x))

        # Layer 2
        x = F.gelu(self.pr1(x) + self.fconv1(x) + self.w1(x))

        # Layer 3
        x = F.gelu(self.pr2(x) + self.fconv2(x) + self.w2(x))

        # Layer 4
        x = self.pr3(x) + self.fconv3(x) + self.w3(x)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# ====================================
# Training Setup
# ====================================
modes = 16
width = 4
epochs = 1000
step_size = 100
gamma = 0.5
learning_rate = 0.002
batch_size_train = 20
batch_size_vali = 20

reader = MatReader('data.mat')
x_train = reader.read_field('f_train').reshape(-1, 2048, 1)
y_train = reader.read_field('u_train')
x_vali = reader.read_field('f_vali').reshape(-1, 2048, 1)
y_vali = reader.read_field('u_vali')
grid_x_train = reader.read_field('x_train')
grid_x_vali = reader.read_field('x_vali')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)

# Model
model = FLNO1d(modes, width).cuda()

# Optimizer and Loss
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=True)

# Training
train_loss = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
        mse.backward()
        optimizer.step()
        train_mse += mse.item()

    train_loss[ep] = train_mse / len(train_loader)

    # Validation
    model.eval()
    vali_mse = 0
    with torch.no_grad():
        for x, y in vali_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            mse = F.mse_loss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1), reduction='mean')
            vali_mse += mse.item()
    vali_loss[ep] = vali_mse / len(vali_loader)

    scheduler.step()

    print(f"Epoch {ep + 1}/{epochs}, Train Loss: {train_loss[ep, 0]:.6f}, Val Loss: {vali_loss[ep, 0]:.6f}")

# Save Results
np.save(save_results_to + 'train_loss.npy', train_loss)
np.save(save_results_to + 'vali_loss.npy', vali_loss)

# Plot Loss
plt.plot(range(epochs), train_loss, label='Train Loss')
plt.plot(range(epochs), vali_loss, label='Validation Loss')
plt.legend()
plt.yscale('log')
plt.savefig(save_results_to + 'loss_history.png')

