import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from utilities3 import *  # Utility functions for data handling
from Adam import Adam     # Custom optimizer

# ====================================
# Directory and Saving Setup
# ====================================
save_index = 1
num_layers = 4  # Dynamically set the number of layers
current_directory = os.getcwd()
case = f"Case_FLNO_{num_layers}Layers"
results_dir = f"/{case}{save_index}/"
save_results_to = current_directory + results_dir

if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.weights_pole = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x):
        t = grid_x_train.cuda()  # Positional grid for time
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x)  # FFT of the input, shape: (batch_size, in_channels, x_size)

        # Generate frequency tensor and align dimensions
        lambda0 = torch.fft.fftfreq(t.shape[0], dt) * 2 * np.pi * 1j  # Shape: (x_size,)
        lambda1 = lambda0.reshape(1, 1, -1, 1, 1).cuda()  # Shape: (1, 1, x_size, 1, 1)

        # Expand weights_pole for broadcasting
        weights_pole = self.weights_pole.unsqueeze(0).unsqueeze(3)  # Shape: (1, in_channels, out_channels, 1, modes1)

        # Subtraction and term calculation
        term = torch.div(1, lambda1 - weights_pole)  # Shape: (1, in_channels, out_channels, x_size, modes1)

        # Compute Hw
        Hw = self.weights_residue.unsqueeze(0).unsqueeze(3) * term  # Shape: (1, in_channels, out_channels, x_size, modes1)
        Hw = Hw.sum(dim=-1)  # Sum over modes1 -> Shape: (1, in_channels, out_channels, x_size)

        # Combine with alpha
        output_residue = torch.einsum("bix,ioxk->box", alpha, Hw.squeeze(0).permute(1, 0, 2))  # Shape: (batch_size, out_channels, x_size)

        # Return to the time domain
        return torch.real(torch.fft.ifft(output_residue, n=x.size(-1)))








# ====================================
# FLNO Model with Dynamic Layers
# ====================================
class FLNO1d(nn.Module):
    def __init__(self, modes, width, num_layers):
        super(FLNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers

        self.fc0 = nn.Linear(2, self.width)  # Input includes positional encoding

        # Use ModuleList to dynamically create layers
        self.pr_layers = nn.ModuleList([PR(self.width, self.width, self.modes1) for _ in range(num_layers)])
        self.fconv_layers = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes1) for _ in range(num_layers)])
        self.w_layers = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(num_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Add positional encoding
        t = self.get_grid(x.shape, x.device)
        x = torch.cat((x, t), dim=-1)
        x = F.gelu(self.fc0(x)).permute(0, 2, 1)

        # Loop through the dynamically created layers
        for i in range(self.num_layers):
            x = F.gelu(self.pr_layers[i](x) + self.fconv_layers[i](x) + self.w_layers[i](x))

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1)
        return gridx.repeat(batchsize, 1, 1)


# ====================================
# Training Setup
# ====================================
modes, width, epochs = 16, 4, 500
step_size, gamma, learning_rate = 100, 0.5, 0.002
batch_size_train, batch_size_vali = 20, 20

# Data Loading
reader = MatReader('data.mat')
x_train = reader.read_field('f_train').reshape(-1, 2048, 1)
y_train = reader.read_field('u_train')
x_vali = reader.read_field('f_vali').reshape(-1, 2048, 1)
y_vali = reader.read_field('u_vali')
grid_x_train = reader.read_field('x_train')

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)

# Model Initialization
model = FLNO1d(modes, width, num_layers).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_loss, vali_loss = np.zeros((epochs, 1)), np.zeros((epochs, 1))

# Training Loop
for ep in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(x.cuda()).view(batch_size_train, -1), y.cuda().view(batch_size_train, -1))
        loss.backward()
        optimizer.step()
    train_loss[ep] = loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        vali_loss[ep] = np.mean([F.mse_loss(model(x.cuda()).view(batch_size_vali, -1), y.cuda().view(batch_size_vali, -1)).item()
                                 for x, y in vali_loader])

    scheduler.step()
    print(f"Epoch {ep + 1}, Train Loss: {train_loss[ep, 0]:.6f}, Val Loss: {vali_loss[ep, 0]:.6f}")

# Save Results
np.save(save_results_to + f'train_loss_{num_layers}Layer.npy', train_loss)
np.save(save_results_to + f'vali_loss_{num_layers}Layer.npy', vali_loss)

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(vali_loss, label='Validation Loss')
plt.yscale('log')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss ({num_layers} Layers)')
plt.savefig(save_results_to + f'loss_history_{num_layers}Layer.png')
plt.show()

