import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from utilities3 import MatReader
from Adam import Adam

# ====================================
# SpectralConv1d Layer: Fourier Transform
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

# ====================================
# PR1d Layer: Laplace Transform
# ====================================
class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()
        self.weights_pole = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))

    def forward(self, x):
        t = grid_x_train.cuda()  # Positional grid for time
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x)  # FFT of the input, shape: (batch_size, in_channels, x_size)

        # Generate frequency tensor and align dimensions
        lambda0 = torch.fft.fftfreq(t.shape[0], dt) * 2 * np.pi * 1j  # Shape: (x_size,)
        lambda1 = lambda0.reshape(1, 1, -1, 1).cuda()  # Reshape for broadcasting

        weights_pole = self.weights_pole.unsqueeze(2)  # Shape: (in_channels, out_channels, 1, modes1)
        term = torch.div(1, lambda1 - weights_pole)  # Broadcasting happens here

        Hw = self.weights_residue.unsqueeze(2) * term  # Shape: (in_channels, out_channels, x_size, modes1)
        Hw = Hw.sum(dim=-1)  # Collapse modes1 -> Shape: (in_channels, out_channels, x_size)

        output_residue = torch.einsum("bix,iox->box", alpha, Hw.permute(1, 0, 2))  # Shape: (batch_size, out_channels, x_size)
        return torch.real(torch.fft.ifft(output_residue, n=x.size(-1)))



# ====================================
# FLNO1d: Flexible Model
# ====================================
class FLNO1d(nn.Module):
    def __init__(self, modes, width, num_layers, use_fourier=True, use_laplace=True):
        super(FLNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.num_layers = num_layers
        self.use_fourier = use_fourier
        self.use_laplace = use_laplace

        self.fc0 = nn.Linear(2, self.width)
        self.fconv_layers = nn.ModuleList()
        self.pr_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.fconv_layers.append(SpectralConv1d(self.width, self.width, self.modes1) if use_fourier else None)
            self.pr_layers.append(PR1d(self.width, self.width, self.modes1) if use_laplace else None)
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        t = self.get_grid(x.shape, x.device)
        x = torch.cat((x, t), dim=-1)
        x = F.gelu(self.fc0(x)).permute(0, 2, 1)

        for i in range(self.num_layers):
            x_res = 0
            if self.use_fourier:
                x_res += self.fconv_layers[i](x)
            if self.use_laplace:
                x_res += self.pr_layers[i](x)
            x_res += self.w_layers[i](x)
            x = F.gelu(x_res)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1)
        return gridx.repeat(batchsize, 1, 1)

# ====================================
# Training Function
# ====================================
def train_FLNO1d_scenarios():
    # Hyperparameters
    ntrain, nvali = 200, 50
    batch_size_train, batch_size_vali = 50, 50
    learning_rate, epochs = 0.002, 250
    modes, width = 16, 4

    # Load Data
    reader = MatReader('data.mat')
    x_train = reader.read_field('f_train').reshape(ntrain, 2048, 1)
    y_train = reader.read_field('u_train')
    x_vali = reader.read_field('f_vali').reshape(nvali, 2048, 1)
    y_vali = reader.read_field('u_vali')

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=False)

    results = {"train_loss": [], "vali_loss": [], "config": []}

    # Run scenarios
    for use_fourier, use_laplace, label in [(True, False, "FourierOnly"),
                                            (False, True, "LaplaceOnly"),
                                            (True, True, "Combined")]:
        for num_layers in range(1, 6):
            print(f"\nRunning {label} with {num_layers} layers...")
            model = FLNO1d(modes, width, num_layers, use_fourier, use_laplace).cuda()
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            train_losses, vali_losses = [], []
            for epoch in range(epochs):
                model.train()
                batch_loss = 0
                for x, y in train_loader:
                    optimizer.zero_grad()
                    loss = loss_fn(model(x.cuda()).view(batch_size_train, -1), y.cuda().view(batch_size_train, -1))
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                train_losses.append(batch_loss / len(train_loader))

                model.eval()
                vali_loss = 0
                with torch.no_grad():
                    for x, y in vali_loader:
                        loss = loss_fn(model(x.cuda()).view(batch_size_vali, -1), y.cuda().view(batch_size_vali, -1))
                        vali_loss += loss.item()
                vali_losses.append(vali_loss / len(vali_loader))

                print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {vali_losses[-1]:.6f}")

            results["train_loss"].append(train_losses)
            results["vali_loss"].append(vali_losses)
            results["config"].append(f"{label}_{num_layers}Layers")

    sio.savemat("FLNO1d_results.mat", results)
    print("\nAll results saved to FLNO1d_results.mat")

# ====================================
# Main Function
# ====================================
if __name__ == "__main__":
    train_FLNO1d_scenarios()

