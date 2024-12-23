import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities3 import *
from Adam import Adam

# ====================================
# SpectralConv2d (FNO Layer)
# ====================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weights1 = nn.Parameter(torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

# ====================================
# PR2D (LNO Layer)
# ====================================
class PR2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(PR2d, self).__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weights_pole1 = nn.Parameter(torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        alpha = torch.fft.fft2(x, dim=[-2, -1])
        lambda1 = torch.fft.fftfreq(x.size(-2), 1.0).to(x.device).unsqueeze(-1)
        lambda2 = torch.fft.fftfreq(x.size(-1), 1.0).to(x.device).unsqueeze(-1)

        term1 = 1.0 / (lambda1 - self.weights_pole1.unsqueeze(-1))
        term2 = 1.0 / (lambda2 - self.weights_pole2.unsqueeze(-1))
        Hw = self.weights_residue * (term1 @ term2)

        x1 = torch.fft.ifft2(alpha * Hw, s=(x.size(-2), x.size(-1)))
        return torch.real(x1)

# ====================================
# Combined FLNO2D Model
# ====================================
class FLNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_fno_layers=0, num_lno_layers=0):
        super(FLNO2d, self).__init__()
        self.num_fno_layers = num_fno_layers
        self.num_lno_layers = num_lno_layers
        self.width = width

        # Input Lift
        self.fc0 = nn.Linear(3, width)

        # Dynamically define FNO and LNO layers
        self.fno_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(num_fno_layers)])
        self.lno_layers = nn.ModuleList([PR2d(width, width, modes1, modes2) for _ in range(num_lno_layers)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(num_fno_layers + num_lno_layers)])

        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)

        # Apply FNO and LNO layers
        for i in range(max(self.num_fno_layers, self.num_lno_layers)):
            if i < self.num_fno_layers:
                x = x + self.fno_layers[i](x)
            if i < self.num_lno_layers:
                x = x + self.lno_layers[i](x)
            x = F.gelu(x + self.w_layers[i](x))

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

# ====================================
# Training and Configuration
# ====================================
if __name__ == "__main__":
    ntrain, nvali, ntest = 200, 50, 130
    batch_size, learning_rate, epochs = 50, 0.002, 500
    modes1, modes2, width = 4, 4, 16

    reader = MatReader('data_beam.mat')
    x_train = reader.read_field('f_train')
    y_train = reader.read_field('u_train')
    T = reader.read_field('t')
    X = reader.read_field('x')
    x_vali = reader.read_field('f_vali')
    y_vali = reader.read_field('u_vali')

    x_test = reader.read_field('f_test')
    y_test = reader.read_field('u_test')

    x_train = x_train.reshape(ntrain,x_train.shape[1],x_train.shape[2],1)
    x_vali = x_vali.reshape(nvali,x_vali.shape[1],x_vali.shape[2],1)
    x_test = x_test.reshape(ntest,x_test.shape[1],x_test.shape[2],1)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size, shuffle=False)

    # Scenarios: Adjust num_fno_layers and num_lno_layers
    scenarios = [
        {"num_fno_layers": 2, "num_lno_layers": 0, "name": "FNO_Only"},
        {"num_fno_layers": 0, "num_lno_layers": 2, "name": "LNO_Only"},
        {"num_fno_layers": 2, "num_lno_layers": 2, "name": "FNO_LNO_Combined"}
    ]

    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        model = FLNO2d(modes1, modes2, width, scenario["num_fno_layers"], scenario["num_lno_layers"]).cuda()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_fn = LpLoss(size_average=True)

        for ep in range(epochs):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x.cuda()), y.cuda())
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = np.mean([loss_fn(model(x.cuda()), y.cuda()).item() for x, y in vali_loader])

            print(f"Epoch {ep + 1}, Validation Loss: {val_loss:.6e}")

