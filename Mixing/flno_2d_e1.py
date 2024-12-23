import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities3 import *
from Adam import Adam

# ====================================
# SpectralConv2d Layer (FNO Layer)
# ====================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.weights = nn.Parameter(torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])
        out_ft = torch.zeros_like(x_ft, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.weights.shape[2], :self.weights.shape[3]] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.weights.shape[2], :self.weights.shape[3]], self.weights
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))).real

# ====================================
# PR2d Layer (LNO Layer)
# ====================================
class PR2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Laplace Neural Operator (LNO) Layer.
        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            modes1 (int): Number of modes in the first dimension.
            modes2 (int): Number of modes in the second dimension.
        """
        super(PR2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))

        # Initialize weights
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compute_frequencies(self, x):
        """
        Compute the FFT frequencies dynamically for the input tensor `x`.
        Args:
            x (Tensor): Input tensor [batch, channels, size_x, size_y].
        Returns:
            lambda1 (Tensor): FFT frequencies in the first dimension.
            lambda2 (Tensor): FFT frequencies in the second dimension.
        """
        size_x, size_y = x.shape[-2], x.shape[-1]
        lambda1 = torch.fft.fftfreq(size_x, 1.0).to(x.device) * 2 * np.pi * 1j
        lambda2 = torch.fft.fftfreq(size_y, 1.0).to(x.device) * 2 * np.pi * 1j
        return lambda1.unsqueeze(0), lambda2.unsqueeze(0)

    def forward(self, x):
    # Compute dynamic FFT frequencies
        size_x, size_y = x.shape[-2], x.shape[-1]
        alpha = torch.fft.fft2(x, dim=[-2, -1])
        lambda1 = torch.fft.fftfreq(size_x, 1.0).to(x.device) * 2 * np.pi * 1j
        lambda2 = torch.fft.fftfreq(size_y, 1.0).to(x.device) * 2 * np.pi * 1j

    # Align shapes for broadcasting
        lambda1 = lambda1.reshape(1, 1, -1, 1)  # [1, 1, size_x, 1]
        lambda2 = lambda2.reshape(1, 1, 1, -1)  # [1, 1, 1, size_y]

    # Compute pole-residue operation
        term1 = 1.0 / (lambda1 - self.weights_pole1.unsqueeze(-1))  # Shape: [in, out, size_x, 1]
        term2 = 1.0 / (lambda2 - self.weights_pole2.unsqueeze(-1))  # Shape: [in, out, 1, size_y]

    # Perform outer product and multiply with weights_residue
        Hw = self.weights_residue.unsqueeze(-1).unsqueeze(-1) * term1 * term2  # Shape: [in, out, modes1, modes2, size_x, size_y]

    # Combine FFT coefficients and return to spatial domain
        output_fft = torch.einsum("biox,ioxpq->bopq", alpha, Hw)
        output = torch.fft.ifft2(output_fft, s=(size_x, size_y)).real
        return output




# ====================================
# Combined FLNO2d Model
# ====================================
class FLNO2d(nn.Module):
    def __init__(self, width, num_layers, modes1_fno, modes2_fno, modes1_lno, modes2_lno):
        super(FLNO2d, self).__init__()
        self.fc0 = nn.Linear(3, width)

        self.fno_layers = nn.ModuleList([SpectralConv2d(width, width, modes1_fno, modes2_fno) for _ in range(num_layers)])
        self.lno_layers = nn.ModuleList([PR2d(width, width, modes1_lno, modes2_lno) for _ in range(num_layers)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(num_layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)

        for i in range(len(self.fno_layers)):
            fno_output = self.fno_layers[i](x)
            lno_output = self.lno_layers[i](x)
            w_output = self.w_layers[i](x)
            x = F.gelu(fno_output + lno_output + w_output)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

# ====================================
# Training FLNO2d
# ====================================
ntrain = 200
nvali = 50
ntest=130

batch_size_train = 50
batch_size_vali = 50

learning_rate = 0.002

epochs = 1000
step_size = 100
gamma = 0.5
num_layers = 1

modes1_fno, modes2_fno = 12, 12
modes1_lno, modes2_lno = 4, 4
width = 16
num_layers = 4

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

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)
# model



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=False)

# Model Initialization
model = FLNO2d(width, num_layers, modes1_fno, modes2_fno, modes1_lno, modes2_lno).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(size_average=True)

# Training Loop
for ep in range(epochs):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x.cuda()), y.cuda())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    print(f"Epoch {ep}, Training Loss: {train_loss / len(train_loader):.6f}")

