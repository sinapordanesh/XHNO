import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timeit import default_timer
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
        batchsize, channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])
        out_ft = torch.zeros_like(x_ft, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.weights.shape[2], :self.weights.shape[3]] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.weights.shape[2], :self.weights.shape[3]], self.weights
        )
        return torch.fft.irfft2(out_ft, s=(height, width)).real

# ====================================
# PR2d Layer (Your Original Laplace Layer)
# ====================================
class PR2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(PR2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def output_PR(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
        Hw = torch.zeros(weights_residue.shape[0], weights_residue.shape[0], weights_residue.shape[2], weights_residue.shape[3],
                         lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1 = torch.div(1, torch.einsum("pbix,qbik->pqbixk", torch.sub(lambda1, weights_pole1), torch.sub(lambda2, weights_pole2)))
        Hw = torch.einsum("bixk,pqbixk->pqbixk", weights_residue, term1)
        Pk = Hw  # for 2D PDE, Pk = Hw
        output_residue1 = torch.einsum("biox,oxikpq->bkox", alpha, Hw)
        output_residue2 = torch.einsum("biox,oxikpq->bkpq", alpha, Pk)
        return output_residue1, output_residue2

    def forward(self, x):
        tx = T.cuda()
        ty = X.cuda()
        dty = (ty[0, 1] - ty[0, 0]).item()
        dtx = (tx[0, 1] - tx[0, 0]).item()
        alpha = torch.fft.fft2(x, dim=[-2, -1])
        omega1 = torch.fft.fftfreq(ty.shape[1], dty) * 2 * np.pi * 1j
        omega2 = torch.fft.fftfreq(tx.shape[1], dtx) * 2 * np.pi * 1j
        omega1 = omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
        omega2 = omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()

        output_residue1, output_residue2 = self.output_PR(omega1, omega2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)

        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1))).real
        term1 = torch.einsum("bip,kz->bipz", self.weights_pole1, ty.type(torch.complex64).reshape(1, -1))
        term2 = torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1, -1))
        term3 = torch.einsum("bipz,biqx->bipqzx", torch.exp(term1), torch.exp(term2))
        x2 = torch.einsum("kbpq,bipqzx->kizx", output_residue2, term3).real
        return x1 + x2 / x.size(-1) / x.size(-2)

# ====================================
# Combined FLNO2D Model
# ====================================
class FLNO2d(nn.Module):
    def __init__(self, width, modes1, modes2, num_layers):
        super(FLNO2d, self).__init__()
        self.fc0 = nn.Linear(3, width)

        self.fno_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)])
        self.pr_layers = nn.ModuleList([PR2d(width, width, modes1, modes2) for _ in range(num_layers)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(num_layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)

        for i in range(len(self.fno_layers)):
            x = F.gelu(self.fno_layers[i](x) + self.pr_layers[i](x) + self.w_layers[i](x))

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

# ====================================
# Training FLNO2D
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

modes1 = 4  
modes2 = 4   
width = 16

reader = MatReader('Data/data.mat')
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

model = FLNO2d(width, modes1, modes2, num_layers).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = LpLoss(size_average=True)

for ep in range(epochs):
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {ep}, Loss: {loss.item():.6f}")

