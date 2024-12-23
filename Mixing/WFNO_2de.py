#!/usr/bin/env python3from timeit import default_timer
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:07:27 2024

@author: hossein
"""

"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 2-D Darcy equation (time-independent problem).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from scipy.io import savemat
from timeit import default_timer
from utilities3 import *
from wavelet_convolution import WaveConv2d
from Adam import Adam
torch.manual_seed(0)
np.random.seed(0)

# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=10):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    






class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# Define WFNO2d combining WNO2d and FNO2d
import torch
import torch.nn as nn
import torch.nn.functional as F

class WFNO2d(nn.Module):
    def __init__(self, width, level, size, wavelet, in_channel, grid_range, modes1, modes2, n_wavelet_layers, n_fourier_layers, n_mixed_layers, padding=1):
        super(WFNO2d, self).__init__()

        self.width = width
        self.level = level
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_wavelet_layers = n_wavelet_layers
        self.n_fourier_layers = n_fourier_layers
        self.n_mixed_layers = n_mixed_layers

        # Input layer
        self.fc0 = nn.Linear(3, self.width)

        # Fourier layers and skip connections
        self.fourier_layers = nn.ModuleList()
        self.fourier_skip_layers = nn.ModuleList()
        for _ in range(n_fourier_layers):
            self.fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.fourier_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

        # Mixed layers and skip connections
        self.mixed_layers = nn.ModuleList()
        self.mixed_skip_layers = nn.ModuleList()
        for _ in range(n_mixed_layers):
            self.mixed_layers.append(nn.ModuleList([
                SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
                WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet)
            ]))
            self.mixed_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

        # Wavelet layers and skip connections
        self.wavelet_layers = nn.ModuleList()
        self.wavelet_skip_layers = nn.ModuleList()
        for _ in range(n_wavelet_layers):
            self.wavelet_layers.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet))
            self.wavelet_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

        self.norm = nn.InstanceNorm2d(self.width)

        # Output layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Add grid information
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # Change to (batch, channels, x, y)

        # Apply padding
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Fourier layers
        for fourier_layer, skip_layer in zip(self.fourier_layers, self.fourier_skip_layers):
            x1 = fourier_layer(x)
            x2 = skip_layer(x)
            x = F.gelu(x1 + x2)

        # Mixed layers
        for mixed_layer, skip_layer in zip(self.mixed_layers, self.mixed_skip_layers):
            x1 = self.norm(mixed_layer[0](self.norm(x)))  # Fourier branch
            x2 = self.norm(mixed_layer[1](self.norm(x)))  # Wavelet branch
            x3 = skip_layer(x)
            x = F.gelu(x1 + x2 + x3)

        # Wavelet layers
        for wavelet_layer, skip_layer in zip(self.wavelet_layers, self.wavelet_skip_layers):
            x1 = self.norm(wavelet_layer(self.norm(x)))
            x2 = skip_layer(x)
            x = F.gelu(x1 + x2)

        # Remove padding
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        # Final processing
        x = x.permute(0, 2, 3, 1)  # Change back to (batch, x, y, channels)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        """Generate grid information."""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    


        # Initialize layer containers
        #self.conv_wno = nn.ModuleList()
        #self.w_wno = nn.ModuleList()

        #self.conv_fno = nn.ModuleList()
        #self.w_fno = nn.ModuleList()

        #self.conv_wf = nn.ModuleList()
        #self.w_wf = nn.ModuleList()

        # Define WNO layers
        #for _ in range(self.num_w):
        #    self.conv_wno.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet))
        #    self.w_wno.append(nn.Conv2d(self.width, self.width, 1))

        # Define FNO layers
        #for _ in range(self.num_f):
        #    self.conv_fno.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
        #    self.w_fno.append(nn.Conv2d(self.width, self.width, 1))

        # Define mixed layers (Wavelet + Fourier)
        #for _ in range(self.num_wf):
        #    self.conv_wf.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet))
        #    self.conv_wf.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
        #    self.w_wf.append(nn.Conv2d(self.width, self.width, 1))

        # Define input and output layers
        #self.fc0 = nn.Linear(self.in_channel, self.width)
        #self.fc1 = nn.Linear(self.width, 128)
        #self.fc2 = nn.Linear(128, 1)

    #def forward(self, x):
    #    grid = self.get_grid(x.shape, x.device)
    #    x = torch.cat((x, grid), dim=-1)
    #    x = self.fc0(x)
    #    x = x.permute(0, 3, 1, 2)

    #    if self.padding != 0:
    #        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Apply WNO layers
    #    for conv, w in zip(self.conv_wno, self.w_wno):
    #        x = conv(x) + w(x)
    #        x = F.gelu(x)

        # Apply FNO layers
    #    for conv, w in zip(self.conv_fno, self.w_fno):
    #        x = conv(x) + w(x)
    #        x = F.gelu(x)

        # Apply mixed layers (Wavelet + Fourier)
    #    for i in range(0, len(self.conv_wf), 2):
    #        x1 = self.conv_wf[i](x)  # Wavelet
    #        x2 = self.conv_wf[i + 1](x)  # Fourier
    #        x3 = self.w_wf[i // 2](x)
    #        x = x1 + x2 + x3
    #        x = F.gelu(x)

    #    if self.padding != 0:
    #        x = x[..., :-self.padding, :-self.padding]

    #    x = x.permute(0, 2, 3, 1)
    #    x = F.gelu(self.fc1(x))
    #    x = self.fc2(x)
    #    return x

    #def get_grid(self, shape, device):
    #    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #    gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
    #    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #    gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
    #    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #    return torch.cat((gridx, gridy), dim=-1).to(device)

# %%
""" Model configurations """

PATH_Train = 'piececonst_r421_N1024_smooth1.mat'
PATH_Test = 'piececonst_r421_N1024_smooth2.mat'
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 100
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 4        # lavel of wavelet decomposition
width = 64       # uplifting dimension
layers = 4       # no of wavelet layers
modes1 = 16 
modes2 = 16 
sub = 5
h = int(((421 - 1)/sub) + 1) # total grid size divided by the subsampling rate
grid_range = [1, 1]          # The grid boundary in x and y direction
in_channel = 3   # (a(x, y), x, y) for this case

# %%
""" Read data """
reader = MatReader(PATH_Train)
x_train = reader.read_field('coeff')[:ntrain,::sub,::sub][:,:h,:h]
y_train = reader.read_field('sol')[:ntrain,::sub,::sub][:,:h,:h]

reader.load_file(PATH_Test)
x_test = reader.read_field('coeff')[:ntest,::sub,::sub][:,:h,:h]
y_test = reader.read_field('sol')[:ntest,::sub,::sub][:,:h,:h]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,h,h,1)
x_test = x_test.reshape(ntest,h,h,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)





# Fetch a single batch from x_train
x_batch, _ = next(iter(train_loader))  # Fetch one batch from the DataLoader

# Move the batch to the same device as the model

x_batch = x_batch.cuda()
model1 = WNO2d(width=width, level=level, layers=layers, size=[h,h], wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range, padding=1).to(device)
print(count_params(model1))

output1 = model1(x_batch)
print(output1.shape)

model1 = FNO2d(modes1, modes2, width).cuda()
model2 = FNO2d(modes1, modes2, width).cuda()

#output2 = model2(x_batch)
#print(output2.shape)




model2 = WFNO2d(width=32, level=3, size=[h, h], wavelet='db1', in_channel=3, grid_range=grid_range, modes1=16,
    modes2=16,
    n_wavelet_layers=1, n_fourier_layers=1, n_mixed_layers=1,
    padding=0
).to(device)
#output3 = model3(x_batch)
#print(output3.shape)




# # Define optimizers, schedulers, and losses for both models
optimizer1 = Adam(model1.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)

optimizer2 = Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

train_loss1 = torch.zeros(epochs)
test_loss1 = torch.zeros(epochs)
train_loss2 = torch.zeros(epochs)
test_loss2 = torch.zeros(epochs)

myloss = LpLoss(size_average=False)
y_normalizer.to(device)

for ep in range(epochs):
#     # Training for model1
    model1.train()
    t1 = default_timer()
    train_mse1 = 0
    train_l21 = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer1.zero_grad()
        out1 = model1(x).reshape(batch_size, h, h)
        out1 = y_normalizer.decode(out1)
        y = y_normalizer.decode(y)

        mse1 = F.mse_loss(out1.view(batch_size, -1), y.view(batch_size, -1))
        loss1 = myloss(out1.view(batch_size, -1), y.view(batch_size, -1))
        loss1.backward()
        optimizer1.step()

        train_mse1 += mse1.item()
        train_l21 += loss1.item()

    scheduler1.step()

#     # Evaluate model1
    model1.eval()
    test_l21 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out1 = model1(x).reshape(batch_size, h, h)
            out1 = y_normalizer.decode(out1)

            test_l21 += myloss(out1.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse1 /= len(train_loader)
    train_l21 /= ntrain
    test_l21 /= ntest

    train_loss1[ep] = train_l21
    test_loss1[ep] = test_l21

#     # Training for model2
    model2.train()
    train_mse2 = 0
    train_l22 = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer2.zero_grad()
        out2 = model2(x).reshape(batch_size, h, h)
        out2 = y_normalizer.decode(out2)
        y = y_normalizer.decode(y)

        mse2 = F.mse_loss(out2.view(batch_size, -1), y.view(batch_size, -1))
        loss2 = myloss(out2.view(batch_size, -1), y.view(batch_size, -1))
        loss2.backward()
#         # Ensure output matches the target shape
        optimizer2.step()

        train_mse2 += mse2.item()
        train_l22 += loss2.item()

    scheduler2.step()

#     # Evaluate model2
    model2.eval()
    test_l22 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out2 = model2(x).reshape(batch_size, h, h)
            out2 = y_normalizer.decode(out2)

            test_l22 += myloss(out2.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse2 /= len(train_loader)
    train_l22 /= ntrain
    test_l22 /= ntest

    train_loss2[ep] = train_l22
    test_loss2[ep] = test_l22

#     # Print results for both models
    t2 = default_timer()
    print("Epoch-{}, Time-{:0.4f}".format(ep, t2 - t1))
    print("Model1 -> Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}".format(train_mse1, train_l21, test_l21))
    print("Model2 -> Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}".format(train_mse2, train_l22, test_l22))

# # Save results
# results = {
#     "train_loss_model1": train_loss1.numpy(),
#     "test_loss_model1": test_loss1.numpy(),
#     "train_loss_model2": train_loss2.numpy(),
#     "test_loss_model2": test_loss2.numpy()
# }

# savemat("training_results.mat", results)



# # %%
# # """ The model definition """
# # model = WNO2d(width=width, level=level, layers=layers, size=[h,h], wavelet=wavelet,
# #               in_channel=in_channel, grid_range=grid_range, padding=1).to(device)
# # print(count_params(model))

# # """ Training and testing """
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# # train_loss = torch.zeros(epochs)
# # test_loss = torch.zeros(epochs)
# # myloss = LpLoss(size_average=False)
# # y_normalizer.to(device)
# # for ep in range(epochs):
# #     model.train()
# #     t1 = default_timer()
# #     train_mse = 0
# #     train_l2 = 0
# #     for x, y in train_loader:
# #         x, y = x.to(device), y.to(device)

# #         optimizer.zero_grad()
# #         out = model(x).reshape(batch_size, h, h)
# #         out = y_normalizer.decode(out)
# #         y = y_normalizer.decode(y)
        
# #         mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
# #         loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
# #         loss.backward()
# #         optimizer.step()
        
# #         train_mse += mse.item()
# #         train_l2 += loss.item()
    
# #     scheduler.step()
# #     model.eval()
# #     test_l2 = 0.0
# #     with torch.no_grad():
# #         for x, y in test_loader:
# #             x, y = x.to(device), y.to(device)

# #             out = model(x).reshape(batch_size, h, h)
# #             out = y_normalizer.decode(out)

# #             test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

# #     train_mse /= len(train_loader)
# #     train_l2/= ntrain
# #     test_l2 /= ntest
    
# #     train_loss[ep] = train_l2
# #     test_loss[ep] = test_l2
    
# #     t2 = default_timer()
# #     print("Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}"
# #           .format(ep, t2-t1, train_mse, train_l2, test_l2))
    
# # # %%
# # """ Prediction """
# # pred = []
# # test_e = []
# # with torch.no_grad():
    
# #     index = 0
# #     for x, y in test_loader:
# #         test_l2 = 0
# #         x, y = x.to(device), y.to(device)

# #         out = model(x).reshape(batch_size, h, h)
# #         out = y_normalizer.decode(out)
# #         pred.append( out.cpu() )

# #         test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
# #         test_e.append( test_l2/batch_size )
        
# #         print("Batch-{}, Loss-{}".format(index, test_l2/batch_size) )
# #         index += 1

# # pred = torch.cat((pred))
# # test_e = torch.tensor((test_e))  
# # print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# # # %%
# # """ Plotting """  
# # plt.rcParams["font.family"] = "serif"
# # plt.rcParams['font.size'] = 14

# # figure1 = plt.figure(figsize = (18, 14))
# # figure1.text(0.04,0.17,'\n Error', rotation=90, color='purple', fontsize=20)
# # figure1.text(0.04,0.34,'\n Prediction', rotation=90, color='green', fontsize=20)
# # figure1.text(0.04,0.57,'\n Truth', rotation=90, color='red', fontsize=20)
# # figure1.text(0.04,0.75,'Permeability \n field', rotation=90, color='b', fontsize=20)
# # plt.subplots_adjust(wspace=0.7)
# # index = 0
# # for value in range(y_test.shape[0]):
# #     if value % 26 == 1:
# #         plt.subplot(4,4, index+1)
# #         plt.imshow(x_test[value,:,:,0], cmap='rainbow', extent=[0,1,0,1], interpolation='Gaussian')
# #         plt.title('a(x,y)-{}'.format(index+1), color='b', fontsize=20, fontweight='bold')
# #         plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
# #         plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
# #         plt.subplot(4,4, index+1+4)
# #         plt.imshow(y_test[value,:,:], cmap='rainbow', extent=[0,1,0,1], interpolation='Gaussian')
# #         plt.colorbar(fraction=0.045)
# #         plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
# #         plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
# #         plt.subplot(4,4, index+1+8)
# #         plt.imshow(pred[value,:,:], cmap='rainbow', extent=[0,1,0,1], interpolation='Gaussian')
# #         plt.colorbar(fraction=0.045)
# #         plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
# #         plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
# #         plt.subplot(4,4, index+1+12)
# #         plt.imshow(np.abs(pred[value,:,:]-y_test[value,:,:]), cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
# #         plt.xlabel('x', fontweight='bold'); plt.ylabel('y', fontweight='bold'); 
# #         plt.colorbar(fraction=0.045,format='%.0e')
        
# #         plt.margins(0)
# #         index = index + 1

# # # %%
# # """
# # For saving the trained model and prediction data
# # """
# # torch.save(model, 'model/WNO_darcy')
# # scipy.io.savemat('results/wno_results_darcy.mat', mdict={'x_test':x_test.cpu().numpy(),
# #                                                     'y_test':y_test.cpu().numpy(),
# #                                                     'pred':pred.cpu().numpy(),  
# #                                                     'test_e':test_e.cpu().numpy()})
