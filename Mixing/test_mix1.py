#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:29:18 2024

@author: hossein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:15:05 2024

@author: hossein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:46:18 2024

@author: hossein
"""

'''
Module Description:
------
This module implements the Laplace Neural Operator for beam (Example 7 in LNO paper)
Author: 
------
Qianying Cao (qianying_cao@brown.edu)
'''

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
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT 
except ImportError:
    print('Wavelet convolution requires <Pytorch Wavelets>, <PyWavelets>, <Pytorch Wavelet Toolbox> \n \
                    For Pytorch Wavelet Toolbox: $ pip install ptwt \n \
                    For PyWavelets: $ conda install pywavelets \n \
                    For Pytorch Wavelets: $ git clone https://github.com/fbcotter/pytorch_wavelets \n \
                                          $ cd pytorch_wavelets \n \
                                          $ pip install .')

# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================  
################################################################
# fourier layer
################################################################
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
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(self.nor(x))
        x2 = self.w0(x)
        x = x1 + x2

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




class LpLoss(nn.Module):
    def __init__(self, d=2, p=2):
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p

    def forward(self, y_pred, y_true):
        batch_size = y_true.size(0)
        diff = torch.abs(y_pred - y_true) ** self.p
        loss = torch.mean(torch.sum(diff, dim=tuple(range(1, self.d + 1))) ** (1.0 / self.p))
        return loss




def train_model(model, train_loader, vali_loader, epochs, learning_rate, step_size, gamma):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    lp_loss = LpLoss(d=2, p=2)  # L2 Loss in 2D

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = lp_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        if epoch % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in vali_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred = model(x_batch)
                    loss = lp_loss(y_pred, y_batch)
                    val_loss += loss.item()
            val_loss /= len(vali_loader)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")




def evaluate_model(model, test_loader):
    model.eval()
    lp_loss = LpLoss(d=2, p=2)  # L2 Loss in 2D
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = lp_loss(y_pred, y_batch)
            test_loss += loss.item()
    return test_loss / len(test_loader)




# ====================================
#  Define parameters and Load data
# ====================================
ntrain = 2000
nvali = 100
ntest=130

batch_size_train = 50
batch_size_vali = 50

learning_rate = 0.002

epochs = 100
step_size = 100
gamma = 0.5

modes1 = 4
modes2 = 4
width = 16

reader = MatReader('data_beam.mat')
reader1 = MatReader('lognorm_dataset.mat')

x_train = reader1.read_field('lognorm_a_training')
y_train = reader1.read_field('lognorm_p_training')
T = reader.read_field('t')
X = reader.read_field('x')

x_vali = reader1.read_field('lognorm_a_validation')
y_vali = reader1.read_field('lognorm_p_validation')

x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')

x_train = x_train.reshape(ntrain,x_train.shape[1],x_train.shape[2],1)
x_vali = x_vali.reshape(nvali,x_vali.shape[1],x_vali.shape[2],1)
x_test = x_test.reshape(ntest,x_test.shape[1],x_test.shape[2],1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)
# model
model = LNO2d(width,modes1, modes2).cuda()

# Fetch a single batch from x_train
x_batch, _ = next(iter(train_loader))  # Fetch one batch from the DataLoader

# Move the batch to the same device as the model
x_batch = x_batch.cuda()
output = model(x_batch)



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)


# Fetch a single batch from x_train
x_batch, _ = next(iter(train_loader))  # Fetch one batch from the DataLoader

# Move the batch to the same device as the model
x_batch = x_batch.cuda()
output = model(x_batch)

model1 = FNO2d(width,modes1, modes2).cuda()
output1 = model1(x_batch)
# Scenario 1: Pure FNO2D
pure_fno_model = FNO2d(modes1, modes2, width).to(device)
# Train models
print("Training Pure FNO2D Model...")
train_model(pure_fno_model, train_loader, vali_loader, epochs, learning_rate, step_size, gamma)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)
 



results = {
    "Pure FNO2D": evaluate_model(pure_fno_model, test_loader),
}

# Print results
print("Test Results:")
for config, loss in results.items():
    print(f"{config}: Test Loss = {loss:.4f}")


# class PR2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(PR2d, self).__init__()

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.scale = (1 / (in_channels*out_channels))
#         self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
#         self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
#         self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, dtype=torch.cfloat))
    
#     def output_PR(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
#         Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
#         term1=torch.div(1,torch.einsum("pbix,qbik->pqbixk",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2)))
#         Hw=torch.einsum("bixk,pqbixk->pqbixk",weights_residue,term1)
#         Pk=Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
#         output_residue1=torch.einsum("biox,oxikpq->bkox", alpha, Hw) 
#         output_residue2=torch.einsum("biox,oxikpq->bkpq", alpha, Pk) 
#         return output_residue1,output_residue2

#     def forward(self, x):
#         tx=T.cuda()
#         ty=X.cuda()
#         #Compute input poles and resudes by FFT
#         dty=(ty[0,1]-ty[0,0]).item()  # location interval
#         dtx=(tx[0,1]-tx[0,0]).item()  # time interval
#         alpha = torch.fft.fft2(x, dim=[-2,-1])
#         omega1=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
#         omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # time frequency
#         omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         lambda1=omega1.cuda()
#         lambda2=omega2.cuda()
 
#         # Obtain output poles and residues for transient part and steady-state part
#         output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)

#         # Obtain time histories of transient response and steady-state response
#         x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
#         x1 = torch.real(x1)    
#         term1=torch.einsum("bip,kz->bipz", self.weights_pole1, ty.type(torch.complex64).reshape(1,-1))
#         term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
#         term3=torch.einsum("bipz,biqx->bipqzx", torch.exp(term1),torch.exp(term2))
#         x2=torch.einsum("kbpq,bipqzx->kizx", output_residue2,term3)
#         x2=torch.real(x2)
#         x2=x2/x.size(-1)/x.size(-2)
#         return x1+x2


# class LNO2d(nn.Module):
#     def __init__(self, modes1, modes2,  width):
#         super(LNO2d, self).__init__()

#         self.width = width
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.fc0 = nn.Linear(3, self.width) 

#         self.conv0 = PR2d(self.width, self.width, self.modes1, self.modes2)
#         self.w0 = nn.Conv2d(self.width, self.width, 1)
#         self.norm = nn.InstanceNorm2d(self.width)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self,x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)

#         x1 = self.norm(self.conv0(self.norm(x)))
#         x2 = self.w0(x)
#         x = x1 +x2

#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = torch.sin(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)



# class WaveConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric'):
#         super(WaveConv2d, self).__init__()

#         """
#         2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
#         Input parameters: 
#         -----------------
#         in_channels  : scalar, input kernel dimension
#         out_channels : scalar, output kernel dimension
#         level        : scalar, levels of wavelet decomposition
#         size         : scalar, length of input 1D signal
#         wavelet      : string, wavelet filters
#         mode         : string, padding style for wavelet decomposition
        
#         It initializes the kernel parameters: 
#         -------------------------------------
#         self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Approximate wavelet coefficients
#         self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Horizontal-Detailed wavelet coefficients
#         self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Vertical-Detailed wavelet coefficients
#         self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Diagonal-Detailed wavelet coefficients
#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.level = level
#         if isinstance(size, list):
#             if len(size) != 2:
#                 raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
#             else:
#                 self.size = size
#         else:
#             raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
#         self.wavelet = wavelet       
#         self.mode = mode
#         dummy_data = torch.randn( 1,1,*self.size )        
#         dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
#         mode_data, mode_coef = dwt_(dummy_data)
#         self.modes1 = mode_data.shape[-2]
#         self.modes2 = mode_data.shape[-1]
        
#         # Parameter initilization
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
#         self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
#         self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

#     # Convolution
#     def mul2d(self, input, weights):
#         """
#         Performs element-wise multiplication

#         Input Parameters
#         ----------------
#         input   : tensor, shape-(batch * in_channel * x * y )
#                   2D wavelet coefficients of input signal
#         weights : tensor, shape-(in_channel * out_channel * x * y)
#                   kernel weights of corresponding wavelet coefficients

#         Returns
#         -------
#         convolved signal : tensor, shape-(batch * out_channel * x * y)
#         """
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x):
#         """
#         Input parameters: 
#         -----------------
#         x : tensor, shape-[Batch * Channel * x * y]
#         Output parameters: 
#         ------------------
#         x : tensor, shape-[Batch * Channel * x * y]
#         """
#         if x.shape[-1] > self.size[-1]:
#             factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
#             # Compute single tree Discrete Wavelet coefficients using some wavelet
#             dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
#             x_ft, x_coeff = dwt(x)
            
#         elif x.shape[-1] < self.size[-1]:
#             factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
#             # Compute single tree Discrete Wavelet coefficients using some wavelet
#             dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
#             x_ft, x_coeff = dwt(x)
        
#         else:
#             # Compute single tree Discrete Wavelet coefficients using some wavelet
#             dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
#             x_ft, x_coeff = dwt(x)

#         # Instantiate higher level coefficients as zeros
#         out_ft = torch.zeros_like(x_ft, device= x.device)
#         out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
#         # Multiply the final approximate Wavelet modes
#         out_ft = self.mul2d(x_ft, self.weights1)
#         # Multiply the final detailed wavelet coefficients
#         out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
#         out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
#         out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
#         # Return to physical space        
#         idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
#         x = idwt((out_ft, out_coeff))
#         return x

# class WNO2d(nn.Module):
#     def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
#         super(WNO2d, self).__init__()

#         """
#         The WNO network. It contains l-layers of the Wavelet integral layer.
#         1. Lift the input using v(x) = self.fc0 .
#         2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
#             --> W is defined by self.w; K is defined by self.conv.
#         3. Project the output of last layer using self.fc1 and self.fc2.
        
#         Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
#               : shape: (batchsize * x=width * x=height * c=3)
#         Output: Solution of a later timestep (u(x,y))
#               : shape: (batchsize * x=width * x=height * c=1)
        
#         Input parameters:
#         -----------------
#         width : scalar, lifting dimension of input
#         level : scalar, number of wavelet decomposition
#         layers: scalar, number of wavelet kernel integral blocks
#         size  : list with 2 elements (for 2D), image size
#         wavelet: string, wavelet filter
#         in_channel: scalar, channels in input including grid
#         grid_range: list with 2 elements (for 2D), right supports of 2D domain
#         padding   : scalar, size of zero padding
#         """

#         self.level = level
#         self.width = width
#         self.layers = layers
#         self.size = size
#         self.wavelet = wavelet
#         self.in_channel = in_channel
#         self.grid_range = grid_range 
#         self.padding = padding
        
#         self.conv = nn.ModuleList()
#         self.w = nn.ModuleList()
        
#         self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
#         for i in range( self.layers ):
#             self.conv.append( WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet) )
#             self.w.append( nn.Conv2d(self.width, self.width, 1) )
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)    
#         x = self.fc0(x)                      # Shape: Batch * x * y * Channel
#         x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
#         if self.padding != 0:
#             x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
#         for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
#             x = convl(x) + wl(x) 
#             if index != self.layers - 1:     # Final layer has no activation    
#                 x = F.mish(x)                # Shape: Batch * Channel * x * y
                
#         if self.padding != 0:
#             x = x[..., :-self.padding, :-self.padding]     
#         x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
#         x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
#         x = self.fc2(x)                      # Shape: Batch * x * y * Channel
#         return x
    
#     def get_grid(self, shape, device):
#         # The grid of the solution
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)








# class FWNO2d(nn.Module):
#     def __init__(self, modes1, modes2,  width,level, layers, size, wavelet, in_channel, grid_range, padding=0):
#         super(FWNO2d, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the coefficient function and locations (a(x, y), x, y)
#         input shape: (batchsize, x=s, y=s, c=3)
#         output: the solution 
#         output shape: (batchsize, x=s, y=s, c=1)
#         """

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width = width
#         self.padding = padding # pad the domain if input is non-periodic
#         self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

#         self.w0 = nn.Conv2d(self.width, self.width, 1)


#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)
#         x = F.pad(x, [0,self.padding, 0,self.padding])

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2


#         x = x[..., :-self.padding, :-self.padding]
#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x
    
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)




















# class FLNO2d(nn.Module):
#     def __init__(self, modes1, modes2,  width):
#         super(FLNO2d, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the coefficient function and locations (a(x, y), x, y)
#         input shape: (batchsize, x=s, y=s, c=3)
#         output: the solution 
#         output shape: (batchsize, x=s, y=s, c=1)
#         """

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width = width
#         self.padding = 9 # pad the domain if input is non-periodic
#         self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
#         self.conv1 = PR2d(self.width, self.width, self.modes1, self.modes2)
#         self.w0 = nn.Conv2d(self.width, self.width, 1)
#         self.norm = nn.InstanceNorm2d(self.width)
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)
#     def forward(self,x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1,2)

#     # Apply the convolutional layers properly
#         x1 = self.conv0(self.norm(x))
#         x2 = self.w0(x)
#         x3 = self.conv1(self.norm(x))  # Apply PR2d layer to x
#     # Combine the layers' outputs
#         x = x1 + x2 + x3



#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = torch.sin(x)
#         x = self.fc2(x)
#         return x
        
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

# class PureLaplaceFourier2D(nn.Module):
#     def __init__(self, width, modes1, modes2, n_laplace_layers, n_fourier_layers):
#         """
#         Args:
#             width: Channel width of the model.
#             modes1: Number of modes in one direction.
#             modes2: Number of modes in another direction.
#             n_laplace_layers: Number of Laplace layers.
#             n_fourier_layers: Number of Fourier layers.
#         """
#         super(PureLaplaceFourier2D, self).__init__()

#         self.width = width
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.n_laplace_layers = n_laplace_layers
#         self.n_fourier_layers = n_fourier_layers

#         # Input layer
#         self.fc0 = nn.Linear(3, self.width)

#         # Laplace layers and skip connections
#         self.laplace_layers = nn.ModuleList()
#         self.laplace_skip_layers = nn.ModuleList()
#         for _ in range(n_laplace_layers):
#             self.laplace_layers.append(PR2d(self.width, self.width, self.modes1, self.modes2))
#             self.laplace_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

#         # Fourier layers and skip connections
#         self.fourier_layers = nn.ModuleList()
#         self.fourier_skip_layers = nn.ModuleList()
#         for _ in range(n_fourier_layers):
#             self.fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
#             self.fourier_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

#         # Normalization layer
#         self.norm = nn.InstanceNorm2d(self.width)

#         # Output layers
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         # Add grid information
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)  # Change to (batch, channels, x, y)

#         # Apply Laplace layers with skip connections
#         for laplace_layer, skip_layer in zip(self.laplace_layers, self.laplace_skip_layers):
#             x1 = self.norm(laplace_layer(self.norm(x)))
#             x2 = skip_layer(x)
#             x = F.gelu(x1 + x2)

#         # Apply Fourier layers with skip connections
#         for fourier_layer, skip_layer in zip(self.fourier_layers, self.fourier_skip_layers):
#             x1 = self.norm(fourier_layer(self.norm(x)))
#             x2 = skip_layer(x)
#             x = F.gelu(x1 + x2)

#         # Final processing
#         x = x.permute(0, 2, 3, 1)  # Change back to (batch, x, y, channels)
#         x = self.fc1(x)
#         x = torch.sin(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         """Generate grid information."""
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1)


# class PureLaplace2D(nn.Module):
#     def __init__(self, width, modes1, modes2, n_layers):
#         """
#         Args:
#             width: Channel width of the model.
#             modes1: Number of Laplace modes in one direction.
#             modes2: Number of Laplace modes in another direction.
#             n_layers: Number of Laplace layers.
#         """
#         super(PureLaplace2D, self).__init__()

#         self.width = width
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.n_layers = n_layers

#         # Input layer
#         self.fc0 = nn.Linear(3, self.width)

#         # Laplace layers and skip connections
#         self.laplace_layers = nn.ModuleList()
#         self.skip_layers = nn.ModuleList()
#         for _ in range(n_layers):
#             self.laplace_layers.append(PR2d(self.width, self.width, self.modes1, self.modes2))
#             self.skip_layers.append(nn.Conv2d(self.width, self.width, 1))

#         # Normalization layer
#         self.norm = nn.InstanceNorm2d(self.width)

#         # Output layers
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         # Add grid information
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)  # Change to (batch, channels, x, y)

#         # Apply Laplace layers with skip connections
#         for laplace_layer, skip_layer in zip(self.laplace_layers, self.skip_layers):
#             x1 = self.norm(laplace_layer(self.norm(x)))
#             x2 = skip_layer(x)
#             x = F.gelu(x1 + x2)

#         # Final processing
#         x = x.permute(0, 2, 3, 1)  # Change back to (batch, x, y, channels)
#         x = self.fc1(x)
#         x = torch.sin(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         """Generate grid information."""
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1)






# class LaplaceFourierMixed2D(nn.Module):
#     def __init__(self, width, modes1, modes2, n_laplace_layers, n_fourier_layers, n_mixed_layers):
#         """
#         Args:
#             width: Channel width of the model.
#             modes1: Number of modes in one direction.
#             modes2: Number of modes in another direction.
#             n_laplace_layers: Number of Laplace layers.
#             n_fourier_layers: Number of Fourier layers.
#             n_mixed_layers: Number of mixed Fourier-Laplace layers.
#         """
#         super(LaplaceFourierMixed2D, self).__init__()

#         self.width = width
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.n_laplace_layers = n_laplace_layers
#         self.n_fourier_layers = n_fourier_layers
#         self.n_mixed_layers = n_mixed_layers

#         # Input layer
#         self.fc0 = nn.Linear(3, self.width)

#         # Fourier layers and skip connections
#         self.fourier_layers = nn.ModuleList()
#         self.fourier_skip_layers = nn.ModuleList()
#         for _ in range(n_fourier_layers):
#             self.fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
#             self.fourier_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

#         # Mixed layers and skip connections
#         self.mixed_layers = nn.ModuleList()
#         self.mixed_skip_layers = nn.ModuleList()
#         for _ in range(n_mixed_layers):
#             self.mixed_layers.append(nn.ModuleList([
#                 SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
#                 PR2d(self.width, self.width, self.modes1, self.modes2)
#             ]))
#             self.mixed_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

#         # Laplace layers and skip connections
#         self.laplace_layers = nn.ModuleList()
#         self.laplace_skip_layers = nn.ModuleList()
#         for _ in range(n_laplace_layers):
#             self.laplace_layers.append(PR2d(self.width, self.width, self.modes1, self.modes2))
#             self.laplace_skip_layers.append(nn.Conv2d(self.width, self.width, 1))

#         # Normalization layer
#         self.norm = nn.InstanceNorm2d(self.width)

#         # Output layers
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         # Add grid information
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)  # Change to (batch, channels, x, y)

#         # Apply Fourier layers with skip connections
#         for fourier_layer, skip_layer in zip(self.fourier_layers, self.fourier_skip_layers):
#             x1 = self.norm(fourier_layer(self.norm(x)))
#             x2 = skip_layer(x)
#             x = F.gelu(x1 + x2)

#         # Apply Mixed Fourier-Laplace layers with skip connections
#         for mixed_layer, skip_layer in zip(self.mixed_layers, self.mixed_skip_layers):
#             x1 = self.norm(mixed_layer[0](self.norm(x)))  # Fourier branch
#             x2 = self.norm(mixed_layer[1](self.norm(x)))  # Laplace branch
#             x3 = skip_layer(x)
#             x = F.gelu(x1 + x2 + x3)

#         # Apply Laplace layers with skip connections
#         for laplace_layer, skip_layer in zip(self.laplace_layers, self.laplace_skip_layers):
#             x1 = self.norm(laplace_layer(self.norm(x)))
#             x2 = skip_layer(x)
#             x = F.gelu(x1 + x2)

#         # Final processing
#         x = x.permute(0, 2, 3, 1)  # Change back to (batch, x, y, channels)
#         x = self.fc1(x)
#         x = torch.sin(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         """Generate grid information."""
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.linspace(0, 1, size_x, device=device).view(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1)




# class LpLoss(nn.Module):
#     def __init__(self, d=2, p=2):
#         super(LpLoss, self).__init__()
#         self.d = d
#         self.p = p

#     def forward(self, y_pred, y_true):
#         batch_size = y_true.size(0)
#         diff = torch.abs(y_pred - y_true) ** self.p
#         loss = torch.mean(torch.sum(diff, dim=tuple(range(1, self.d + 1))) ** (1.0 / self.p))
#         return loss








# def train_model(model, train_loader, vali_loader, epochs, learning_rate, step_size, gamma):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#     lp_loss = LpLoss(d=2, p=2)  # L2 Loss in 2D

#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             y_pred = model(x_batch)
#             loss = lp_loss(y_pred, y_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
        
#         train_loss /= len(train_loader)
#         scheduler.step()

#         if epoch % 100 == 0:
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for x_batch, y_batch in vali_loader:
#                     x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                     y_pred = model(x_batch)
#                     loss = lp_loss(y_pred, y_batch)
#                     val_loss += loss.item()
#             val_loss /= len(vali_loader)
#             print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")




# def evaluate_model(model, test_loader):
#     model.eval()
#     lp_loss = LpLoss(d=2, p=2)  # L2 Loss in 2D
#     test_loss = 0.0
#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             y_pred = model(x_batch)
#             loss = lp_loss(y_pred, y_batch)
#             test_loss += loss.item()
#     return test_loss / len(test_loader)



# # ====================================
# #  Define parameters and Load data
# # ====================================
# ntrain = 200
# nvali = 50
# ntest=130

# batch_size_train = 50
# batch_size_vali = 50

# learning_rate = 0.002

# epochs = 100
# step_size = 100
# gamma = 0.5

# modes1 = 4  
# modes2 = 4   
# width = 16

# reader = MatReader('data_beam.mat')
# x_train = reader.read_field('f_train')
# y_train = reader.read_field('u_train')
# T = reader.read_field('t')
# X = reader.read_field('x')

# x_vali = reader.read_field('f_vali')
# y_vali = reader.read_field('u_vali')

# x_test = reader.read_field('f_test')
# y_test = reader.read_field('u_test')

# x_train = x_train.reshape(ntrain,x_train.shape[1],x_train.shape[2],1)
# x_vali = x_vali.reshape(nvali,x_vali.shape[1],x_vali.shape[2],1)
# x_test = x_test.reshape(ntest,x_test.shape[1],x_test.shape[2],1)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
# vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)
# # model
# model = LNO2d(width,modes1, modes2).cuda()

# # Fetch a single batch from x_train
# x_batch, _ = next(iter(train_loader))  # Fetch one batch from the DataLoader

# # Move the batch to the same device as the model
# x_batch = x_batch.cuda()
# output = model(x_batch)

# model1 = FNO2d(width,modes1, modes2).cuda()
# output1 = model1(x_batch)


# print(f"Input shape: {x_batch.shape}")
# print(f"LNO Output shape: {output.shape}")
# print(f"FNO Output shape: {output1.shape}")


# model2 = FLNO2d(width,modes1, modes2).cuda()
# output2 = model2(x_batch)











# # # Forward pass through the model


# # # Print shapes of the input and output


# # # Print shapes of the input and output
# # print(f"Input shape: {x_batch.shape}")
# # print(f"FNO Output shape: {output.shape}")
# # print(f"LNO Output shape: {output1.shape}")
# # print(f"FLNO Output shape: {output2.shape}")



# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # Move model to the correct device
# # model3 =  PureLaplace2D(width,modes1,modes2,1).to(device)

# # # Move input data to the correct device
# # x_batch = x_batch.to(device)

# # # Perform forward pass
# # output3 = model3(x_batch)


# # print(f"Pure Laplace Output shape: {output3.shape}")







# # # Move model to the correct device
# # model4 =  PureLaplaceFourier2D(width,modes1,modes2,2,2).to(device)

# # # Move input data to the correct device
# # x_batch = x_batch.to(device)

# # # Perform forward pass
# # output4 = model4(x_batch)


# # print(f"Pure Laplace-Pure Fourier Output shape: {output3.shape}")





# # model5 =  LaplaceFourierMixed2D(width,modes1,modes2,0,0,2).to(device)

# # # Move input data to the correct device
# # x_batch = x_batch.to(device)

# # # Perform forward pass
# # output5 = model5(x_batch)








# # Scenario 1: Pure FNO2D
# pure_fno_model = FNO2d(modes1, modes2, width).to(device)

# # Scenario 2: Pure LNO2D
# pure_lno_model = LNO2d( modes1, modes2, width).to(device)

# # Scenario 3: Mixed FLNO2D
# mixed_flno_model = FLNO2d( modes1, modes2, width).to(device)

# # Scenario 4: (1 Laplace, 0 Fourier, 0 Mixed)
# laplace_config = LaplaceFourierMixed2D(width, modes1, modes2, n_laplace_layers=1, n_fourier_layers=0, n_mixed_layers=0).to(device)

# # Scenario 5: (0 Laplace, 1 Fourier, 0 Mixed)
# fourier_config = LaplaceFourierMixed2D(width, modes1, modes2, n_laplace_layers=0, n_fourier_layers=1, n_mixed_layers=0).to(device)

# # Scenario 6: (0 Laplace, 0 Fourier, 1 Mixed)
# mixed_config = LaplaceFourierMixed2D(width, modes1, modes2, n_laplace_layers=0, n_fourier_layers=0, n_mixed_layers=1).to(device)










# # Train models
# print("Training Pure FNO2D Model...")
# train_model(pure_fno_model, train_loader, vali_loader, epochs, learning_rate, step_size, gamma)

# print("Training Pure LNO2D Model...")
# train_model(pure_lno_model, train_loader, vali_loader, epochs, learning_rate, step_size, gamma)

# print("Training Mixed FLNO2D Model...")
# train_model(mixed_flno_model, train_loader, vali_loader, epochs, learning_rate, step_size, gamma)

# # Evaluate models
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size_vali, shuffle=False)

# results = {
#     "Pure FNO2D": evaluate_model(pure_fno_model, test_loader),
#     "Pure LNO2D": evaluate_model(pure_lno_model, test_loader),
#     "Mixed FLNO2D": evaluate_model(mixed_flno_model, test_loader),
#     "(1 Laplace, 0 Fourier, 0 Mixed)": evaluate_model(laplace_config, test_loader),
#     "(0 Laplace, 1 Fourier, 0 Mixed)": evaluate_model(fourier_config, test_loader),
#     "(0 Laplace, 0 Fourier, 1 Mixed)": evaluate_model(mixed_config, test_loader)
# }

# # Print results
# print("Test Results:")
# for config, loss in results.items():
#     print(f"{config}: Test Loss = {loss:.4f}")

































# # #print(f"Mix Laplace- Fourier Output shape: {output3.shape}")


# # # # ====================================
# # # # Training 
# # # # ====================================
# # optimizer = Adam(model5.parameters(), lr=learning_rate, weight_decay=1e-4)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# # start_time = time.time()
# # myloss = LpLoss(size_average=True)

# # train_error = np.zeros((epochs, 1))
# # train_loss = np.zeros((epochs, 1))
# # vali_error = np.zeros((epochs, 1))
# # vali_loss = np.zeros((epochs, 1))
# # for ep in range(epochs):
# #      model.train()
# #      t1 = default_timer()
# #      train_mse = 0
# #      train_l2 = 0
# #      n_train=0
# #      for x, y in train_loader:
# #          x, y = x.cuda(), y.cuda()

# #          optimizer.zero_grad()
# #          out = model(x)   
# #          mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
# #          l2 = myloss(out.view(-1,x_train.shape[1],x_train.shape[2]), y)
# #          l2.backward()

# #          optimizer.step()
# #          train_mse += mse.item()
# #          train_l2 += l2.item()
# #          n_train += 1

# #      scheduler.step()
# #      model.eval()
# #      vali_mse = 0.0
# #      vali_l2 = 0.0
# #      with torch.no_grad():
# #          n_vali=0
# #          for x, y in vali_loader:
# #              x, y = x.cuda(), y.cuda()
# #              out = model(x)
# #              mse=F.mse_loss(out.view(-1,x_vali.shape[1],x_vali.shape[2]), y, reduction='mean')
# #              vali_l2 += myloss(out.view(-1,x_vali.shape[1],x_vali.shape[2]), y).item()
# #              vali_mse += mse.item()
# #              n_vali += 1

# #      train_mse /= n_train
# #      vali_mse /= n_vali
# #      train_l2 /= n_train
# #      vali_l2 /= n_vali
# #      train_error[ep,0] = train_l2
# #      vali_error[ep,0] = vali_l2
# #      train_loss[ep,0] = train_mse
# #      vali_loss[ep,0] = vali_mse
# #      t2 = default_timer()
# #      print("Epoch: %d, time: %.3f, Train Loss: %.3e,Vali Loss: %.3e, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_mse, vali_mse,train_l2, vali_l2))
# #      elapsed = time.time() - start_time
# #  print("\n=============================")
# #  print("Training done...")
# #  print('Training time: %.3f'%(elapsed))
# #  print("=============================\n")

# # # ====================================
# # # saving settings
# # # ====================================
# # current_directory = os.getcwd()
# # case = "Case_"
# # save_index = 1  
# # folder_index = str(save_index)

# # results_dir = "/" + case + folder_index +"/"
# # save_results_to = current_directory + results_dir
# # if not os.path.exists(save_results_to):
# #     os.makedirs(save_results_to)

# # x = np.linspace(0, epochs-1, epochs)
# # np.savetxt(save_results_to+'/epoch.txt', x)
# # np.savetxt(save_results_to+'/train_loss.txt', train_loss)
# # np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
# # np.savetxt(save_results_to+'/train_error.txt', train_error)
# # np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
# # save_models_to = save_results_to +"model/"
# # if not os.path.exists(save_models_to):
# #     os.makedirs(save_models_to)
    
# # torch.save(model, save_models_to+'Wave_states')

# # # ====================================
# # # testing
# # # ====================================
# # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
# # pred_u = torch.zeros(ntest,y_test.shape[1],y_test.shape[2])
# # index = 0
# # test_l2 = 0.0
# # with torch.no_grad():
# #     for x, y in test_loader:
# #         x, y = x.cuda(), y.cuda()
# #         out = model(x)
# #         test_l2 += myloss(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
# #         pred_u[index,:,:] = out.view(-1,x_test.shape[1],x_test.shape[2])
# #         index = index + 1
# # test_l2 /= index
# # scipy.io.savemat(save_results_to+'wave_states_test.mat', 
# #                      mdict={ 'test_err': test_l2,
# #                             'T': T.numpy(),
# #                             'X': X.numpy(),
# #                             'y_test': y_test.numpy(), 
# #                             'y_pred': pred_u.cpu().numpy()})  
    
    
# # print("\n=============================")
# # print('Testing error: %.3e'%(test_l2))
# # print("=============================\n")


# # # Plotting the loss history
# # num_epoch = epochs
# # epoch = np.linspace(1, num_epoch, num_epoch)
# # fig = plt.figure(constrained_layout=False, figsize=(7, 7))
# # gs = fig.add_gridspec(1, 1)
# # ax = fig.add_subplot(gs[0])
# # ax.plot(epoch, train_loss[:,0], color='blue', label='Train Loss')
# # ax.plot(epoch, vali_loss[:,0], color='red', label='Validation Loss')
# # ax.set_yscale('log')
# # ax.set_ylabel('Loss')
# # ax.set_xlabel('Epochs')
# # ax.legend(loc='upper left')
# # fig.savefig(save_results_to+'loss_history.png')
# # plt.show()