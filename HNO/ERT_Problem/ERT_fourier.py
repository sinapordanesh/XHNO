#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:38:17 2024

@author: hossein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:21:04 2024

@author: hossein
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:56:39 2024

@author: hossein
"""

import os
# Set the current working directory
#os.chdir("/home/hossein/fourier-neural-operator-main")

import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import scipy.io as sio
#from Adam import Adam
torch.manual_seed(0)
np.random.seed(0)

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from hilbert import hilbert, hilbert2 # ADDED
import argparse             # ADDED
parser = argparse.ArgumentParser()

parser.add_argument("--fn", help="File Name for saving prediction and model files.", type=str, required=True)
parser.add_argument("--train", help="Train dataset file path.", type=str, required=True)
parser.add_argument("--test", help="Test dataset file path.", type=str, required=True)
parser.add_argument("--eval", help="Evaluation dataset file path.", type=str, required=True)
parser.add_argument("--ts", help="Train size of the training dataset", type=int, required=True)


args = parser.parse_args()

pfn = args.fn
train_path = args.train
test_path = args.test
eval_path = args.eval
train_size = args.ts

torch.manual_seed(0)
np.random.seed(0)
# Define dropout probability and Weight Decay
torch.cuda.empty_cache()
p =0
wd = 9e-4
patience = 20
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
        self.dropout = nn.Dropout(p=p)

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
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        #self.mlp3 = nn.Conv2d(mid_channels , out_channels,1)
        #self.mlp4 = nn.Conv2d(mid_channels*4,out_channels,1) 
        self.dropout = nn.Dropout(p=p)
    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        #x = F.gelu(x)
        #x = self.mlp3(x)
        #x = F.gelu(x)
        #x = self.mlp4(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, dropout_prob=p):  # Define a dropout probability
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 20  # pad the domain if input is non-periodic
        self.dropout_prob = dropout_prob  # Dropout probability

        self.p = nn.Linear(4, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dropout_conv0 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dropout_conv1 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dropout_conv2 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        #self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        #self.dropout_conv3 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.dropout_mlp0 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.dropout_mlp1 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.dropout_mlp2 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        #self.mlp3 = MLP(self.width, self.width, self.width)
        #self.dropout_mlp3 = nn.Dropout(p=self.dropout_prob)  # Add dropout layer
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        #self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x1 = self.dropout_conv0(x1)  # Apply dropout
        x1 = self.mlp0(x1)
        x1 = self.dropout_mlp0(x1)  # Apply dropout
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.dropout_conv1(x1)  # Apply dropout
        x1 = self.mlp1(x1)
        x1 = self.dropout_mlp1(x1)  # Apply dropout
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        #x1 = self.conv2(x)
        #x1 = self.dropout_conv2(x1)  # Apply dropout
        #x1 = self.mlp2(x1)
        #x1 = self.dropout_mlp2(x1)  # Apply dropout
        #x2 = self.w2(x)
        #x = x1 + x2
        #x = F.gelu(x)

        #x1 = self.conv3(x)
        #x1 = self.dropout_conv3(x1)  # Apply dropout
        #x1 = self.mlp3(x1)
        #x1 = self.dropout_mlp3(x1)  # Apply dropout
        #x2 = self.w3(x)
        #x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
TRAIN_PATH = train_path
TEST_PATH = test_path
NEW_TEST_PATH = eval_path



ntrain = train_size
ntest = 100
nval = 100
batch_size = 50
learning_rate = 0.01
epochs = 100
iterations = epochs*(ntrain//batch_size)

modes = 16
width = 18
modes1 = 16
modes2 = 16

r = 1

# ################################################################
# # load data and data normalization
# ################################################################

# from scipy.io import loadmat



reader = MatReader(TRAIN_PATH)
x_train1 = reader.read_field('coeff')[:ntrain,::r,::r]#[:,:s,:s]
x_train2 = reader.read_field('II')[:ntrain,::r,::r]#[:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r]#[:,:s,:s]
#just this time
#x_train1 = np.transpose(x_train1,axes=(0,2,1))
x_train1 = 1/x_train1
print(x_train1.shape)
print(x_train2.shape)
##
reader.load_file(TEST_PATH)
x_test1 = reader.read_field('coeff')[:ntest,::r,::r]#[:,:s,:s]
x_test2 = reader.read_field('II')[:ntest,::r,::r]#[:,:s,:s]
y_test = reader.read_field('sol')[:ntest,::r,::r]#[:,:s,:s]
## just this time
#x_test1 = np.transpose(x_test1,axes=(0,2,1))
x_test1 = 1/x_test1
print(x_test1.shape)
print(x_test2.shape)

##

# Load data from new_test.mat
reader.load_file(NEW_TEST_PATH)
x_new_test1 = reader.read_field('coeff')[:nval,::r,::r]
x_new_test2 = reader.read_field('II')[:nval,::r,::r]
print(x_new_test2.shape)

## just this time
#x_new_test1 = np.transpose(x_new_test1,axes=(0,2,1))
x_new_test1 = 1/x_new_test1
print(x_new_test1.shape)
##


x_normalizer1 = UnitGaussianNormalizer(x_train1)
x_train1 = x_normalizer1.encode(x_train1)
x_test1 = x_normalizer1.encode(x_test1)
x_new_test1 = x_normalizer1.encode(x_new_test1)

x_normalizer2 = UnitGaussianNormalizer(x_train2)
x_train2 = x_normalizer2.encode(x_train2)
x_test2 = x_normalizer2.encode(x_test2)
x_new_test2 = x_normalizer2.encode(x_new_test2)

x_test11 = x_test1.numpy();
dimension = x_test11.shape
s1 = dimension[1]
s2 = dimension[2]
print(s1)
print(s2)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
x_train =torch.zeros(ntrain,s1,s2,2)

for ii in range(ntrain):
    x_train[ii,:,:,1] = x_train1[ii,:,:]
    x_train[ii,:,:,0] = x_train2[ii,:,:]
    

x_test =torch.zeros(ntest,s1,s2,2)

for ii in range(ntest):
    x_test[ii,:,:,1] = x_test1[ii,:,:]
    x_test[ii,:,:,0] = x_test2[ii,:,:]    
    
    

# Create the input tensor
x_new_test = torch.zeros(x_new_test1.shape[0], s1, s2, 2)
for ii in range(x_new_test1.shape[0]):
    x_new_test[ii, :, :, 1] = x_new_test1[ii, :, :]
    x_new_test[ii, :, :, 0] = x_new_test2[ii, :, :]
    
    
    
    
    
    

# model = FNO2d(modes1=16, modes2=16, width=18)
# model.load_state_dict(torch.load('best_modelK1.pth'))

# model.to(device)  # Move model to device
# model.eval()
# # Additional code for making predictions on new_test.mat and measuring time


# # Predictions for new_test.mat
# start_time = default_timer()  # Start measuring time
# with torch.no_grad():
#     x_new_test = x_new_test.to(device)
#     out_new = model(x_new_test).reshape(-1, s1, s2)
#     y_normalizer.mean = y_normalizer.mean.to(device)
#     y_normalizer.std = y_normalizer.std.to(device)
#     # Assuming y_normalizer is defined elsewhere
#     out_new = y_normalizer.decode(out_new)
# end_time = default_timer()  # End measuring time

# # Calculate the time taken for predictions
# prediction_time = end_time - start_time
# print(f"Time taken for predictions on new_test.mat: {prediction_time} seconds")

# # Save the predicted matrices
# predicted = out_new.cpu().numpy()  # Move tensor to CPU before converting to numpy
# sio.savemat('predicted1K.mat', {'predicted': predicted})    

#x_train.reshape(ntrain,s1,s2,1)
#x_test = x_test.reshape(ntest,s1,s2,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)



# Run on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = FNO2d(modes, modes, width).to(device)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer.to(device)

train_losses = []
test_losses = []
y_predicted_all = []
y_input_all = []

#####
best_loss = float('inf')
  # Number of epochs to wait for improvment
counter = 0




for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            # Save y_predicted and y_input
            y_predicted_all.append(out.cpu().numpy())
            y_input_all.append(y.cpu().numpy())

    train_l2 /= ntrain
    test_l2 /= ntest

    train_losses.append(train_l2)
    test_losses.append(test_l2)

    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)


    if test_l2 <1.5*train_l2:
        best_loss = test_l2
        counter = 0
        # Save the model 
        torch.save(model.state_dict(), f'best_model_{pfn}.pth')
    else:
        counter +=1

    #Check if early stopping criteria met
    if counter >= patience:
        print(f"Early stopping after epoch {ep}.")
        break



#  # Save predicted matrices
 
# #sys.exit()
# y_predicted_all_combined = np.concatenate(y_predicted_all, axis=0)
# y_input_all_combined = np.concatenate(y_input_all, axis=0)
# a = len(y_input_all_combined)-1 - ntest
 
# y_input_all_combined_1 = y_input_all_combined[a:len(y_input_all_combined)-1,:,:]
# y_predicted_all_combined_1 = y_predicted_all_combined [a:len(y_input_all_combined)-1,:,:]
# sio.savemat('predicted_matricesRG.mat', {'y_predicted': y_predicted_all_combined_1 })
# sio.savemat('input_matrices.mat', {'y_input': y_input_all_combined_1 }) 



# # Additional code for making predictions on new_test.mat and measuring time
# NEW_TEST_PATH = 'data/evalK.mat'


# # Predictions for new_test.mat
# start_time = default_timer()  # Start measuring time
# with torch.no_grad():
#     x_new_test = x_new_test.to(device)
#     out_new = model(x_new_test).reshape(-1, s1, s2)
#     out_new = y_normalizer.decode(out_new)
# end_time = default_timer()  # End measuring time

# # Calculate the time taken for predictions
# prediction_time = end_time - start_time
# print(f"Time taken for predictions on new_test.mat: {prediction_time} seconds")


# # Save the predicted matrices
# predicted = out_new.cpu().numpy()
# sio.savemat('predictedK.mat', {'predicted': predicted})




model = FNO2d(modes1=16, modes2=16, width=18)
model.load_state_dict(torch.load(f'best_model_{pfn}.pth'))

model.to(device)  # Move model to device
model.eval()
# Additional code for making predictions on new_test.mat and measuring time


# Predictions for new_test.mat
start_time = default_timer()  # Start measuring time
with torch.no_grad():
    x_new_test = x_new_test.to(device)
    out_new = model(x_new_test).reshape(-1, s1, s2)
    y_normalizer.mean = y_normalizer.mean.to(device)
    y_normalizer.std = y_normalizer.std.to(device)
    # Assuming y_normalizer is defined elsewhere
    out_new = y_normalizer.decode(out_new)
end_time = default_timer()  # End measuring time

# Calculate the time taken for predictions
prediction_time = end_time - start_time
print(f"Time taken for predictions on new_test.mat: {prediction_time} seconds")

# Save the predicted matrices
predicted = out_new.cpu().numpy()  # Move tensor to CPU before converting to numpy
sio.savemat(f'output/predicted_{pfn}.mat', {'predicted': predicted})
