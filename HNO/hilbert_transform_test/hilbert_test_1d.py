# Import necessary libraries
import torch
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Function to compute the Hilbert transform using PyTorch
def hilbert_transform_torch(x):
    """
    Compute the Hilbert transform of x using PyTorch.
    x: tensor of shape [batchsize, channels, N]
    """
    N = x.size(-1)
    x_ft = torch.fft.fft(x, n=N)
    freqs = torch.fft.fftfreq(N).to(x.device)
    H = -1j * torch.sign(freqs)
    H = H.reshape(1, 1, N)
    x_ht_ft = x_ft * H
    x_ht = torch.fft.ifft(x_ht_ft, n=N).real
    return x_ht

# Test parameters
N = 1024  # Number of points
L = 2 * np.pi  # Period
t = np.linspace(0, L, N, endpoint=False)
omega = 5  # Frequency

# ===========================
# Test Case 1: sin(ωt)
# ===========================

# Create the test function: sin(ωt)
f_sin = np.sin(omega * t)

# Analytical Hilbert transform: -cos(ωt)
f_sin_ht_analytical = -np.cos(omega * t)

# Convert f_sin to torch tensor
f_sin_torch = torch.from_numpy(f_sin).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N]

# Compute Hilbert transform using PyTorch
f_sin_ht_torch = hilbert_transform_torch(f_sin_torch)
f_sin_ht_torch_np = f_sin_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_sin_hilbert_scipy = hilbert(f_sin)
f_sin_ht_scipy = np.imag(f_sin_hilbert_scipy)

# Compute relative errors
error_sin_torch = np.linalg.norm(f_sin_ht_torch_np - f_sin_ht_analytical) / np.linalg.norm(f_sin_ht_analytical)
error_sin_scipy = np.linalg.norm(f_sin_ht_scipy - f_sin_ht_analytical) / np.linalg.norm(f_sin_ht_analytical)

print(f"Relative error between PyTorch Hilbert transform and analytical result (sin): {error_sin_torch}")
print(f"Relative error between SciPy Hilbert transform and analytical result (sin): {error_sin_scipy}")

# Plot the results for sin(ωt)
plt.figure(figsize=(12, 6))
plt.plot(t, f_sin_ht_analytical, label='Analytical Hilbert Transform')
plt.plot(t, f_sin_ht_torch_np, '--', label='PyTorch Hilbert Transform')
plt.plot(t, f_sin_ht_scipy, ':', label='SciPy Hilbert Transform')
plt.legend()
plt.title('Hilbert Transform of sin(ωt)')
plt.xlabel('t')
plt.ylabel('Hilbert Transform')
plt.show()

# ===========================
# Test Case 2: cos(ωt)
# ===========================

# Create the test function: cos(ωt)
f_cos = np.cos(omega * t)

# Analytical Hilbert transform: sin(ωt)
f_cos_ht_analytical = np.sin(omega * t)

# Convert f_cos to torch tensor
f_cos_torch = torch.from_numpy(f_cos).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N]

# Compute Hilbert transform using PyTorch
f_cos_ht_torch = hilbert_transform_torch(f_cos_torch)
f_cos_ht_torch_np = f_cos_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_cos_hilbert_scipy = hilbert(f_cos)
f_cos_ht_scipy = np.imag(f_cos_hilbert_scipy)

# Compute relative errors
error_cos_torch = np.linalg.norm(f_cos_ht_torch_np - f_cos_ht_analytical) / np.linalg.norm(f_cos_ht_analytical)
error_cos_scipy = np.linalg.norm(f_cos_ht_scipy - f_cos_ht_analytical) / np.linalg.norm(f_cos_ht_analytical)

print(f"Relative error between PyTorch Hilbert transform and analytical result (cos): {error_cos_torch}")
print(f"Relative error between SciPy Hilbert transform and analytical result (cos): {error_cos_scipy}")

# Plot the results for cos(ωt)
plt.figure(figsize=(12, 6))
plt.plot(t, f_cos_ht_analytical, label='Analytical Hilbert Transform')
plt.plot(t, f_cos_ht_torch_np, '--', label='PyTorch Hilbert Transform')
plt.plot(t, f_cos_ht_scipy, ':', label='SciPy Hilbert Transform')
plt.legend()
plt.title('Hilbert Transform of cos(ωt)')
plt.xlabel('t')
plt.ylabel('Hilbert Transform')
plt.show()

# ===========================
# Test Case 3: Exponential Function e^(iωt)
# ===========================

# Create the test function: e^(iωt)
f_exp = np.exp(1j * omega * t)

# Analytical Hilbert transform: e^(i(ωt - π/2)) for ω > 0
f_exp_ht_analytical = np.exp(1j * (omega * t - np.pi / 2))

# Convert f_exp to torch tensor (real and imaginary parts separately)
f_exp_real_torch = torch.from_numpy(np.real(f_exp)).float().unsqueeze(0).unsqueeze(0)
f_exp_imag_torch = torch.from_numpy(np.imag(f_exp)).float().unsqueeze(0).unsqueeze(0)

# Compute Hilbert transform using PyTorch (real and imaginary parts separately)
f_exp_ht_real_torch = hilbert_transform_torch(f_exp_real_torch)
f_exp_ht_imag_torch = hilbert_transform_torch(f_exp_imag_torch)

# Combine real and imaginary parts
f_exp_ht_torch = f_exp_ht_real_torch.detach().numpy().squeeze() + 1j * f_exp_ht_imag_torch.detach().numpy().squeeze()

# Compute relative error (magnitude)
error_exp_torch = np.linalg.norm(f_exp_ht_torch - f_exp_ht_analytical) / np.linalg.norm(f_exp_ht_analytical)

print(f"Relative error between PyTorch Hilbert transform and analytical result (exp): {error_exp_torch}")

# Plot the results for e^(iωt)
plt.figure(figsize=(12, 6))
plt.plot(t, np.real(f_exp_ht_analytical), label='Analytical Hilbert Transform (Real)')
plt.plot(t, np.real(f_exp_ht_torch), '--', label='PyTorch Hilbert Transform (Real)')
plt.legend()
plt.title('Hilbert Transform of e^(iωt) - Real Part')
plt.xlabel('t')
plt.ylabel('Real Part')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, np.imag(f_exp_ht_analytical), label='Analytical Hilbert Transform (Imag)')
plt.plot(t, np.imag(f_exp_ht_torch), '--', label='PyTorch Hilbert Transform (Imag)')
plt.legend()
plt.title('Hilbert Transform of e^(iωt) - Imaginary Part')
plt.xlabel('t')
plt.ylabel('Imaginary Part')
plt.show()
