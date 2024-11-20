# Import necessary libraries
import torch
import numpy as np
from scipy.signal import hilbert2
import matplotlib.pyplot as plt

# Function to compute the 2D Hilbert transform using PyTorch
def hilbert_transform_2d_torch(x):
    """
    Compute the 2D Hilbert transform of x using PyTorch.
    x: tensor of shape [batchsize, channels, N1, N2]
    """
    N1, N2 = x.size(-2), x.size(-1)  # Dimensions along each axis

    # Compute the 2D Fourier transform of the input
    x_ft = torch.fft.fft2(x, s=(N1, N2))

    # Create the Hilbert transform multipliers along each axis
    freqs1 = torch.fft.fftfreq(N1).to(x.device)  # Frequency components along axis 1 (rows)
    freqs2 = torch.fft.fftfreq(N2).to(x.device)  # Frequency components along axis 2 (columns)

    H1 = -1j * torch.sign(freqs1)  # Hilbert multiplier along axis 1
    H2 = -1j * torch.sign(freqs2)  # Hilbert multiplier along axis 2

    # Reshape for broadcasting
    H1 = H1.reshape(1, 1, N1, 1)
    H2 = H2.reshape(1, 1, 1, N2)

    # Apply the Hilbert transform along each axis separately
    x_ht_ft_1 = x_ft * H1  # Hilbert Transform along axis 1
    x_ht_ft_2 = x_ft * H2  # Hilbert Transform along axis 2

    # Combine the transformed components
    x_ht_ft = x_ht_ft_1 + x_ht_ft_2

    # Return to the spatial domain
    x_ht = torch.fft.ifft2(x_ht_ft, s=(N1, N2)).real  # Take the real part
    return x_ht

# Set up the grid
N1, N2 = 128, 128  # Number of points along each axis
L1, L2 = 2 * np.pi, 2 * np.pi  # Periods along each axis
x = np.linspace(0, L1, N1, endpoint=False)
y = np.linspace(0, L2, N2, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Define frequencies
omega_x = 5
omega_y = 3

# ===========================
# Test Case 1: sin(ωx x) * sin(ωy y)
# ===========================

# Create the test function
f_sin_sin = np.sin(omega_x * X) * np.sin(omega_y * Y)

# Since the analytical Hilbert transform in 2D is complex, we will compare with SciPy's hilbert2 function
# Compute Hilbert transform using PyTorch
f_sin_sin_torch = torch.from_numpy(f_sin_sin).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N1, N2]
f_sin_sin_ht_torch = hilbert_transform_2d_torch(f_sin_sin_torch)
f_sin_sin_ht_torch_np = f_sin_sin_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_sin_sin_hilbert_scipy = hilbert2(f_sin_sin)
f_sin_sin_ht_scipy = np.imag(f_sin_sin_hilbert_scipy)

# Compute relative error between PyTorch and SciPy results
error_sin_sin = np.linalg.norm(f_sin_sin_ht_torch_np - f_sin_sin_ht_scipy) / np.linalg.norm(f_sin_sin_ht_scipy)
print(f"Relative error between PyTorch and SciPy Hilbert transform (sin(ωx x) * sin(ωy y)): {error_sin_sin}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(f_sin_sin_ht_torch_np, extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(f_sin_sin_ht_scipy, extent=(0, L2, 0, L1))
plt.title('SciPy Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(f_sin_sin_ht_torch_np - f_sin_sin_ht_scipy, extent=(0, L2, 0, L1))
plt.title('Difference')
plt.colorbar()
plt.suptitle('Hilbert Transform of sin(ωx x) * sin(ωy y)')
plt.show()

# ===========================
# Test Case 2: cos(ωx x) * cos(ωy y)
# ===========================

# Create the test function
f_cos_cos = np.cos(omega_x * X) * np.cos(omega_y * Y)

# Compute Hilbert transform using PyTorch
f_cos_cos_torch = torch.from_numpy(f_cos_cos).float().unsqueeze(0).unsqueeze(0)
f_cos_cos_ht_torch = hilbert_transform_2d_torch(f_cos_cos_torch)
f_cos_cos_ht_torch_np = f_cos_cos_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_cos_cos_hilbert_scipy = hilbert2(f_cos_cos)
f_cos_cos_ht_scipy = np.imag(f_cos_cos_hilbert_scipy)

# Compute relative error
error_cos_cos = np.linalg.norm(f_cos_cos_ht_torch_np - f_cos_cos_ht_scipy) / np.linalg.norm(f_cos_cos_ht_scipy)
print(f"Relative error between PyTorch and SciPy Hilbert transform (cos(ωx x) * cos(ωy y)): {error_cos_cos}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(f_cos_cos_ht_torch_np, extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(f_cos_cos_ht_scipy, extent=(0, L2, 0, L1))
plt.title('SciPy Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(f_cos_cos_ht_torch_np - f_cos_cos_ht_scipy, extent=(0, L2, 0, L1))
plt.title('Difference')
plt.colorbar()
plt.suptitle('Hilbert Transform of cos(ωx x) * cos(ωy y)')
plt.show()

# ===========================
# Test Case 3: sin(ωx x + ωy y)
# ===========================

# Create the test function
f_sin_sum = np.sin(omega_x * X + omega_y * Y)

# Compute Hilbert transform using PyTorch
f_sin_sum_torch = torch.from_numpy(f_sin_sum).float().unsqueeze(0).unsqueeze(0)
f_sin_sum_ht_torch = hilbert_transform_2d_torch(f_sin_sum_torch)
f_sin_sum_ht_torch_np = f_sin_sum_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_sin_sum_hilbert_scipy = hilbert2(f_sin_sum)
f_sin_sum_ht_scipy = np.imag(f_sin_sum_hilbert_scipy)

# Compute relative error
error_sin_sum = np.linalg.norm(f_sin_sum_ht_torch_np - f_sin_sum_ht_scipy) / np.linalg.norm(f_sin_sum_ht_scipy)
print(f"Relative error between PyTorch and SciPy Hilbert transform (sin(ωx x + ωy y)): {error_sin_sum}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(f_sin_sum_ht_torch_np, extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(f_sin_sum_ht_scipy, extent=(0, L2, 0, L1))
plt.title('SciPy Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(f_sin_sum_ht_torch_np - f_sin_sum_ht_scipy, extent=(0, L2, 0, L1))
plt.title('Difference')
plt.colorbar()
plt.suptitle('Hilbert Transform of sin(ωx x + ωy y)')
plt.show()

# ===========================
# Test Case 4: Gaussian Function
# ===========================

# Create the test function
sigma = 0.5
f_gaussian = np.exp(-((X - L1/2)**2 + (Y - L2/2)**2) / (2 * sigma**2))

# Compute Hilbert transform using PyTorch
f_gaussian_torch = torch.from_numpy(f_gaussian).float().unsqueeze(0).unsqueeze(0)
f_gaussian_ht_torch = hilbert_transform_2d_torch(f_gaussian_torch)
f_gaussian_ht_torch_np = f_gaussian_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_gaussian_hilbert_scipy = hilbert2(f_gaussian)
f_gaussian_ht_scipy = np.imag(f_gaussian_hilbert_scipy)

# Compute relative error
error_gaussian = np.linalg.norm(f_gaussian_ht_torch_np - f_gaussian_ht_scipy) / np.linalg.norm(f_gaussian_ht_scipy)
print(f"Relative error between PyTorch and SciPy Hilbert transform (Gaussian): {error_gaussian}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(f_gaussian_ht_torch_np, extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(f_gaussian_ht_scipy, extent=(0, L2, 0, L1))
plt.title('SciPy Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(f_gaussian_ht_torch_np - f_gaussian_ht_scipy, extent=(0, L2, 0, L1))
plt.title('Difference')
plt.colorbar()
plt.suptitle('Hilbert Transform of Gaussian Function')
plt.show()

# ===========================
# Test Case 5: Exponential Function e^{i(ωx x + ωy y)}
# ===========================

# Create the test function
f_exp = np.exp(1j * (omega_x * X + omega_y * Y))

# Compute Hilbert transform using PyTorch (real and imaginary parts separately)
f_exp_real_torch = torch.from_numpy(np.real(f_exp)).float().unsqueeze(0).unsqueeze(0)
f_exp_imag_torch = torch.from_numpy(np.imag(f_exp)).float().unsqueeze(0).unsqueeze(0)

f_exp_ht_real_torch = hilbert_transform_2d_torch(f_exp_real_torch)
f_exp_ht_imag_torch = hilbert_transform_2d_torch(f_exp_imag_torch)


scipy_hilbert2 = hilbert2(f_exp)
scipy_hilbert2_imag = np.imag(scipy_hilbert2)

# Combine real and imaginary parts
f_exp_ht_torch = f_exp_ht_real_torch.detach().numpy().squeeze() + 1j * f_exp_ht_imag_torch.detach().numpy().squeeze()

# Analytical Hilbert transform: e^{i(ωx x + ωy y - π/2)} for ω > 0
f_exp_ht_analytical = np.exp(1j * (omega_x * X + omega_y * Y - np.pi / 2))

# Compute relative error
error_exp = np.linalg.norm(f_exp_ht_torch - f_exp_ht_analytical) / np.linalg.norm(f_exp_ht_analytical)
print(f"Relative error between PyTorch Hilbert transform and analytical result (exp): {error_exp}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.real(f_exp_ht_torch), extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform (Real)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.real(f_exp_ht_analytical), extent=(0, L2, 0, L1))
plt.title('Analytical Hilbert Transform (Real)')
plt.colorbar()
plt.suptitle('Hilbert Transform of e^{i(ωx x + ωy y)} - Real Part')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.imag(f_exp_ht_torch), extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform (Imag)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.imag(f_exp_ht_analytical), extent=(0, L2, 0, L1))
plt.title('Analytical Hilbert Transform (Imag)')
plt.colorbar()
plt.suptitle('Hilbert Transform of e^{i(ωx x + ωy y)} - Imaginary Part')
plt.show()

# ===========================
# Test Case 6: Sinc Function sin(r)/r
# ===========================

# Compute radial distance from center
r = np.sqrt((X - L1/2)**2 + (Y - L2/2)**2) + 1e-8  # Add small value to avoid division by zero

# Create the test function
f_sinc = np.sin(r) / r

# Compute Hilbert transform using PyTorch
f_sinc_torch = torch.from_numpy(f_sinc).float().unsqueeze(0).unsqueeze(0)
f_sinc_ht_torch = hilbert_transform_2d_torch(f_sinc_torch)
f_sinc_ht_torch_np = f_sinc_ht_torch.detach().numpy().squeeze()

# Compute Hilbert transform using SciPy
f_sinc_hilbert_scipy = hilbert2(f_sinc)
f_sinc_ht_scipy = np.imag(f_sinc_hilbert_scipy)

# Compute relative error
error_sinc = np.linalg.norm(f_sinc_ht_torch_np - f_sinc_ht_scipy) / np.linalg.norm(f_sinc_ht_scipy)
print(f"Relative error between PyTorch and SciPy Hilbert transform (sinc function): {error_sinc}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(f_sinc_ht_torch_np, extent=(0, L2, 0, L1))
plt.title('PyTorch Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(f_sinc_ht_scipy, extent=(0, L2, 0, L1))
plt.title('SciPy Hilbert Transform')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(f_sinc_ht_torch_np - f_sinc_ht_scipy, extent=(0, L2, 0, L1))
plt.title('Difference')
plt.colorbar()
plt.suptitle('Hilbert Transform of sinc(r)')
plt.show()
