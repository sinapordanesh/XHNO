import numpy as np
from scipy.fft import fft, ifft
import scipy.io as sio
from scipy.ndimage import gaussian_filter

# Constants and Parameters
hbar = 1.0      # Reduced Planck constant (set to 1 for simplicity)
m = 1.0         # Particle mass (set to 1 for simplicity)
dx = 0.1        # Spatial step size
dt = 0.001      # Temporal step size
N_x = 2048      # Number of spatial points
N_t = 8192      # Number of time points
L = dx * N_x    # Length of the spatial domain

# Spatial and Temporal Grids
x = np.linspace(-L/2, L/2, N_x)
t = np.linspace(0, dt * (N_t - 1), N_t)

# Potential Function V(x)
V = np.zeros(N_x)  # Free particle; modify if you want a different potential

# Initial Wave Function: Gaussian Wave Packet
x0 = 0.0       # Initial position
k0 = 5.0       # Initial momentum
sigma = 1.0    # Width of the packet
psi0 = (1/(sigma * np.sqrt(np.pi)))**0.5 * \
       np.exp(- (x - x0)**2 / (2 * sigma**2)) * \
       np.exp(1j * k0 * x)

# Wavenumber Grid for FFT
k = np.fft.fftfreq(N_x, d=dx) * 2 * np.pi

# Precompute Exponential Operators
expV = np.exp(-1j * V * dt / hbar)
expK = np.exp(-1j * (hbar * k**2) / (2 * m) * dt)

# Initialize Wave Function Array
psi = np.zeros((N_t, N_x), dtype=complex)
psi[0, :] = psi0

# Time Evolution Using Split-Step Fourier Method
for n in range(1, N_t):
    # Potential Step
    psiV = expV * psi[n - 1, :]
    # Kinetic Step in Fourier Space
    psiK = fft(psiV)
    psiK = expK * psiK
    psi[n, :] = ifft(psiK)

# Extract Real and Imaginary Parts
psi_real = np.real(psi)             # Shape: (8192, 2048)
psi_imag = np.imag(psi)             # Shape: (8192, 2048)

# Transpose to Match Desired Shape (2048, 8192)
psi_real = psi_real.T               # Shape: (2048, 8192)
psi_imag = psi_imag.T               # Shape: (2048, 8192)

# Calculate Spatial Derivatives
psi_x_real = np.gradient(psi_real, dx, axis=0)  # Shape: (2048, 8192)
psi_x_imag = np.gradient(psi_imag, dx, axis=0)  # Shape: (2048, 8192)
psi_x = psi_x_real + 1j * psi_x_imag            # Complex derivative

# Magnitude of the Spatial Derivative
psi_x_mag = np.abs(psi_x)                       # Shape: (2048, 8192)

# Remove Last Time Point to Match Shape (2048, 8191)
psi_x_mag = psi_x_mag[:, :-1]                   # Shape: (2048, 8191)

# Smooth Versions Using Gaussian Filter
psi_abs = np.abs(psi_real + 1j * psi_imag)
psi_smooth = gaussian_filter(psi_abs, sigma=1)  # Shape: (2048, 8192)
psi_smooth_x = np.gradient(psi_smooth, dx, axis=0)
psi_smooth_x = psi_smooth_x[:, :-1]             # Shape: (2048, 8191)

# Prepare Data Dictionary
data = {
    'a': psi_real,                # Shape: (2048, 8192)
    'a_smooth': psi_smooth,       # Shape: (2048, 8192)
    'a_smooth_x': psi_smooth_x,   # Shape: (2048, 8191)
    'a_x': psi_x_mag,             # Shape: (2048, 8191)
    'u': psi_imag                 # Shape: (2048, 8192)
}

# Save to .mat File
sio.savemat('schrodinger_data.mat', data)
