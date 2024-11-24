"""
@author: Saman Pordanesh
"""

"""
Hilbert transformation, compatible with the Pytorch tensors. 
The code has a test funciton for comparing the newly implemented Torch Hilbert transform, with Scipy package hilbert transform, calculating errors and plotting resulted transformation. 
"""

import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import os

def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    Parameters
    ----------
    x : Tensor
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Default: x.shape[axis]
    axis : int, optional
        Axis along which to do the transformation. Default: -1.

    Returns
    -------
    x : Tensor
        Analytic signal of x, along the specified axis.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.size(axis)
    if N <= 0:
        raise ValueError("N must be positive.")

    # Compute FFT along the specified axis
    Xf = torch.fft.fft(x, n=N, dim=axis)

    # Construct the filter
    h = torch.zeros(N, dtype=Xf.dtype, device=Xf.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    # Reshape h to broadcast along the correct axis
    shape = [1] * x.ndim
    shape[axis] = N
    h = h.view(shape)

    # Multiply Xf by h
    Xf = Xf * h

    # Compute inverse FFT
    x = torch.fft.ifft(Xf, n=None, dim=axis)

    return x

def hilbert2(x, N=None):
    """
    Compute the 2-D analytic signal of x along axes (0,1)

    Parameters
    ----------
    x : Tensor
        Signal data. Must be at least 2-D and real.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is x.shape[:2]

    Returns
    -------
    x : Tensor
        Analytic signal of x taken along axes (0,1).
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim < 2:
        raise ValueError("x must be at least 2-D.")
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    if N is None:
        N = x.shape[:2]
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or any(n <= 0 for n in N):
        raise ValueError("When given as a tuple, N must hold exactly two positive integers")

    # Compute 2D FFT along axes (0,1)
    Xf = torch.fft.fft2(x, s=N, dim=(0, 1))

    # Construct the filters
    h1 = torch.zeros(N[0], dtype=Xf.dtype, device=Xf.device)
    N0 = N[0]
    if N0 % 2 == 0:
        h1[0] = h1[N0 // 2] = 1
        h1[1:N0 // 2] = 2
    else:
        h1[0] = 1
        h1[1:(N0 + 1) // 2] = 2

    h2 = torch.zeros(N[1], dtype=Xf.dtype, device=Xf.device)
    N1 = N[1]
    if N1 % 2 == 0:
        h2[0] = h2[N1 // 2] = 1
        h2[1:N1 // 2] = 2
    else:
        h2[0] = 1
        h2[1:(N1 + 1) // 2] = 2

    # Construct the 2D filter h
    h = h1[:, None] * h2[None, :]

    # Expand h to match the dimensions of x
    h_shape = list(h.shape) + [1] * (x.ndim - 2)
    h = h.view(h_shape)

    # Multiply Xf by h
    Xf = Xf * h

    # Compute inverse FFT
    x = torch.fft.ifft2(Xf, s=None, dim=(0, 1))

    return x

def run_tests():
    # Create lists to store results
    test_results = []
    plot_index = 1

    # Create a directory for plots
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Define test scenarios
    scenarios = [
        # 1D Scenarios
        {'type': '1D', 'function': np.sin, 'desc': 'Sine Wave'},
        {'type': '1D', 'function': np.cos, 'desc': 'Cosine Wave'},
        {'type': '1D', 'function': lambda t: np.sign(np.sin(t)), 'desc': 'Square Wave'},
        {'type': '1D', 'function': lambda t: np.random.randn(len(t)), 'desc': 'Random Noise'},
        {'type': '1D', 'function': lambda t: np.exp(-t**2), 'desc': 'Gaussian'},
        # 2D Scenarios
        {'type': '2D', 'function': lambda x, y: np.sin(x) + np.cos(y), 'desc': 'Sine + Cosine'},
        {'type': '2D', 'function': lambda x, y: np.exp(-0.1*(x**2 + y**2)), 'desc': '2D Gaussian'},
        {'type': '2D', 'function': lambda x, y: np.sign(np.sin(x) * np.sin(y)), 'desc': '2D Square Wave'},
        {'type': '2D', 'function': lambda x, y: np.random.randn(*x.shape), 'desc': '2D Random Noise'},
        {'type': '2D', 'function': lambda x, y: np.sin(5*x + 5*y), 'desc': 'High Frequency Sine'},
    ]

    for idx, scenario in enumerate(scenarios):
        desc = scenario['desc']
        if scenario['type'] == '1D':
            # Generate test signal
            t = np.linspace(0, 2 * np.pi, 500)
            x_np = scenario['function'](t)

            # Apply PyTorch Hilbert Transform
            x_torch = torch.from_numpy(x_np)
            analytic_torch = hilbert(x_torch)
            hilbert_torch = analytic_torch.imag.numpy()

            # Apply SciPy Hilbert Transform
            analytic_scipy = scipy.signal.hilbert(x_np)
            hilbert_scipy = np.imag(analytic_scipy)

            # Calculate Error
            error = np.abs(hilbert_torch - hilbert_scipy)
            max_error = np.max(error)
            mean_error = np.mean(error)

            # Plot Results
            plt.figure(figsize=(10, 6))
            plt.plot(t, hilbert_scipy, label='SciPy Hilbert', alpha=0.7)
            plt.plot(t, hilbert_torch, '--', label='PyTorch Hilbert', alpha=0.7)
            plt.title(f'1D Hilbert Transform - {desc}')
            plt.xlabel('Time')
            plt.ylabel('Hilbert Transform')
            plt.legend()
            plt.savefig(f'plots/plot_{plot_index}.png')
            plt.close()
            plot_index += 1

            # Store Results
            test_results.append({
                'Scenario': f'1D - {desc}',
                'Max Error': max_error,
                'Mean Error': mean_error
            })

        elif scenario['type'] == '2D':
            # Generate test signal
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            Z_np = scenario['function'](X, Y)

            # Apply PyTorch Hilbert2 Transform
            Z_torch = torch.from_numpy(Z_np)
            analytic_torch = hilbert2(Z_torch)
            hilbert_torch = analytic_torch.imag.numpy()

            # Apply SciPy Hilbert2 Transform
            analytic_scipy = scipy.signal.hilbert2(Z_np)
            hilbert_scipy = np.imag(analytic_scipy)

            # Calculate Error
            error = np.abs(hilbert_torch - hilbert_scipy)
            max_error = np.max(error)
            mean_error = np.mean(error)

            # Plot Results (Display a slice or the central row)
            plt.figure(figsize=(10, 6))
            idx_slice = Z_np.shape[0] // 2
            plt.plot(x, hilbert_scipy[idx_slice, :], label='SciPy Hilbert2', alpha=0.7)
            plt.plot(x, hilbert_torch[idx_slice, :], '--', label='PyTorch Hilbert2', alpha=0.7)
            plt.title(f'2D Hilbert Transform - {desc} (Slice at y=0)')
            plt.xlabel('x')
            plt.ylabel('Hilbert Transform')
            plt.legend()
            plt.savefig(f'plots/plot_{plot_index}.png')
            plt.close()
            plot_index += 1

            # Store Results
            test_results.append({
                'Scenario': f'2D - {desc}',
                'Max Error': max_error,
                'Mean Error': mean_error
            })

    # Generate CSV Report
    df_results = pd.DataFrame(test_results)
    df_results.to_csv('hilbert_transform_test_results.csv', index=False)
    print("Test results saved to 'hilbert_transform_test_results.csv'")

    # Display the DataFrame
    print(df_results)

if __name__ == '__main__':
    run_tests()
