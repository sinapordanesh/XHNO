
""" IDEAS

    -------------------------------------------
    Idea 0: Follows the idea of FNO and LNO, that an integral transforming time domain, is calculated by hilbert and inverse hilbert.
    
    x_ht = hilbert(x).imag
    x_ht_ft = torch.fft.rfft(x_ht) # (fft is another option here)
    out_ht_ft[:, :, :self.modes1] = compl_mul1d(x_ht_ft[:, :, :self.modes1], self.weights1)
    
    out_ht = torch.fft.irfft(out_ht_ft, n=x.size(-1))
    x = -hilbert(out_ht).imag
    
    return x
    
    -------------------------------------------
    Idea 1: transform input to its hilbert transform multiply by weight, then inverse of the hilbert
    
    x_ht = hilbert(x).imag
    out_ht = compl_mul1d(x_ht, weights)
    
    return -hilbert(out_ht).imag
    
    -------------------------------------------
    Idea 2: Transform input to its analytic signal multiply analytic signal by a weight then inverse of analytic signal transformation.
    
    x_ht_analytic = hilbert(x) # this function calculates analytic signals: x+iH(x)
    
    out_ht_analytic = compl_mul1d(x_ht_analytic, weights)
    return out_ht_analytic.real
    
    -------------------------------------------
    Idea 3: Fourier of Hilbert transform of the signal, take modes, multiply with weights and inverse fourier on the result. 
    
    x_ht = hilbert(x).imag
    x_ft_ht = torch.fft.rfft(x_ht)

    # Multiply relevant Fourier modes
    out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
    out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

    #Return to physical space
    x = torch.fft.irfft(out_ft, n=x.size(-1))
    return x
    
    -------------------------------------------
    Idea 4: Fourier of Analytic signal of the signal, take modes, multiply with weights and inverse fourier on the result.
    
    x_ht = hilbert(x)
    x_ft_ht = torch.fft.fft(x_ht)

    # Multiply relevant Fourier modes
    out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
    out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

    #Return to physical space
    x = torch.fft.ifft(out_ft, n=x.size(-1))
    return x.real
    
    # we need to make inverse of hilber
"""

from hilbert import hilbert, hilbert2
import torch

""" ----------------- HILBERT 1D ----------------- """
def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)
    
def hilbert_kernel_1d_1(self, x):
    batchsize = x.shape[0]
    
    # Compute the Analyctical Signal of the input x, then compute the Fourier coeffcients
    x_ht = hilbert(x).imag
    x_ht_ft = torch.fft.rfft(x_ht)
    
    # Multiply relevant Fourier modes
    out_ht_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
    out_ht_ft[:, :, :self.modes1] = compl_mul1d(x_ht_ft[:, :, :self.modes1], self.weights1)
    
    # Return to time domain space
    out_ht = torch.fft.irfft(out_ht_ft, n=x.size(-1))
    # Return to physical space
    x = -hilbert(out_ht).imag
    
    return x

""" ----------------- HILBERT 2D ----------------- """
# Complex multiplication
def compl_mul2d(self, input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", input, weights)

# --- IDEA 0 ---
def hilbert_kernel_1d_2(self, x):
    batchsize = x.shape[0]

    # Compute the Analyctical Signal of the input x, then compute the Fourier coeffcients
    x_ht = hilbert2(x).imag
    x_ht_ft = torch.fft.rfft2(x_ht)

    # Multiply relevant modes
    out_ht_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
    out_ht_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ht_ft[:, :, :self.modes1, :self.modes2], self.weights1)
    out_ht_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ht_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
    
    # Return to physical space
    out_ht = torch.fft.irfft2(out_ht_ft, s=(x.size(-2), x.size(-1)))
    x = -hilbert2(out_ht).imag

    return x