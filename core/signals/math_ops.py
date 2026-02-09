import numpy as np
from core.signals.fourier import fft, ifft

def linear_convolution(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    z = np.zeros(Nz)
    
    # Формула Линейной свертки:
    # z[n] = sum(x[k] * y[n-k])
    for n in range(Nz):
        for k in range(Nx):
            if 0 <= n - k < Ny:
                z[n] += x[k] * y[n - k]
    return z

def fft_convolution(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    
    # Дополнение нулями (Padding)
    # n_fft = 2^k >= Nz
    n_fft = 1 << (Nz - 1).bit_length()
    
    # Теорема о свертке:
    # z = ifft( fft(x) * fft(y) )
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    return ifft(X * Y)[:Nz]

def correlation(x, y):
    # Формула Взаимной корреляции (через свертку):
    # Rxy[n] = sum(x[k] * y[k-n]) = x * reverse(y)
    return linear_convolution(x, y[::-1])

def fft_correlation(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length()
    
    # Теорема о корреляции:
    # Rxy = ifft( fft(x) * conjugate(fft(y)) )
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    return ifft(X * np.conjugate(Y))[:Nz]
