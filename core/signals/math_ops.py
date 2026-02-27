import numpy as np
from core.signals.fourier import fft, ifft
from core.utils.aspects import log_dsp_action

@log_dsp_action
def linear_convolution(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    z = np.zeros(Nz)
    for n in range(Nz):
        for k in range(Nx):
            if 0 <= n - k < Ny:
                z[n] += x[k] * y[n - k]
    return z

@log_dsp_action
def fft_convolution(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length()
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    result = ifft(X * Y)
    return result.real[:Nz]

@log_dsp_action
def correlation(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    res = np.zeros(Nz)
    for lag in range(-(Ny - 1), Nx):
        s = 0
        for i in range(Nx):
            j = i - lag
            if 0 <= j < Ny:
                s += x[i] * y[j]
        res[lag + (Ny - 1)] = s
    return res

@log_dsp_action
def fft_correlation(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length()
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    raw_corr = ifft(X * np.conjugate(Y))
    pos_lags = raw_corr[:Nx]
    neg_lags = raw_corr[n_fft - (Ny - 1):]
    return np.concatenate([neg_lags, pos_lags]).real
