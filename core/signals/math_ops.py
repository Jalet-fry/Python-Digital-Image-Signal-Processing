import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func): return func

from core.signals.fourier import fft, ifft
from core.utils.aspects import log_dsp_action

@log_dsp_action
@njit
def linear_convolution(x, y):
    """
    Классическая линейная свертка (стр. 27).
    Оптимизирована для Numba: убраны проверки if внутри циклов.
    """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    z = np.zeros(Nz, dtype=np.float64)
    
    for n in range(Nz):
        # Вычисляем границы k заранее, чтобы n-k всегда попадал в массив y
        k_start = max(0, n - Ny + 1)
        k_end = min(Nx, n + 1)
        
        s = 0.0
        for k in range(k_start, k_end):
            s += x[k] * y[n - k]
        z[n] = s
    return z

@log_dsp_action
def fft_convolution(x, y):
    """ Быстрая свертка через БПФ (стр. 30). """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length() 
    
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    result = ifft(X * Y)
    return result.real[:Nz]

@log_dsp_action
@njit
def correlation(x, y):
    """
    Взаимная корреляция (стр. 29). Оптимизирована для Numba.
    """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    res = np.zeros(Nz, dtype=np.float64)
    
    for lag_idx in range(Nz):
        lag = lag_idx - (Ny - 1)
        # i_start и i_end гарантируют, что индексы i и i-lag в пределах массивов
        i_start = max(0, lag)
        i_end = min(Nx, Ny + lag)
        
        s = 0.0
        for i in range(i_start, i_end):
            s += x[i] * y[i - lag]
        res[lag_idx] = s
    return res

@log_dsp_action
def fft_correlation(x, y):
    """ Быстрая корреляция через спектры (стр. 29). """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length()
    
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    raw_corr = ifft(X * np.conjugate(Y))
    
    pos_lags = raw_corr[:Nx]
    neg_lags = raw_corr[n_fft - (Ny - 1):]
    return np.concatenate([neg_lags, pos_lags]).real
