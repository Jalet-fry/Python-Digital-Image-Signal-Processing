import numpy as np
from core.signals.fourier import fft, ifft

def linear_convolution(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    z = np.zeros(Nz)
    for n in range(Nz):
        for k in range(Nx):
            if 0 <= n - k < Ny:
                z[n] += x[k] * y[n - k]
    return z

def fft_convolution(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    # БПФ требует длину - степень двойки
    n_fft = 1 << (Nz - 1).bit_length()
    
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    # Теорема о свертке: z = ifft(X * Y)
    result = ifft(X * Y)
    return result.real[:Nz]

def correlation(x, y):
    # Прямое вычисление корреляции (как в np.correlate(mode='full'))
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    res = np.zeros(Nz)
    # Сдвигаем y относительно x от -(Ny-1) до (Nx-1)
    for lag in range(-(Ny - 1), Nx):
        s = 0
        for i in range(Nx):
            j = i - lag
            if 0 <= j < Ny:
                s += x[i] * y[j]
        res[lag + (Ny - 1)] = s
    return res

def fft_correlation(x, y):
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length()
    
    # Теорема о корреляции: Rxy = ifft( fft(x) * conj(fft(y)) )
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    raw_corr = ifft(X * np.conjugate(Y))
    
    # Переупорядочиваем результат, чтобы lag 0 был в центре (как в библиотеках)
    # В результате IFFT: [0, 1, ..., Nx-1, ..., 0, ..., -Ny+1]
    # Нам нужно состыковать их: [-(Ny-1), ..., -1, 0, 1, ..., Nx-1]
    
    # Отрицательные лаги находятся в конце массива raw_corr
    pos_lags = raw_corr[:Nx]
    neg_lags = raw_corr[n_fft - (Ny - 1):]
    
    return np.concatenate([neg_lags, pos_lags]).real
