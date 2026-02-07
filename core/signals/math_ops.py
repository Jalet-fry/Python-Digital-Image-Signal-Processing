import numpy as np
from .fourier import fft, ifft

def circular_convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Круговая свертка."""
    n = len(x)
    z = np.zeros(n)
    for i in range(n):
        for j in range(n):
            z[i] += x[j] * y[(i - j) % n]
    return z

def linear_convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Линейная свертка (ручная реализация)."""
    n = len(x)
    m = len(y)
    z = np.zeros(n + m - 1)
    for i in range(n + m - 1):
        for j in range(n):
            if 0 <= i - j < m:
                z[i] += x[j] * y[i - j]
    return z

def fft_convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Быстрая линейная свертка через БПФ с дополнением нулями."""
    n = len(x)
    m = len(y)
    target_size = n + m - 1
    # Для БПФ нужна длина - степень двойки
    fft_size = 1 << (target_size - 1).bit_length()
    
    x_padded = np.pad(x, (0, fft_size - n))
    y_padded = np.pad(y, (0, fft_size - m))
    
    res = ifft(fft(x_padded) * fft(y_padded))
    return res[:target_size]

def correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Взаимная корреляция (ручная реализация)."""
    n = len(x)
    m = len(y)
    # Результат имеет длину n + m - 1
    z = np.zeros(n + m - 1)
    for k in range(-(m - 1), n):
        for i in range(n):
            j = i - k
            if 0 <= j < m:
                z[k + m - 1] += x[i] * y[j]
    return z

def fft_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Быстрая корреляция через БПФ."""
    n = len(x)
    m = len(y)
    target_size = n + m - 1
    fft_size = 1 << (target_size - 1).bit_length()
    
    x_padded = np.pad(x, (0, fft_size - n))
    y_padded = np.pad(y, (0, fft_size - m))
    
    # Корреляция в частотной области: FFT(x) * conj(FFT(y))
    # Но для соответствия np.correlate(x, y) часто делают обратное
    res = ifft(fft(x_padded) * np.conj(fft(y_padded)))
    # Сдвигаем результат для соответствия стандартному выводу корреляции
    return np.roll(res[:target_size], 0)
