import numpy as np
from core.signals.fourier import fft, ifft
from core.utils.aspects import log_dsp_action

@log_dsp_action
def linear_convolution(x, y):
    """
    Классическая линейная свертка (через циклы).
    Именно так она описана в методичке на стр. 27.
    """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1 # Длина результата всегда N + M - 1
    z = np.zeros(Nz)
    
    # Вложенные циклы: мы "протаскиваем" сигнал Y вдоль X
    for n in range(Nz):
        for k in range(Nx):
            # n - k - это индекс в сигнале Y (сдвинутый)
            if 0 <= n - k < Ny:
                z[n] += x[k] * y[n - k]
    return z

@log_dsp_action
def fft_convolution(x, y):
    """
    БЫСТРАЯ СВЕРТКА (через Теорему о свертке, стр. 30).
    Свертка во времени = Умножение в частотной области.
    """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    
    # 1. PADDING (Дополнение нулями)
    # Находим ближайшую степень двойки для БПФ
    n_fft = 1 << (Nz - 1).bit_length() 
    
    # 2. ПЕРЕХОД В ЧАСТОТЫ
    X = fft(np.pad(x, (0, n_fft - Nx)))
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    # 3. УМНОЖЕНИЕ СПЕКТРОВ
    # Каждая частота умножается на частоту
    result = ifft(X * Y)
    
    # Возвращаем вещественную часть (т.к. мнимая после IFFT близка к 0)
    return result.real[:Nz]

@log_dsp_action
def correlation(x, y):
    """
    Взаимная корреляция (через циклы).
    Показывает степень сходства сигналов при разных сдвигах.
    """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    res = np.zeros(Nz)
    
    # В отличие от свертки, здесь сигнал НЕ инвертируется (j = i - lag)
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
    """
    БЫСТРАЯ КОРРЕЛЯЦИЯ (Теорема корреляции, стр. 29).
    Использует комплексно-сопряженный спектр одного из сигналов.
    """
    Nx, Ny = len(x), len(y)
    Nz = Nx + Ny - 1
    n_fft = 1 << (Nz - 1).bit_length()
    
    X = fft(np.pad(x, (0, n_fft - Nx)))
    # КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ СВЕРТКИ: np.conjugate(Y)
    Y = fft(np.pad(y, (0, n_fft - Ny)))
    
    # Теорема корреляции: Z = X * conj(Y)
    raw_corr = ifft(X * np.conjugate(Y))
    
    # Центрируем результат (сдвигаем отрицательные лаги в начало)
    pos_lags = raw_corr[:Nx]
    neg_lags = raw_corr[n_fft - (Ny - 1):]
    return np.concatenate([neg_lags, pos_lags]).real
