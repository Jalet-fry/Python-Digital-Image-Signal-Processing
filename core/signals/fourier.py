import numpy as np

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Формула Дискретного Преобразования Фурье (ДПФ)
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def idft(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Формула Обратного ДПФ (ОДПФ)
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, X) / N

def fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N <= 1: return x
    if N % 2 > 0:
        raise ValueError("Размер должен быть степенью двойки")
        
    # Алгоритм Кули-Тьюки (векторизованный)
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    
    # Коэффициенты поворота
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + T, even - T])

def ifft(X):
    # ОБПФ через сопряжение
    return np.conjugate(fft(np.conjugate(X))) / len(X)
