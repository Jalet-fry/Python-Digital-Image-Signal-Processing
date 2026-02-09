import numpy as np

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Формула Дискретного Преобразования Фурье (ДПФ):
    # X[k] = sum(x[n] * exp(-2j * pi * k * n / N))
    e = np.exp(-2j * np.pi * k * n / N)
    
    return np.dot(e, x)

def idft(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Формула Обратного Дискретного Преобразования Фурье (ОДПФ):
    # x[n] = (1/N) * sum(X[k] * exp(2j * pi * k * n / N))
    e = np.exp(2j * np.pi * k * n / N)
    
    return np.dot(e, X) / N

def fft(x):
    N = len(x)
    if N <= 1: return x
        
    # Алгоритм Кули-Тьюки (разделяй и властвуй)
    # 1. Делим на четные и нечетные
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    
    # 2. "Бабочка" и объединение
    # T = exp(-2j*pi*k/N) * odd[k]
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    
    return np.concatenate([even + T, even - T])

def ifft(X):
    # Формула Обратного Быстрого Преобразования Фурье (ОБПФ)
    # ifft(X) = conjugate(fft(conjugate(X))) / N
    return np.conjugate(fft(np.conjugate(X))) / len(X)
