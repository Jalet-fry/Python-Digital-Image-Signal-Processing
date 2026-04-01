import numpy as np

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def idft(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, X) / N

def fft(x):
    """
    Быстрое преобразование Фурье. 
    Автоматически дополняет сигнал нулями до степени 2, если это необходимо.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    # Проверка на степень двойки (Cooley-Tukey требует N = 2^k)
    if N > 1 and (N & (N - 1)) != 0:
        N_next = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, N_next - N))
        N = N_next

    if N <= 1: return x

    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + T, even - T])

def ifft(X):
    N = len(X)
    X_conj = np.conjugate(X)
    x = fft(X_conj)
    return np.conjugate(x) / N
