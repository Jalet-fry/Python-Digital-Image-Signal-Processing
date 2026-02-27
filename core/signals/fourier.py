import numpy as np
from core.utils.aspects import log_dsp_action

@log_dsp_action
def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

@log_dsp_action
def idft(X):
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.dot(e, X) / N

@log_dsp_action
def fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + T, even - T])

@log_dsp_action
def ifft(X):
    return np.conjugate(fft(np.conjugate(X))) / len(X)
