import numpy as np
from core.dsp.fourier import fft, ifft

def linear_convolution(x, h):
    N, M = len(x), len(h)
    y = np.zeros(N + M - 1)
    for i in range(N):
        for j in range(M):
            y[i + j] += x[i] * h[j]
    return y

def fft_convolution(x, h):
    N = len(x) + len(h) - 1
    size = 1 << (N - 1).bit_length()
    X = fft(np.pad(x, (0, size - len(x))))
    H = fft(np.pad(h, (0, size - len(h))))
    return ifft(X * H).real[:N]

def correlation(x, y):
    N, M = len(x), len(y)
    res = np.zeros(N + M - 1)
    for n in range(-(M - 1), N):
        val = 0
        for m in range(M):
            if 0 <= n + m < N:
                val += x[n + m] * y[m]
        res[n + M - 1] = val
    return res

def fft_correlation(x, y):
    N = len(x) + len(y) - 1
    size = 1 << (N - 1).bit_length()
    X = fft(np.pad(x, (0, size - len(x))))
    Y = fft(np.pad(np.flip(y), (0, size - len(y))))
    res = ifft(X * Y).real[:N]
    return res
