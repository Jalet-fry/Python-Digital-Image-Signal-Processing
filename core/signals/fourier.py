import numpy as np

def dft(x: np.ndarray) -> np.ndarray:
    """Дискретное преобразование Фурье (медленное)."""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def idft(X: np.ndarray) -> np.ndarray:
    """Обратное дискретное преобразование Фурье."""
    N = len(X)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return (np.dot(e, X) / N).real

def fft_recursive(x: np.ndarray, mode: int = 1) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return x
    
    hn = n // 2
    # Для корректной рекурсии n должно быть четным
    if n % 2 != 0:
        # Если вдруг попало нечетное, просто обрезаем или дополняем (но лучше доводить до степени 2 заранее)
        x = x[:-1]
        n -= 1
        hn = n // 2

    b = np.zeros(hn, dtype=complex)
    c = np.zeros(hn, dtype=complex)
    
    w_n = np.exp(-1j * mode * 2 * np.pi / n)
    w = 1.0 + 0j
    
    for i in range(hn):
        b[i] = x[i] + x[i + hn]
        c[i] = (x[i] - x[i + hn]) * w
        w *= w_n
        
    return np.concatenate([fft_recursive(b, mode), fft_recursive(c, mode)])

def reverse_index(x: np.ndarray) -> np.ndarray:
    n = len(x)
    bits = int(np.log2(n))
    res = np.copy(x)
    for i in range(n):
        j = int('{:0{width}b}'.format(i, width=bits)[::-1], 2)
        if j > i:
            res[i], res[j] = res[j], res[i]
    return res

def fft(signal: np.ndarray) -> np.ndarray:
    """Прямое БПФ с дополнением до степени двойки."""
    n = len(signal)
    # Находим ближайшую степень двойки вверх
    next_pow2 = 1 << (n - 1).bit_length()
    if next_pow2 != n:
        signal = np.pad(signal, (0, next_pow2 - n))
    
    res = fft_recursive(signal.astype(complex), mode=1)
    return reverse_index(res)

def ifft(complex_signal: np.ndarray) -> np.ndarray:
    """Обратное БПФ."""
    n = len(complex_signal)
    res = fft_recursive(complex_signal, mode=-1)
    res = reverse_index(res)
    return (res / n).real
