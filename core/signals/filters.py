import numpy as np

def moving_average_recursive(x, M):
    """
    Рекурсивный однородный фильтр (Moving Average).
    y[n] = y[n-1] + (x[n] - x[n-M]) / M
    """
    y = np.zeros_like(x)
    current_sum = np.sum(x[:M])
    y[M-1] = current_sum / M
    
    for n in range(M, len(x)):
        current_sum += x[n] - x[n-M]
        y[n] = current_sum / M
    return y

def fir_window_bandpass(f_low, f_high, M, sr=8000):
    """
    КИХ-фильтр полосовой (Bandpass) методом окон.
    Использует окно Блэкмана (Blackman).
    """
    # Нормированные частоты
    w_low = 2 * np.pi * f_low / sr
    w_high = 2 * np.pi * f_high / sr
    
    n = np.arange(M)
    center = (M - 1) / 2
    
    # Идеальный полосовой фильтр (разность двух ФНЧ)
    # h[n] = sin(w_high*(n-mid)) / (pi*(n-mid)) - sin(w_low*(n-mid)) / (pi*(n-mid))
    def sinc(w, n, mid):
        res = np.zeros_like(n, dtype=float)
        idx = (n != mid)
        res[idx] = np.sin(w * (n[idx] - mid)) / (np.pi * (n[idx] - mid))
        res[~idx] = w / np.pi
        return res

    h = sinc(w_high, n, center) - sinc(w_low, n, center)
    
    # Окно Блэкмана
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (M - 1)) + 0.08 * np.cos(4 * np.pi * n / (M - 1))
    
    return h * window

def iir_bandpass(f0, bw, sr=8000):
    """
    БИХ-фильтр полосовой (Резонансный биквадрат).
    f0 - центральная частота, bw - полоса пропускания.
    Возвращает коэффициенты (a, b) для разностного уравнения.
    """
    omega = 2 * np.pi * f0 / sr
    sn = np.sin(omega)
    cs = np.cos(omega)
    # Коэффициент затухания (Q-factor связан с BW)
    alpha = sn * np.sinh(np.log(2)/2 * bw * omega / sn)
    
    b = [alpha, 0, -alpha]
    a = [1 + alpha, -2 * cs, 1 - alpha]
    
    # Нормировка по a0
    return np.array(b) / a[0], np.array(a) / a[0]

def apply_iir(x, b, a):
    """Применяет БИХ-фильтр через разностное уравнение."""
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] - a[1]*y[n-1] - a[2]*y[n-2]
    return y
