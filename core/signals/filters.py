import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func): return func

from core.utils.aspects import log_dsp_action

@log_dsp_action
@njit
def moving_average_recursive(x, M):
    """
    Рекурсивное среднее. Оптимизировано: убрано деление из цикла.
    """
    N = len(x)
    y = np.zeros(N, dtype=np.float64)
    if M <= 0 or M > N: return y
    
    current_sum = 0.0
    for i in range(M):
        current_sum += x[i]
    
    inv_M = 1.0 / M
    y[M-1] = current_sum * inv_M
    
    for n in range(M, N):
        current_sum += (x[n] - x[n-M])
        y[n] = current_sum * inv_M
    return y

@log_dsp_action
@njit
def fir_manual_filter(x, h):
    """
    КИХ-фильтрация (свертка). Оптимизирована: цикл разбит на 2 части, чтобы убрать 'if'.
    """
    N, M = len(x), len(h)
    y = np.zeros(N, dtype=np.float64)
    
    # Обработка начала (пока фильтр не зашел целиком)
    for i in range(min(M, N)):
        s = 0.0
        for j in range(i + 1):
            s += x[i - j] * h[j]
        y[i] = s
    
    # Основной быстрый цикл
    if N > M:
        for i in range(M, N):
            s = 0.0
            for j in range(M):
                s += x[i - j] * h[j]
            y[i] = s
    return y

@log_dsp_action
def fir_window_design(f_low, f_high, M, sr=8000, window_type='blackman'):
    """ Расчет коэффициентов (оставляем в обычном Python, тут скорость не важна) """
    w_l = 2 * np.pi * f_low / sr
    w_h = 2 * np.pi * f_high / sr
    n = np.arange(M)
    center = (M - 1) / 2.0
    
    def sinc_ideal(w, n, mid):
        res = np.zeros(len(n), dtype=np.float64)
        for i in range(len(n)):
            diff = n[i] - mid
            if abs(diff) < 1e-9:
                res[i] = w / np.pi
            else:
                res[i] = np.sin(w * diff) / (np.pi * diff)
        return res

    h_d = sinc_ideal(w_h, n, center) - sinc_ideal(w_l, n, center)
    
    if window_type == 'blackman':
        w_n = 0.42 - 0.5 * np.cos(2 * np.pi * n / (M - 1)) + 0.08 * np.cos(4 * np.pi * n / (M - 1))
    elif window_type == 'hamming':
        w_n = 0.54 - 0.46 * np.cos(2 * np.pi * n / (M - 1))
    else:
        w_n = np.ones(M)
        
    return h_d * w_n

@log_dsp_action
def iir_design(params, sr=8000):
    """ Расчет коэффициентов БИХ """
    m_type = params.get('type')
    b, a = np.zeros(3), np.ones(3)
    if m_type == 'bandpass':
        R = 1 - 3 * (params['bw'] / sr)
        K = np.cos(2 * np.pi * params['f0'] / sr)
        b[0] = 1 - R
        b[2] = -(1 - R)
        a[1] = -2 * R * K
        a[2] = R**2
    return b, a

@log_dsp_action
@njit
def apply_iir(x, b, a):
    """ Применение БИХ через разностное уравнение """
    N = len(x)
    y = np.zeros(N, dtype=np.float64)
    
    # Первые два отсчета отдельно
    if N > 0:
        y[0] = b[0] * x[0]
    if N > 1:
        y[1] = b[0] * x[1] + b[1] * x[0] - a[1] * y[0]
        
    # Основной цикл
    for n in range(2, N):
        y[n] = b[0] * x[n] + b[1] * x[n-1] + b[2] * x[n-2] - a[1] * y[n-1] - a[2] * y[n-2]

    # Нормировка
    sum_b = b[0] + b[1] + b[2]
    sum_a = 1.0 + a[1] + a[2]
    if abs(sum_b) > 1e-5:
        gain = sum_b / (sum_a + 1e-12)
        if abs(gain) > 1e-6 and abs(gain - 1.0) > 1e-3:
            return y / gain
    return y

def iir_bandpass(f0, bw, sr=8000):
    return iir_design({'type': 'bandpass', 'f0': f0, 'bw': bw}, sr)
