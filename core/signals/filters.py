import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(func): return func

from core.utils.aspects import log_dsp_action

@log_dsp_action
@njit
def moving_average_recursive(x, M):
    """ Однородный рекурсивный фильтр (стр. 38, формула 34). """
    y = np.zeros_like(x)
    current_sum = 0.0
    for i in range(M):
        current_sum += x[i]

    y[M-1] = current_sum / M
    for n in range(M, len(x)):
        current_sum += (x[n] - x[n-M])
        y[n] = current_sum / M
    return y

@log_dsp_action
@njit
def fir_manual_filter(x, h):
    """ Программная реализация КИХ-фильтрации (стр. 39, формула 35). """
    N, M = len(x), len(h)
    y = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(M):
            if i - j >= 0:
                s += x[i - j] * h[j]
        y[i] = s
    return y

@log_dsp_action
def fir_window_design(f_low, f_high, M, sr=8000, window_type='blackman'):
    """ Проектирование КИХ-фильтра методом взвешивания (стр. 40). """
    w_l = 2 * np.pi * f_low / sr
    w_h = 2 * np.pi * f_high / sr
    n = np.arange(M)
    center = (M - 1) / 2

    def sinc_ideal(w, n, mid):
        res = np.zeros_like(n, dtype=float)
        idx = (n != mid)
        res[idx] = np.sin(w * (n[idx] - mid)) / (np.pi * (n[idx] - mid))
        res[~idx] = w / np.pi
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
    """ Расчет коэффициентов БИХ (стр. 41-43). """
    m_type = params.get('type')
    b, a = np.zeros(3), np.ones(3)
    if m_type == 'lpf':
        x_val = np.exp(-2 * np.pi * params['fc'] / sr)
        b[0] = 1 - x_val; a[1] = -x_val; a[2] = 0
    elif m_type == 'hpf':
        x_val = np.exp(-2 * np.pi * params['fc'] / sr)
        b[0] = (1+x_val)/2; b[1] = -(1+x_val)/2; a[1] = -x_val; a[2] = 0
    elif m_type == 'bandpass':
        R = 1 - 3 * (params['bw'] / sr)
        K = np.cos(2 * np.pi * params['f0'] / sr)
        b[0] = 1 - R; b[2] = -(1 - R)
        a[1] = -2 * R * K; a[2] = R**2
    elif m_type == 'reject':
        R = 1 - 3 * (params['bw'] / sr)
        K = np.cos(2 * np.pi * params['f0'] / sr)
        b[0] = 1; b[1] = -2*K; b[2] = 1
        a[1] = -2*R*K; a[2] = R**2
    return b, a

@log_dsp_action
@njit
def apply_iir(x, b, a):
    """ Реализация БИХ-фильтра через разностное уравнение (стр. 41). """
    y = np.zeros_like(x)
    for n in range(len(x)):
        out = b[0] * x[n]
        if n >= 1:
            out += b[1] * x[n-1]
            out -= a[1] * y[n-1]
        if n >= 2:
            out += b[2] * x[n-2]
            out -= a[2] * y[n-2]
        y[n] = out

    # Нормировка усиления (Step 7, стр. 48)
    sum_b = np.sum(b)
    sum_a = np.sum(a)
    if abs(sum_b) > 1e-5:
        gain = sum_b / (sum_a + 1e-12)
        if abs(gain) > 1e-6 and abs(gain - 1.0) > 1e-3:
            return y / gain
    return y

def iir_bandpass(f0, bw, sr=8000):
    return iir_design({'type': 'bandpass', 'f0': f0, 'bw': bw}, sr)
