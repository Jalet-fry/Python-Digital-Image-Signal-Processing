import numpy as np
from core.utils.aspects import log_dsp_action

@log_dsp_action
def moving_average_recursive(x, M):
    """ Однородный рекурсивный фильтр (стр. 38, формула 34). """
    y = np.zeros_like(x)
    current_sum = np.sum(x[:M])
    y[M-1] = current_sum / M
    for n in range(M, len(x)):
        current_sum += (x[n] - x[n-M])
        y[n] = current_sum / M
    return y

@log_dsp_action
def fir_manual_filter(x, h):
    """ Программная реализация КИХ-фильтрации (стр. 39, формула 35). """
    N, M = len(x), len(h)
    y = np.zeros(N)
    for i in range(M, N):
        y[i] = np.sum(x[i-M+1 : i+1][::-1] * h)
    return y

def fir_window_bandpass(f_low, f_high, M, sr=8000):
    """ Проектирование КИХ методом взвешивания (стр. 40). """
    w_l, w_h = 2 * np.pi * f_low / sr, 2 * np.pi * f_high / sr
    n = np.arange(M); center = (M - 1) / 2
    def sinc(w, n, mid):
        res = np.zeros_like(n, dtype=float)
        idx = (n != mid)
        res[idx] = np.sin(w * (n[idx] - mid)) / (np.pi * (n[idx] - mid))
        res[~idx] = w / np.pi
        return res
    h_d = sinc(w_h, n, center) - sinc(w_l, n, center)
    w_n = 0.42 - 0.5*np.cos(2*np.pi*n/(M-1)) + 0.08*np.cos(4*np.pi*n/(M-1))
    return h_d * w_n

def iir_design(params, sr=8000):
    """ Расчет коэффициентов БИХ (стр. 41-43). b - вход (x), a - выход (y). """
    m_type = params.get('type')
    b, a = np.zeros(3), np.ones(3) # a0 = 1
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
        b[0] = 1; b[1] = -2*K; b[2] = 1; a[1] = -2*R*K; a[2] = R**2
        gain = (1 + a[1] + a[2]) / (b[0] + b[1] + b[2] + 1e-9)
        b *= gain
    return b, a

@log_dsp_action
def apply_iir(x, b, a):
    """
    Разностное уравнение (стр. 41):
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    """
    y = np.zeros_like(x)
    for n in range(2, len(x)):
        y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] - a[1]*y[n-1] - a[2]*y[n-2]
    return y

def iir_bandpass(f0, bw, sr=8000):
    return iir_design({'type': 'bandpass', 'f0': f0, 'bw': bw}, sr)
