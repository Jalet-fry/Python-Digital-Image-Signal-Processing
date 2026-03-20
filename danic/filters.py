from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from config import FilterConfig, SignalConfig

def design_recursive_ma(cfg_filter: FilterConfig):
    """
    Рекурсивный однородный фильтр
    y(n) = y(n-1) + (1/M) * (x(n) - x(n-M))
    """
    m = cfg_filter.m_ma
    b = np.zeros(m + 1)
    b[0] = 1.0 / m
    b[m] = -1.0 / m
    a = [1.0, -1.0]
    return b, a

def design_fir_blackman_lf(cfg_filter: FilterConfig, cfg_signal: SignalConfig):
    """
    КИХ-фильтр НЧ (низкочастотный) методом взвешивания.
    Использует идеальную импульсную характеристику
    и окно Блэкмана
    """
    M = cfg_filter.m_fir
    fs = cfg_signal.fs
    fc = cfg_filter.f_cutoff_fir

    f_c_norm = fc / fs
    w_c = 2*np.pi*f_c_norm

    n = np.arange(M)
    alpha = (M - 1) / 2
    n_rel = n - alpha

    h_d = np.zeros(M)
    for i in range(M):
        curr_n = n_rel[i]
        if curr_n == 0:
            h_d[i] = 2*f_c_norm
        else:
            h_d[i] = 2*f_c_norm * (np.sin(curr_n*w_c))/(curr_n *w_c)

    w = 0.42 - 0.5*np.cos(2*np.pi*n / (M-1)) + 0.08 * np.cos(4*np.pi*n / (M-1))
    # taps = sp_signal.firwin(
    #     numtaps=cfg_filter.m_fir,
    #     cutoff=cfg_filter.f_cutoff_fir,
    #     window='blackman',
    #     pass_zero=True,
    #     fs=cfg_signal.fs
    # )
    taps = h_d * w
    return taps / np.sum(taps)

def design_iir_one_poly_hf(cfg_filter: FilterConfig,
                   cfg_signal: SignalConfig):
    """
    Однополюсный рекурсивный БИХ ВЧ-фильтр
    """
    fc_norm = cfg_filter.f_cutoff_iir / cfg_signal.fs
    
    # a = exp(-2 * pi * fc)

    alpha = np.exp(-2 * np.pi * fc_norm)
    
    # a0 = (1 + x) / 2
    # a1 = -(1 + x) / 2
    # b1 = x

    a = [(1 + alpha) / 2, -(1 + alpha) / 2]
    b = [1.0, -alpha]
    
    return a, b, alpha

def apply_ma(signal: np.ndarray, cfg_filter: FilterConfig):
    # b, a = design_recursive_ma(cfg_filter)
    # y = sp_signal.lfilter(b, a, signal)
    # delay = int((cfg_filter.m_ma - 1) // 2)
    # y_shifted = np.zeros_like(y)
    # y_shifted[:-delay] = y[delay:]

    N = len(signal)
    y = np.zeros(N)
    M = cfg_filter.m_ma
    scale = 1.0 / M
    current_sum = 0.0

    for n in range(N):
        add_val = signal[n]
        rem_val = signal[n - M] if n>=M else 0.0

        current_sum +=add_val - rem_val
        y[n] = current_sum * scale

    delay = int((M - 1) // 2)
    y_shifted = np.zeros_like(y)
    y_shifted[:-delay] = y[delay:]
    return y_shifted

def apply_fir(signal: np.ndarray, cfg_filter: FilterConfig, cfg_signal: SignalConfig):
    taps = design_fir_blackman_lf(cfg_filter, cfg_signal)
    y = np.convolve(signal, taps, mode='same') #  <--  !!!свёртка тута!!!
    return y

def apply_iir(signal: np.ndarray, cfg_filter: FilterConfig, cfg_signal: SignalConfig):
    a_iir, b_iir, _ = design_iir_one_poly_hf(cfg_filter, cfg_signal)
    N = len(signal)
    y = np.zeros(N)
    for n in range(1,N):
        y[n] = a_iir[0]*signal[n]+a_iir[1]*signal[n-1] - b_iir[1]*y[n-1]
    #return sp_signal.lfilter(b_iir, a_iir, signal)
    return y

def freqresp_moving_average(cfg_filter: FilterConfig, cfg_signal: SignalConfig):
    b, a = design_recursive_ma(cfg_filter)
    w, h = sp_signal.freqz(b, a, worN=8192, fs=cfg_signal.fs)
    return w, h

def freqresp_fir(taps: np.ndarray, cfg_signal: SignalConfig):
    w, h = sp_signal.freqz(taps, 1.0, worN=8192, fs=cfg_signal.fs)
    return w, h

def freqresp_iir(b: list, a: list, cfg_signal: SignalConfig):
    w, h = sp_signal.freqz(b, a, worN=8192, fs=cfg_signal.fs)
    return w, h

def print_ma_info(cfg: FilterConfig) -> None:
    print(f"       Скользящее среднее: M = {cfg.m_ma}")
    print(f"       h[k] = 1/{cfg.m_ma} = {1 / cfg.m_ma:.6f}")

def print_fir_coefficients(h: np.ndarray, n: int = 10) -> None:
    print(f"\nПервые {n} коэффициентов КИХ-фильтра h[k]:")
    for i in range(min(len(h), n)):
        print(f"   h[{i:3d}] = {h[i]: .8f}")

def print_iir_params(B: list, A: list, alpha: float, cfg: FilterConfig) -> None:
    print(f"\nБИХ-фильтр (ВЧ), fc = {cfg.f_cutoff_iir} Гц:")
    print(f"   alpha = {alpha:.8f}")
    print(f"   Уравнение: y[n] = {B[0]:.4f}*x[n] + {B[1]:.4f}*x[n-1] + {(-A[1]):.4f}*y[n-1]")