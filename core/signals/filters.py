import numpy as np
from .fourier import fft, ifft

def hamming_window(n: int) -> np.ndarray:
    """
    Создает окно Хемминга заданной длины.
    """
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))

def reject_filter(signal: np.ndarray, window_length: int = 9) -> np.ndarray:
    """
    Режекторный фильтр с использованием окна Хемминга.
    """
    n = len(signal)
    cutoff_frequency = 0.05
    
    # 1. Рассчитываем коэффициенты фильтра
    filter_coeffs = np.zeros(n)
    for i in range(n):
        temp = i - window_length / 2
        # Коэффициент режекции (упрощенная модель из вашего TS кода)
        coeff = 1 - 2 * cutoff_frequency * np.cos(2 * np.pi * temp * i / n) + cutoff_frequency
        
        # Окно Хемминга
        if i < window_length:
            hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (window_length - 1))
        else:
            hamming = 0
            
        filter_coeffs[i] = coeff * hamming

    # 2. Нормализация фильтра
    filter_sum = np.sum(filter_coeffs)
    if filter_sum != 0:
        filter_coeffs /= filter_sum

    # 3. Фильтрация через спектральную область (согласно логике вашего TS)
    spec_signal = fft(signal)
    
    # Свертка в спектральной области (в TS у вас была свертка спектра и фильтра)
    # ПРИМЕЧАНИЕ: В классическом DSP фильтрация — это свертка СИГНАЛА и фильтра 
    # (или умножение их спектров). Я сохраняю структуру вашей логики:
    result_spec = np.zeros(n, dtype=complex)
    for i in range(n):
        for j in range(window_length):
            if i - j >= 0:
                result_spec[i] += spec_signal[i - j] * filter_coeffs[j]
                
    return ifft(result_spec)
