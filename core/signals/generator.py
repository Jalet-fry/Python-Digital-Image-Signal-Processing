import numpy as np

def generate_instrument_signal(amplitudes, f0, harmonics, phases, duration=0.02, sr=10000):
    """
    Генерирует сигнал инструмента на основе гармоник.
    :param amplitudes: Список амплитуд гармоник (A)
    :param f0: Основная частота (Гц)
    :param harmonics: Номера гармоник (h)
    :param phases: Начальные фазы (phi)
    :param duration: Длительность в секундах
    :param sr: Частота дискретизации
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    # Суммируем все гармоники
    for a, h in zip(amplitudes, harmonics):
        # f = f0 * h
        signal += a * np.sin(2 * np.pi * (f0 * h) * t + phases)
        
    return t, signal
