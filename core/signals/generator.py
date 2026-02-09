import numpy as np

def generate_instrument_signal(amplitudes, f0, harmonics, phases, duration=0.02, sr=10000):
    """
    Генерирует сигнал инструмента на основе гармоник.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    # Формула аддитивного синтеза:
    # s(t) = sum(A_i * sin(2*pi * (f0 * h_i) * t + phi))
    for a, h in zip(amplitudes, harmonics):
        signal += a * np.sin(2 * np.pi * (f0 * h) * t + phases)
        
    return t, signal
