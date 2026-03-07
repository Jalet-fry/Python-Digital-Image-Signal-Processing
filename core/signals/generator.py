import numpy as np
from core.utils.aspects import log_dsp_action

@log_dsp_action
def generate_instrument_signal(amplitudes, f0, harmonics, phases, duration=0.02, sr=10000):
    """
    Генерирует сложный сигнал инструмента (Виолончель/Контрабас) методом аддитивного синтеза.
    """
    # ШАГ 1: Сетка времени (например, 1024 точки)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # ШАГ 2: Создаем массив-заготовку (ленту) из нулей.
    # Если N=1024, то в signal сейчас 1024 нуля: [0.0, 0.0, 0.0, ..., 0.0]
    signal = np.zeros_like(t)
    
    # ШАГ 3: ПОТОЧЕЧНОЕ СЛОЖЕНИЕ ВОЛН (Магия NumPy)
    for a, h in zip(amplitudes, harmonics):
        # f_current = f0 * h (Например: 110Гц, 220Гц, 330Гц...)
        
        # Это "новая звуковая волна" конкретной гармоники.
        new_wave = a * np.sin(2 * np.pi * (f0 * h) * t + phases)
        
        # ПАРАМИ: signal[0] + new_wave[0], signal[1] + new_wave[1]...
        #
        # ЧТО В ИТОГЕ:
        # 1-й проход: signal = [0,0...] + [волна_110Гц] = [числа волны 110Гц]
        # 2-й проход: signal = [числа волны 110Гц] + [числа волны 220Гц]
        # В итоге в signal[i] лежит СУММА всех синусоид в данный момент времени.
        signal += new_wave
    
    # ПОСЛЕ ЦИКЛА:
    # signal - это теперь "сложная кривая", которая состоит из суммы нескольких синусоид.
    # Каждая точка в массиве - это итоговое значение звука (его амплитуда).

    return t, signal
