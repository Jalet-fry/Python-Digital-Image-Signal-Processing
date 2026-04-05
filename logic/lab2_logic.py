import numpy as np
from core.dsp.generator import generate_instrument_signal
from core.dsp.filters import (
    moving_average_recursive, fir_manual_filter, fir_window_design, 
    iir_bandpass, apply_iir
)

class FilterProcessor:
    def __init__(self, cfg1, cfg2, sr=8000, duration=2.0):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self.sr = sr
        self.duration = duration
        self.results = None

    def prepare(self):
        """Предварительный расчет всех фильтров с компенсацией задержек."""
        t_axis = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        
        # 1. Генерация сигналов (как в старой версии)
        _, x_clean = generate_instrument_signal(
            self.cfg1.x.amplitudes, self.cfg1.x.f0, self.cfg1.x.harmonics, 0, 
            duration=self.duration, sr=self.sr
        )
        
        np.random.seed(42)
        white_noise = np.random.normal(0, 0.08, len(x_clean))
        interference = 0.4 * np.sin(2 * np.pi * 1500 * t_axis)
        total_noise = white_noise + interference
        x_noisy = x_clean + total_noise

        # 2. Расчет параметров фильтров
        # MA
        y_ma = moving_average_recursive(x_noisy, M=self.cfg2.M_ma)
        delay_ma = (self.cfg2.M_ma - 1) // 2
        
        # FIR (КИХ)
        f_low = self.cfg2.fir.f_range[0] if isinstance(self.cfg2.fir.f_range, (list, np.ndarray)) else self.cfg2.fir.f_range
        f_high = self.cfg2.fir.f_range[1] if isinstance(self.cfg2.fir.f_range, (list, np.ndarray)) else self.sr/2 - 1
        h_fir = fir_window_design(f_low, f_high, M=self.cfg2.fir.M, sr=self.sr, window_type=self.cfg2.fir.window)
        y_fir = fir_manual_filter(x_noisy, h_fir)
        delay_fir = (self.cfg2.fir.M - 1) // 2

        # IIR (БИХ)
        b_iir, a_iir = iir_bandpass(self.cfg2.iir.f0, self.cfg2.iir.bw, sr=self.sr)
        y_iir = apply_iir(x_noisy, b_iir, a_iir)
        
        # Оценка задержки для БИХ через корреляцию
        corr = np.correlate(x_clean, y_iir, mode='full')
        delay_iir = max(0, np.argmax(np.abs(corr)) - len(x_clean) + 1)

        def calc_snr_safe(clean, processed, delay=0):
            """Точный расчет SNR с компенсацией задержки (как в старой версии)."""
            c = clean
            p = processed
            if delay > 0:
                c = clean[:-delay]
                p = processed[delay:]
            elif delay < 0: # На всякий случай для опережения
                c = clean[-delay:]
                p = processed[:delay]
            
            noise_part = p - c
            return 10 * np.log10(np.sum(c**2) / (np.sum(noise_part**2) + 1e-12))

        self.results = {
            'x_clean': x_clean, 
            'x_noisy': x_noisy, 
            'noise': total_noise,
            'y_ma': y_ma, 
            'y_fir': y_fir, 
            'y_iir': y_iir,
            'coeffs': {
                'ma': (np.ones(self.cfg2.M_ma)/self.cfg2.M_ma, [1.0]),
                'fir': (h_fir, [1.0]),
                'iir': (b_iir, a_iir)
            },
            'snr': {
                'noisy': calc_snr_safe(x_clean, x_noisy),
                'ma': calc_snr_safe(x_clean, y_ma, delay_ma),
                'fir': calc_snr_safe(x_clean, y_fir, delay_fir),
                'iir': calc_snr_safe(x_clean, y_iir, delay_iir)
            },
            't_axis': t_axis, 
            'sr': self.sr
        }
        return self.results
