import numpy as np
from core.signals.generator import generate_instrument_signal
from core.signals.filters import (
    moving_average_recursive, fir_manual_filter, fir_window_design, 
    iir_design, apply_iir, iir_bandpass
)

class FilterProcessor:
    def __init__(self, cfg1, cfg2, sr=8000, duration=2.0):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self.sr = sr
        self.duration = duration
        self.t_axis = np.linspace(0, duration, int(sr * duration), endpoint=False)
        self.results = {}

    def run_processing(self):
        # 1. Генерация
        _, x_clean = generate_instrument_signal(
            self.cfg1.x.amplitudes, self.cfg1.x.f0, self.cfg1.x.harmonics, 0, 
            duration=self.duration, sr=self.sr
        )
        
        np.random.seed(42)
        white_noise = np.random.normal(0, 0.08, len(x_clean))
        interference = 0.4 * np.sin(2 * np.pi * 1500 * self.t_axis)
        total_noise = white_noise + interference
        x_noisy = x_clean + total_noise

        # 2. Фильтрация
        # MA
        y_ma = moving_average_recursive(x_noisy, M=self.cfg2.M_ma)
        
        # FIR
        f_low = self.cfg2.fir.f_range[0] if isinstance(self.cfg2.fir.f_range, (list, np.ndarray)) else self.cfg2.fir.f_range
        f_high = self.cfg2.fir.f_range[1] if isinstance(self.cfg2.fir.f_range, (list, np.ndarray)) else self.sr/2 - 1
        h_fir = fir_window_design(f_low, f_high, M=self.cfg2.fir.M, sr=self.sr, window_type=self.cfg2.fir.window)
        y_fir = fir_manual_filter(x_noisy, h_fir)

        # IIR
        b_iir, a_iir = iir_bandpass(self.cfg2.iir.f0, self.cfg2.iir.bw, sr=self.sr)
        y_iir = apply_iir(x_noisy, b_iir, a_iir)

        # 3. Расчет метрик
        def calc_snr(clean, processed, delay=0):
            c = clean if delay == 0 else clean[:-delay]
            p = processed if delay == 0 else processed[delay:]
            noise_part = p - c
            return 10 * np.log10(np.sum(c**2) / (np.sum(noise_part**2) + 1e-12))

        delay_ma = (self.cfg2.M_ma - 1) // 2
        delay_fir = (self.cfg2.fir.M - 1) // 2
        
        # Оценка задержки для IIR через корреляцию
        corr = np.correlate(x_clean, y_iir, mode='full')
        delay_iir = max(0, np.argmax(np.abs(corr)) - len(x_clean) + 1)

        self.results = {
            'x_clean': x_clean,
            'x_noisy': x_noisy,
            'noise': total_noise,
            'y_ma': y_ma,
            'y_fir': y_fir,
            'y_iir': y_iir,
            'snr_noisy': calc_snr(x_clean, x_noisy),
            'snr_ma': calc_snr(x_clean, y_ma, delay_ma),
            'snr_fir': calc_snr(x_clean, y_fir, delay_fir),
            'snr_iir': calc_snr(x_clean, y_iir, delay_iir),
            'h_fir': h_fir,
            'iir_coeffs': (b_iir, a_iir),
            't_axis': self.t_axis
        }
        return self.results
