import numpy as np
from core.signals.fourier import dft, fft, idft, ifft
from core.signals.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation
from core.signals.generator import generate_instrument_signal

class InstrumentProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.results = {}

    def run_analysis(self):
        N, sr = self.cfg.N, self.cfg.sr
        # 1. Генерация сигналов
        t, x_raw = generate_instrument_signal(self.cfg.x.amplitudes, self.cfg.x.f0, self.cfg.x.harmonics, self.cfg.x.phi, duration=N/sr, sr=sr)
        _, y_raw = generate_instrument_signal(self.cfg.y.amplitudes, self.cfg.y.f0, self.cfg.y.harmonics, self.cfg.y.phi, duration=N/sr, sr=sr)
        
        x, y = x_raw[:N], y_raw[:N]
        
        # 2. Преобразования
        spec_x = fft(x)
        spec_y = fft(y)
        
        # 3. Восстановление
        x_rec = ifft(spec_x).real[:N]
        y_rec = ifft(spec_y).real[:N]
        
        # 4. Свертка и Корреляция
        conv = fft_convolution(x, y).real
        corr = fft_correlation(x, y).real
        
        # 5. Ошибки (сравнение с библиотечными)
        lib_conv = np.convolve(x, y)
        err_conv = np.max(np.abs(conv - lib_conv))
        
        self.results = {
            't': t[:N], 'x': x, 'y': y,
            'spec_x': spec_x, 'spec_y': spec_y,
            'x_rec': x_rec, 'y_rec': y_rec,
            'conv': conv, 'corr': corr,
            'error_conv': err_conv
        }
        return self.results

    def get_audio_signals(self, duration_audio, sr_audio):
        _, x_audio = generate_instrument_signal(self.cfg.x.amplitudes, self.cfg.x.f0, self.cfg.x.harmonics, self.cfg.x.phi, duration=duration_audio, sr=sr_audio)
        _, y_audio = generate_instrument_signal(self.cfg.y.amplitudes, self.cfg.y.f0, self.cfg.y.harmonics, self.cfg.y.phi, duration=duration_audio, sr=sr_audio)
        return x_audio, y_audio
