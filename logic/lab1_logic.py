import numpy as np
import os
from scipy.io import wavfile
from core.dsp.fourier import dft, idft, fft, ifft
from core.dsp.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation
from core.dsp.generator import generate_instrument_signal

class InstrumentProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.results = None

    def prepare(self):
        """Полный расчет всех 24 графиков согласно оригинальной методичке."""
        N, sr = self.cfg.N, self.cfg.sr
        dt = 1.0 / sr
        
        # 1. Сигналы x(t) и y(t)
        t_raw, x_raw = generate_instrument_signal(self.cfg.x.amplitudes, self.cfg.x.f0, self.cfg.x.harmonics, self.cfg.x.phi, duration=N/sr, sr=sr)
        _, y_raw = generate_instrument_signal(self.cfg.y.amplitudes, self.cfg.y.f0, self.cfg.y.harmonics, self.cfg.y.phi, duration=N/sr, sr=sr)
        
        x_s, y_s = x_raw[:N], y_raw[:N]
        t = t_raw[:N]

        # 2. Преобразования
        d_x, f_x, l_fx = dft(x_s), fft(x_s), np.fft.fft(x_s)
        d_y, f_y, l_fy = dft(y_s), fft(y_s), np.fft.fft(y_s)

        # 3. Восстановление
        id_x, if_x = idft(d_x).real, ifft(f_x).real[:N]
        id_y, if_y = idft(d_y).real, ifft(f_y).real[:N]

        # 4. Свертки и корреляции
        c_m, c_f, c_l = linear_convolution(x_s, y_s), fft_convolution(x_s, y_s).real, np.convolve(x_s, y_s)
        cr_m, cr_f, cr_l = correlation(x_s, y_s), fft_correlation(x_s, y_s).real, np.correlate(x_s, y_s, mode='full')

        # 5. Функция "чистки" как в оригинале
        def get_clean(data, max_f=600):
            mag = np.abs(data) / (N / 2)
            ph = np.angle(data)
            ph[mag < 0.001 * np.max(mag)] = 0
            freqs = np.fft.fftfreq(N, d=dt)
            idx = np.where(freqs[:N//2] <= max_f)[0][-1]
            return freqs[:idx], mag[:idx], ph[:idx]

        f_dx, m_dx, p_dx = get_clean(d_x); f_fx, m_fx, p_fx = get_clean(f_x); f_lx, m_lx, p_lx = get_clean(l_fx)
        f_dy, m_dy, p_dy = get_clean(d_y); f_fy, m_fy, p_fy = get_clean(f_y); f_ly, m_ly, p_ly = get_clean(l_fy)

        # 6. Ошибки
        errors = [
            np.max(np.abs(x_s - id_x)), 
            np.max(np.abs(x_s - if_x)), 
            np.max(np.abs(x_s - np.fft.ifft(l_fx).real)), 
            np.max(np.abs(c_m - c_l)), 
            np.max(np.abs(cr_m - cr_l))
        ]

        # 7. ФОРМИРУЕМ plots_data В ТОЧНОМ ПОРЯДКЕ ОРИГИНАЛА
        plots_data = [
            (t*1000, x_s, "x(t) Исходный", "plot", "blue", "ms"),
            (t*1000, y_s, "y(t) Исходный", "plot", "orange", "ms"),
            (f_dx, m_dx, "x(t) ДПФ: амплитуда", "stem", "blue", "Hz"),
            (f_dx, p_dx, "x(t) ДПФ: фаза", "stem", "blue", "Hz"),
            (t*1000, id_x, "x(t) Восст. ОДПФ", "plot", "green", "ms"),
            (f_fx, m_fx, "x(t) БПФ: амплитуда", "stem", "blue", "Hz"),
            (f_fx, p_fx, "x(t) БПФ: фаза", "stem", "blue", "Hz"),
            (t*1000, if_x, "x(t) Восст. ОБПФ", "plot", "purple", "ms"),
            (f_dy, m_dy, "y(t) ДПФ: амплитуда", "stem", "orange", "Hz"),
            (f_dy, p_dy, "y(t) ДПФ: фаза", "stem", "orange", "Hz"),
            (t*1000, id_y, "y(t) Восст. ОДПФ", "plot", "red", "ms"),
            (f_fy, m_fy, "y(t) БПФ: амплитуда", "stem", "orange", "Hz"),
            (f_fy, p_fy, "y(t) БПФ: фаза", "stem", "orange", "Hz"),
            (t*1000, if_y, "y(t) Восст. ОБПФ", "plot", "brown", "ms"),
            (range(len(c_m)), c_m, "x*y Свертка (Лин)", "plot", "blue", "pts"),
            (range(len(c_f)), c_f, "x*y Свертка (FFT)", "plot", "cyan", "idx"),
            (range(len(cr_m)), cr_m, "x&y Корр (Лин)", "plot", "gray", "idx"),
            (range(len(cr_f)), cr_f, "x&y Корр (FFT)", "plot", "black", "idx"),
            (f_lx, m_lx, "x(t) Спектр (Lib)", "stem", "blue", "Hz"),
            (f_lx, p_lx, "x(t) Фаза (Lib)", "stem", "blue", "Hz"),
            (f_ly, m_ly, "y(t) Спектр (Lib)", "stem", "orange", "Hz"),
            (f_ly, p_ly, "y(t) Фаза (Lib)", "stem", "orange", "Hz"),
            (range(len(c_l)), c_l, "x*y Свертка (Lib)", "plot", "green", "idx"),
            (range(len(cr_l)), cr_l, "x&y Корр (Lib)", "plot", "red", "idx"),
        ]

        # Аудио (2 сек)
        sr_a, dur_a = self.cfg.sr_audio, self.cfg.duration_audio
        _, x_audio = generate_instrument_signal(self.cfg.x.amplitudes, self.cfg.x.f0, self.cfg.x.harmonics, self.cfg.x.phi, duration=dur_a, sr=sr_a)
        _, y_audio = generate_instrument_signal(self.cfg.y.amplitudes, self.cfg.y.f0, self.cfg.y.harmonics, self.cfg.y.phi, duration=dur_a, sr=sr_a)

        self.results = {
            'plots_data': plots_data,
            'errors': errors,
            'audio_x': x_audio,
            'audio_y': y_audio,
            'sr_audio': sr_a
        }
        return self.results

    def save_wav_files(self, BASE_DIR):
        audio_dir = os.path.join(BASE_DIR, "results", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        wavfile.write(os.path.join(audio_dir, f"x_{self.cfg.x.name}.wav"), self.results['sr_audio'], np.int16(self.results['audio_x']/np.max(np.abs(self.results['audio_x'])) * 32767))
        wavfile.write(os.path.join(audio_dir, f"y_{self.cfg.y.name}.wav"), self.results['sr_audio'], np.int16(self.results['audio_y']/np.max(np.abs(self.results['audio_y'])) * 32767))
        return audio_dir
