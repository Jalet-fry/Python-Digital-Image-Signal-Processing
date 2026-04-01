import numpy as np
from core.dsp.fourier import dft, fft, idft, ifft
from core.dsp.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation
from core.dsp.generator import generate_instrument_signal

class InstrumentProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.results = None

    def prepare(self):
        """Вычисляет все 24 графика согласно методичке."""
        N, sr = self.cfg.N, self.cfg.sr
        
        # Сигналы x(t) и y(t)
        t, x_raw = generate_instrument_signal(self.cfg.x.amplitudes, self.cfg.x.f0, self.cfg.x.harmonics, self.cfg.x.phi, duration=N/sr, sr=sr)
        _, y_raw = generate_instrument_signal(self.cfg.y.amplitudes, self.cfg.y.f0, self.cfg.y.harmonics, self.cfg.y.phi, duration=N/sr, sr=sr)
        
        x, y = x_raw[:N], y_raw[:N]
        dt = t[1] - t[0]

        # Прямые преобразования
        dx, dy = dft(x), dft(y)
        fx, fy = fft(x), fft(y)
        lx, ly = np.fft.fft(x), np.fft.fft(y) # Lib

        # Свертки и корреляции
        c_vremya = linear_convolution(x, y)
        c_fft = fft_convolution(x, y)
        c_lib = np.convolve(x, y)
        
        cr_vremya = correlation(x, y)
        cr_fft = fft_correlation(x, y)
        cr_lib = np.correlate(x, y, mode='full')

        def get_mp(data):
            mag = np.abs(data) / (N / 2)
            ph = np.angle(data)
            ph[mag < 0.001 * np.max(mag)] = 0 # Чистка шума фазы
            freqs = np.fft.fftfreq(N, d=dt)
            idx = np.where(freqs[:N//2] <= 600)[0][-1]
            return freqs[:idx], mag[:idx], ph[:idx]

        # Формируем словарь по номерам графиков из методички
        self.results = {
            't': t[:N] * 1000,
            'f_axis': get_mp(fx)[0],
            'n_conv': range(len(c_vremya)),
            'n_corr': range(len(cr_vremya)),
            # Группировка для UI
            'data': {
                '01': (x, "Рис 1. Исходный x(t)", "Время (мс)"),
                '02': (y, "Рис 2. Исходный y(t)", "Время (мс)"),
                '03': (get_mp(dx)[1], "Рис 3. ДПФ Ампл. x(t)", "Частота (Гц)"),
                '04': (get_mp(dx)[2], "Рис 4. ДПФ Фаза x(t)", "Частота (Гц)"),
                '05': (idft(dx).real, "Рис 5. Восст. ОДПФ x(t)", "Время (мс)"),
                '06': (get_mp(fx)[1], "Рис 6. БПФ Ампл. x(t)", "Частота (Гц)"),
                '07': (get_mp(fx)[2], "Рис 7. БПФ Фаза x(t)", "Частота (Гц)"),
                '08': (ifft(fx).real[:N], "Рис 8. Восст. ОБПФ x(t)", "Время (мс)"),
                '09': (get_mp(dy)[1], "Рис 9. ДПФ Ампл. y(t)", "Частота (Гц)"),
                '10': (get_mp(dy)[2], "Рис 10. ДПФ Фаза y(t)", "Частота (Гц)"),
                '11': (idft(dy).real, "Рис 11. Восст. ОДПФ y(t)", "Время (мс)"),
                '12': (get_mp(fy)[1], "Рис 12. БПФ Ампл. y(t)", "Частота (Гц)"),
                '13': (get_mp(fy)[2], "Рис 13. БПФ Фаза y(t)", "Частота (Гц)"),
                '14': (ifft(fy).real[:N], "Рис 14. Восст. ОБПФ y(t)", "Время (мс)"),
                '15': (c_vremya, "Рис 15. Свертка (Время)", "n"),
                '16': (c_fft, "Рис 16. Свертка (FFT)", "n"),
                '17': (cr_vremya, "Рис 17. Корреляция (Время)", "n"),
                '18': (cr_fft, "Рис 18. Корреляция (FFT)", "n"),
                '19': (get_mp(lx)[1], "Рис 19. Lib Ампл. x(t)", "Частота (Гц)"),
                '20': (get_mp(lx)[2], "Рис 20. Lib Фаза x(t)", "Частота (Гц)"),
                '21': (get_mp(ly)[1], "Рис 21. Lib Ампл. y(t)", "Частота (Гц)"),
                '22': (get_mp(ly)[2], "Рис 22. Lib Фаза y(t)", "Частота (Гц)"),
                '23': (c_lib, "Рис 23. Lib Свертка", "n"),
                '24': (cr_lib, "Рис 24. Lib Корреляция", "n")
            },
            'errors': [np.max(np.abs(x - idft(dx).real)), np.max(np.abs(x - ifft(fx).real[:N]))]
        }
        return self.results
