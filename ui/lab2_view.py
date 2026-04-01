import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from core.utils.themes import UIColors
from core.dsp.fourier import fft
from scipy.signal import freqz

class Lab2View:
    def __init__(self, fig, cfg2):
        self.fig = fig
        self.cfg2 = cfg2
        UIColors.apply_dark_theme(plt)
        
        # Сетка из 3 вертикальных подграфиков (Время, Спектр, АЧХ/Гистограмма)
        self.main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
        plt.subplots_adjust(left=0.25, right=0.96, top=0.94, bottom=0.08, hspace=0.55)
        
        self.ax_menu = plt.axes([0.02, 0.45, 0.18, 0.4], facecolor=UIColors.BG_PANEL)
        self.labels = [
            "1. Чистый сигнал", 
            "2. Только Шум", 
            "3. Зашумленный (Вход)",
            "4. Сравнение (Все)", 
            "5. Фильтр MA", 
            "6. КИХ-фильтр (FIR)", 
            "7. БИХ-фильтр (IIR)"
        ]
        self.radio = RadioButtons(self.ax_menu, self.labels, active=0, activecolor=UIColors.RADIO_ACTIVE)
        
        for label in self.radio.labels:
            label.set_color(UIColors.TEXT_MAIN)
            label.set_fontsize(8)
            
        self.ax_play = plt.axes([0.02, 0.35, 0.18, 0.05])
        self.btn_play = Button(self.ax_play, '▶ Play Selected', color=UIColors.BTN_PLAY)
        self.btn_play.label.set_color(UIColors.TEXT_MAIN)
        
        self.status_text = fig.text(0.02, 0.02, "Ready", color=UIColors.TEXT_ACCENT, weight='bold')

    def update(self, label, res):
        for ax in self.main_axes: 
            ax.clear()
            ax.set_axis_on()
            ax.grid(True, color=UIColors.GRID, alpha=0.3)
            
        N_pts = 500
        t = res['t_axis'][:N_pts] * 1000
        sr = res['sr']

        if "Чистый" in label:
            self._draw_standard(res['x_clean'], t, sr, "Чистый сигнал (Виолончель)", UIColors.SIG_CLEAN)
        elif "Шум" in label:
            self._draw_standard(res['noise'], t, sr, "Шум (Белый + Наводка 1500Гц)", UIColors.TEXT_DIM)
        elif "Зашумленный" in label:
            self._draw_standard(res['x_noisy'], t, sr, f"Вход (SNR: {res['snr_noisy']:.1f} dB)", UIColors.SIG_NOISY)
        elif "Сравнение" in label:
            self._draw_comparison(res, t, sr)
        else: # Фильтры
            sig = res['y_ma'] if "MA" in label else res['y_fir'] if "КИХ" in label else res['y_iir']
            snr = res['snr_ma'] if "MA" in label else res['snr_fir'] if "КИХ" in label else res['snr_iir']
            col = UIColors.SIG_MA if "MA" in label else UIColors.SIG_FIR if "КИХ" in label else UIColors.SIG_IIR
            self._draw_filter_view(sig, res['x_noisy'], t, sr, f"{label} (SNR: {snr:.1f} dB)", col, label)

        self.fig.canvas.draw_idle()

    def _draw_standard(self, sig, t, sr, title, col):
        # 1. Время
        UIColors.setup_axis(self.main_axes[0], title, "мс", "Ампл.")
        self.main_axes[0].plot(t, sig[:len(t)], color=col, lw=1.5)
        # 2. Спектр с заливкой
        self._draw_spectrum_filled(self.main_axes[1], sig, sr, col)
        # 3. Гистограмма распределения
        self.main_axes[2].hist(sig, bins=50, color=col, alpha=0.7)
        UIColors.setup_axis(self.main_axes[2], "Распределение амплитуд (Статистика)")

    def _draw_filter_view(self, y, x, t, sr, title, col, label):
        # 1. Время (Вход vs Выход)
        UIColors.setup_axis(self.main_axes[0], title, "мс")
        self.main_axes[0].plot(t, x[:len(t)], color=UIColors.SIG_NOISY, alpha=0.3, label='Вход')
        self.main_axes[0].plot(t, y[:len(t)], color=col, lw=1.5, label='Выход')
        self.main_axes[0].legend(loc='upper right', fontsize=7)

        # 2. Спектральная очистка (Наложение)
        self._draw_spectrum_filled(self.main_axes[1], x, sr, UIColors.SIG_NOISY, alpha=0.2)
        self._draw_spectrum_filled(self.main_axes[1], y, sr, col, alpha=0.6, title="Спектральная очистка (До/После)")

        # 3. АЧХ фильтра
        ax = self.main_axes[2]
        if "MA" in label: 
            b, a = np.ones(self.cfg2.M_ma)/self.cfg2.M_ma, [1.0]
        elif "КИХ" in label:
            from core.dsp.filters import fir_window_design
            b = fir_window_design(self.cfg2.fir.f_range[0], self.cfg2.fir.f_range[1], self.cfg2.fir.M, sr)
            a = [1.0]
        else: # IIR
            from core.dsp.filters import iir_bandpass
            b, a = iir_bandpass(self.cfg2.iir.f0, self.cfg2.iir.bw, sr)
        
        w, h = freqz(b, a, worN=1024, fs=sr)
        ax.plot(w, np.abs(h), color=UIColors.TEXT_ACCENT, lw=2)
        UIColors.setup_axis(ax, "Амплитудно-частотная характеристика (АЧХ)")

    def _draw_spectrum_filled(self, ax, sig, sr, col, alpha=0.5, title="Амплитудный спектр"):
        X = fft(sig)
        N_fft = len(X)
        freqs = np.linspace(0, sr/2, N_fft//2)
        mags = 2.0/N_fft * np.abs(X[:N_fft//2])
        
        ax.fill_between(freqs, mags, color=col, alpha=alpha)
        UIColors.setup_axis(ax, title, "Гц")

    def _draw_comparison(self, res, t, sr):
        ax = self.main_axes[0]
        ax.plot(t, res['x_clean'][:len(t)], 'w--', alpha=0.5, label='Clean')
        ax.plot(t, res['y_ma'][:len(t)], color=UIColors.SIG_MA, label='MA')
        ax.plot(t, res['y_fir'][:len(t)], color=UIColors.SIG_FIR, label='FIR')
        ax.plot(t, res['y_iir'][:len(t)], color=UIColors.SIG_IIR, label='IIR')
        ax.legend(ncol=2, fontsize=7)
        UIColors.setup_axis(ax, "Сравнение всех фильтров")
        
        # Спектр наложения (Вход vs FIR)
        self._draw_spectrum_filled(self.main_axes[1], res['x_noisy'], sr, UIColors.SIG_NOISY, alpha=0.2)
        self._draw_spectrum_filled(self.main_axes[1], res['y_fir'], sr, UIColors.SIG_FIR, alpha=0.6, title="Спектры (Вход vs FIR)")

        self.main_axes[2].axis('off')
