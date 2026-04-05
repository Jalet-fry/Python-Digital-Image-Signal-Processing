import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from core.utils.themes import UIColors
from core.dsp.fourier import fft
from scipy.signal import freqz
import mplcursors

class Lab2View:
    def __init__(self, fig, cfg2):
        self.fig = fig
        self.cfg2 = cfg2
        UIColors.apply_style(plt)
        plt.rcParams['toolbar'] = 'None'
        
        # Сетка: Время, Спектр (дБ), АЧХ
        self.main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
        # Огромный отступ слева (0.45) и снизу (0.18) для исключения наложений
        plt.subplots_adjust(left=0.45, right=0.95, top=0.92, bottom=0.18, hspace=0.7)
        
        self.ax_menu = plt.axes([0.02, 0.45, 0.35, 0.4], facecolor=UIColors.BG_PANEL)
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
            label.set_fontsize(10)
            label.set_weight('bold')
            
        # Кнопки
        self.btn_play = Button(plt.axes([0.02, 0.35, 0.35, 0.05]), 'PLAY SELECTED', color=UIColors.BTN_PLAY)
        self.btn_save_wav = Button(plt.axes([0.02, 0.28, 0.35, 0.05]), 'SAVE WAV', color=UIColors.BTN_PLAY)
        self.btn_save_res = Button(plt.axes([0.02, 0.21, 0.35, 0.05]), 'SAVE RESULTS', color=UIColors.BTN_RUN)
        
        for b in [self.btn_play, self.btn_save_wav, self.btn_save_res]:
            b.label.set_color('white')
            b.label.set_weight('bold')

        self.status_text = fig.text(0.02, 0.02, "Ready", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=12)
        self.current_cursor = None

    def set_status(self, msg, color=None):
        self.status_text.set_text(f"● {msg}")
        self.status_text.set_color(color if color else UIColors.TEXT_ACCENT)
        self.fig.canvas.draw_idle()

    def update(self, label, res):
        for ax in self.main_axes: 
            ax.clear()
            ax.set_axis_on()
            ax.set_facecolor('black')
            
        N_pts = 500
        t = res['t_axis'][:N_pts] * 1000
        sr = res['sr']
        
        X_ref = np.fft.fft(res['x_noisy'] * np.hanning(len(res['x_noisy'])))
        ref_peak = np.max(np.abs(X_ref))

        if "Чистый" in label:
            self._draw_standard(res['x_clean'], t, sr, "ЧИСТЫЙ СИГНАЛ", UIColors.SIG_CLEAN, ref_peak)
                
        elif "Шум" in label:
            self._draw_standard(res['noise'], t, sr, "ШУМ (БЕЛЫЙ + 1500 ГЦ)", UIColors.TEXT_DIM, ref_peak)
            
        elif "Зашумленный" in label:
            snr = res['snr']['noisy']
            self._draw_standard(res['x_noisy'], t, sr, f"ВХОД (SNR: {snr:.2f} dB)", UIColors.SIG_NOISY, ref_peak)
            
        elif "Сравнение" in label:
            self._draw_comparison(res, t, sr)
            
        else: # Фильтры
            key = 'ma' if "MA" in label else 'fir' if "КИХ" in label else 'iir'
            sig = res[f'y_{key}']
            snr = res['snr'][key]
            b, a = res['coeffs'][key]
            col = UIColors.SIG_MA if key == 'ma' else UIColors.SIG_FIR if key == 'fir' else UIColors.SIG_IIR
            self._draw_filter_view(sig, res['x_noisy'], t, sr, f"{label} (SNR: {snr:.2f} dB)", col, b, a, ref_peak)

        if self.current_cursor: self.current_cursor.remove()
        self.current_cursor = mplcursors.cursor(self.main_axes, hover=True)
        self.fig.canvas.draw_idle()

    def _draw_standard(self, sig, t, sr, title, col, ref_peak):
        ax0, ax1, ax2 = self.main_axes
        UIColors.setup_axis(ax0, title, "Время, мс", "Ампл.")
        ax0.plot(t, sig[:len(t)], color=col, lw=2)
        
        self._draw_spectrum_db(ax1, sig, sr, col, ref_peak)
        
        UIColors.setup_axis(ax2, "РАСПРЕДЕЛЕНИЕ АМПЛИТУД", "Значение", "Частота")
        ax2.hist(sig, bins=70, color=col, alpha=0.8)

    def _draw_filter_view(self, y, x, t, sr, title, col, b, a, ref_peak):
        ax0, ax1, ax2 = self.main_axes
        UIColors.setup_axis(ax0, title, "Время, мс", "Ампл.")
        ax0.plot(t, x[:len(t)], color=UIColors.SIG_NOISY, alpha=0.35, label='Вход')
        ax0.plot(t, y[:len(t)], color=col, lw=2, label='Выход')
        ax0.legend(loc='upper right', fontsize=10)

        self._draw_spectrum_db(ax1, x, sr, UIColors.SIG_NOISY, ref_peak, alpha=0.2)
        self._draw_spectrum_db(ax1, y, sr, col, ref_peak, alpha=0.8, title="СПЕКТРАЛЬНАЯ ОЧИСТКА (дБ)")

        w, h = freqz(b, a, worN=2000, fs=sr)
        UIColors.setup_axis(ax2, "АЧХ ФИЛЬТРА", "Частота, Гц", "Ампл (лин)")
        ax2.plot(w, np.abs(h), color='white', lw=2, label='АЧХ (лин)')
        
        ax_db = ax2.twinx()
        ax_db.plot(w, 20*np.log10(np.abs(h) + 1e-12), color=UIColors.TEXT_ACCENT, alpha=0.6, ls='--', label='АЧХ (дБ)')
        ax_db.set_ylabel("дБ", color=UIColors.TEXT_ACCENT, fontsize=10, fontweight='bold')
        ax_db.set_ylim(-60, 5)

    def _draw_spectrum_db(self, ax, sig, sr, col, ref_peak, alpha=0.7, title="СПЕКТР (дБ)"):
        N = len(sig)
        X = np.fft.fft(sig * np.hanning(N))
        freqs = np.fft.fftfreq(N, 1/sr)[:N//2]
        mags = np.abs(X)[:N//2]
        mags_db = 20 * np.log10(mags / (ref_peak + 1e-12) + 1e-12)
        
        ax.fill_between(freqs, -60, mags_db, color=col, alpha=alpha)
        ax.set_ylim(-60, 5); ax.set_xlim(0, 2500)
        UIColors.setup_axis(ax, title, "Гц", "дБ")

    def _draw_comparison(self, res, t, sr):
        ax = self.main_axes[0]
        UIColors.setup_axis(ax, "СРАВНЕНИЕ ФИЛЬТРОВ", "мс", "Ампл.")
        ax.plot(t, res['x_clean'][:len(t)], color='white', lw=1.5, label='Clean', zorder=10)
        ax.plot(t, res['y_ma'][:len(t)], color=UIColors.SIG_MA, label='MA', alpha=0.8)
        ax.plot(t, res['y_fir'][:len(t)], color=UIColors.SIG_FIR, label='FIR', alpha=0.8)
        ax.plot(t, res['y_iir'][:len(t)], color=UIColors.SIG_IIR, label='IIR', alpha=0.8)
        ax.legend(ncol=4, loc='upper right', fontsize=9)
        self.main_axes[1].axis('off'); self.main_axes[2].axis('off')
