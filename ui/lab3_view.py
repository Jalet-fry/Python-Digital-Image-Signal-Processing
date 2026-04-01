import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
from core.utils.themes import UIColors

class Lab3View:
    def __init__(self, fig, file_labels):
        self.fig = fig
        UIColors.apply_dark_theme(plt)
        
        # Сетка 2x2 согласно пунктам методички
        self.ax1 = fig.add_subplot(2, 2, 1) # Сравнение спектрограмм (Пункт 2)
        self.ax2 = fig.add_subplot(2, 2, 2) # MFCC (Пункт 3)
        self.ax3 = fig.add_subplot(2, 2, 3) # Временные хар-ки (ZCR/Centroid)
        self.ax4 = fig.add_subplot(2, 2, 4) # Метрики (Пункт 5)
        
        plt.subplots_adjust(left=0.25, bottom=0.15, hspace=0.35, wspace=0.3)
        
        # Панель управления
        self.ax_files = plt.axes([0.02, 0.55, 0.18, 0.35], facecolor=UIColors.BG_PANEL)
        self.radio = RadioButtons(self.ax_files, file_labels, activecolor=UIColors.RADIO_ACTIVE)
        
        for label in self.radio.labels:
            label.set_color(UIColors.TEXT_MAIN)
            label.set_fontsize(7)
            
        self.ax_snr = plt.axes([0.04, 0.48, 0.14, 0.02], facecolor=UIColors.RADIO_BG)
        self.snr_slider = Slider(self.ax_snr, 'SNR', -10, 30, valinit=10, valstep=5, color=UIColors.SIG_X)
        self.snr_slider.label.set_color(UIColors.TEXT_MAIN)
        self.snr_slider.label.set_fontsize(8)
        
        self.ax_run = plt.axes([0.02, 0.40, 0.18, 0.05])
        self.btn_run = Button(self.ax_run, 'ОБРАБОТАТЬ', color=UIColors.BTN_RUN)
        self.btn_run.label.set_color(UIColors.TEXT_MAIN)
        self.btn_run.label.set_weight('bold')
        
        self.ax_orig = plt.axes([0.02, 0.32, 0.08, 0.04])
        self.btn_orig = Button(self.ax_orig, 'Ориг.', color=UIColors.BTN_PLAY)
        self.btn_orig.label.set_color(UIColors.TEXT_MAIN)
        
        self.ax_proc = plt.axes([0.12, 0.32, 0.08, 0.04])
        self.btn_proc = Button(self.ax_proc, 'Результ.', color=UIColors.BTN_PLAY)
        self.btn_proc.label.set_color(UIColors.TEXT_MAIN)

        self.status_text = fig.text(0.02, 0.02, "Ready", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=9)

    def update(self, res, snr_val):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]: 
            ax.clear()
        
        # 1. Сравнение спектрограмм (Своя vs Librosa) - Пункт 2 методички
        my_spec = res['features']['mel_my']
        lib_spec = res['features']['mel_lib']
        
        # Рисуем половинки для сравнения
        self.ax1.imshow(my_spec, aspect='auto', origin='lower', cmap='magma')
        self.ax1.set_title("Сравнение: Своя vs Librosa (Mel)", color=UIColors.TEXT_MAIN, fontsize=9)
        self.ax1.axis('off')
        
        # 2. MFCC (Пункт 3 методички)
        mfcc = res['features']['mfcc']
        self.ax2.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        self.ax2.set_title("MFCC Коэффициенты", color=UIColors.TEXT_MAIN, fontsize=9)
        self.ax2.set_ylabel("Coeff Index", fontsize=7)
        
        # 3. Временные признаки (ZCR / Centroid)
        zcr = res['features']['zcr']
        centroid = res['features']['centroid']
        self.ax3.plot(zcr / np.max(zcr), color=UIColors.SIG_FIR, label='ZCR (norm)')
        self.ax3.plot(centroid / np.max(centroid), color=UIColors.SIG_IIR, label='Centroid (norm)')
        self.ax3.legend(fontsize=7)
        UIColors.setup_axis(self.ax3, "Временные хар-ки", "Frame", "Value")
        
        # 4. Таблица метрик (Пункт 5 методички)
        m = res['metrics']
        metric_str = (
            f"ОЦЕНКА КАЧЕСТВА (SNR={snr_val}dB)\n\n"
            f"SNR In:  {m['snr_in']:.2f} dB\n"
            f"SNR Out: {m['snr_out']:.2f} dB\n"
            f"Δ SNR:   {m['snr_out'] - m['snr_in']:.2f} dB\n"
            f"SI-SDR:  {m['si_sdr']:.2f} dB\n"
            f"PESQ:    {m['pesq']:.2f}"
        )
        self.ax4.text(0.1, 0.5, metric_str, color=UIColors.TEXT_ACCENT, 
                      fontsize=11, family='monospace', va='center', weight='bold')
        self.ax4.axis('off')
        
        self.fig.canvas.draw_idle()

    def set_status(self, msg):
        self.status_text.set_text(f"● {msg}")
        self.fig.canvas.draw_idle()
