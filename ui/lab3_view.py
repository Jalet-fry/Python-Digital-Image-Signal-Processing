import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
from core.utils.themes import UIColors
import librosa.display

class Lab3View:
    def __init__(self, fig, file_labels):
        self.fig = fig
        UIColors.apply_dark_theme(plt)
        
        # Сетка 2x2 как в оригинале
        self.ax1 = fig.add_subplot(2, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 3)
        self.ax4 = fig.add_subplot(2, 2, 4)
        
        plt.subplots_adjust(left=0.22, bottom=0.15, hspace=0.35, wspace=0.35)
        
        # Левая панель управления
        # 1. Список файлов
        self.ax_files = plt.axes([0.02, 0.6, 0.16, 0.3], facecolor=UIColors.BG_PANEL)
        self.radio = RadioButtons(self.ax_files, file_labels, activecolor=UIColors.RADIO_ACTIVE)
        for label in self.radio.labels:
            label.set_color(UIColors.TEXT_MAIN)
            label.set_fontsize(7)
            
        # 2. Настройка SNR (Слайдер + Кнопки)
        self.ax_snr = plt.axes([0.05, 0.52, 0.10, 0.02], facecolor=UIColors.RADIO_BG)
        self.snr_slider = Slider(self.ax_snr, 'SNR', 8, 24, valinit=10, valstep=2, color=UIColors.TEXT_ACCENT)
        self.snr_slider.label.set_color(UIColors.TEXT_MAIN)
        self.snr_slider.label.set_fontsize(8)
        
        self.ax_minus = plt.axes([0.02, 0.52, 0.02, 0.02])
        self.btn_minus = Button(self.ax_minus, '<', color=UIColors.BG_PANEL, hovercolor=UIColors.GRID)
        self.btn_minus.label.set_color(UIColors.TEXT_MAIN)
        
        self.ax_plus = plt.axes([0.16, 0.52, 0.02, 0.02])
        self.btn_plus = Button(self.ax_plus, '>', color=UIColors.BG_PANEL, hovercolor=UIColors.GRID)
        self.btn_plus.label.set_color(UIColors.TEXT_MAIN)
        
        # 3. Кнопка запуска
        self.ax_run = plt.axes([0.02, 0.44, 0.16, 0.05])
        self.btn_run = Button(self.ax_run, 'ЗАПУСК', color=UIColors.BTN_RUN)
        self.btn_run.label.set_color(UIColors.TEXT_MAIN)
        self.btn_run.label.set_weight('bold')
        
        # 4. Проигрывание
        self.ax_orig = plt.axes([0.02, 0.38, 0.07, 0.04])
        self.btn_orig = Button(self.ax_orig, 'Ориг.', color=UIColors.BTN_PLAY)
        self.btn_orig.label.set_color(UIColors.TEXT_MAIN)
        
        self.ax_proc = plt.axes([0.11, 0.38, 0.07, 0.04])
        self.btn_proc = Button(self.ax_proc, 'Очищ.', color=UIColors.BTN_PLAY)
        self.btn_proc.label.set_color(UIColors.TEXT_MAIN)

        # 5. Инфо-панель (Текст)
        self.ax_info = plt.axes([0.02, 0.05, 0.16, 0.3])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0, 1, "Выберите файл\nи нажмите Запуск", 
                                         va='top', fontsize=9, color=UIColors.TEXT_MAIN, family='monospace')

        self.status_text = fig.text(0.02, 0.02, "Ready", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=8)

        # Обработка кнопок +/- для SNR
        def go_minus(event):
            if self.snr_slider.val > 8: self.snr_slider.set_val(self.snr_slider.val - 2)
        def go_plus(event):
            if self.snr_slider.val < 24: self.snr_slider.set_val(self.snr_slider.val + 2)
        self.btn_minus.on_clicked(go_minus)
        self.btn_plus.on_clicked(go_plus)

    def update(self, res, snr_val):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]: ax.clear()
        
        # 1. Своя мел-спектрограмма (imshow)
        self.ax1.imshow(res['features']['mel_my'], aspect='auto', origin='lower', cmap='magma')
        self.ax1.set_title("Ручная Мел-спектрограмма", color=UIColors.TEXT_MAIN, fontsize=10)
        
        # 2. MFCC (Бар-чарт как в оригинале)
        self.ax2.bar(range(len(res['features']['mfcc'])), res['features']['mfcc'], color=UIColors.LAB3['mfcc'])
        self.ax2.set_title("MFCC (Средние коэф.)", color=UIColors.TEXT_MAIN, fontsize=10)
        UIColors.setup_axis(self.ax2, title="MFCC")
        
        # 3. Зашумленный (Specshow)
        S_noisy = librosa.feature.melspectrogram(y=res['noisy'], sr=res['sr'])
        librosa.display.specshow(librosa.power_to_db(S_noisy), ax=self.ax3, sr=res['sr'], x_axis='time', y_axis='mel')
        self.ax3.set_title(f"Вход (SNR In: {res['metrics']['snr_in']:.1f} dB)", fontsize=9)
        
        # 4. Очищенный (Specshow)
        S_enh = librosa.feature.melspectrogram(y=res['enhanced'], sr=res['sr'])
        librosa.display.specshow(librosa.power_to_db(S_enh), ax=self.ax4, sr=res['sr'], x_axis='time', y_axis='mel')
        self.ax4.set_title(f"Выход (SNR Out: {res['metrics']['snr_out']:.1f} dB)", fontsize=9)
        
        # Обновление инфо-панели
        f = res['features']
        m = res['metrics']
        report = (
            f"РЕЗУЛЬТАТЫ (SNR={snr_val}):\n"
            f"-------------------\n"
            f"SNR Out:  {m['snr_out']:.2f} dB\n"
            f"SI-SDR:   {m['si_sdr_out']:.2f} dB\n"
            f"PESQ:     {m['pesq']:.2f}\n"
            f"-------------------\n"
            f"Rolloff:  {f['rolloff']:.0f} Hz\n"
            f"Centroid: {f['centroid']:.0f} Hz\n"
            f"Bandwidth:{f['bandwidth']:.0f} Hz\n"
            f"ZCR:      {f['zcr']:.4f}\n"
            f"Chroma:   {f['chroma']:.3f}"
        )
        self.info_text.set_text(report)
        self.fig.canvas.draw_idle()

    def set_status(self, msg):
        self.status_text.set_text(f"● {msg}")
        self.fig.canvas.draw_idle()
