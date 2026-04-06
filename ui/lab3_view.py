import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from core.utils.themes import UIColors
import librosa.display

class CustomRadioButtons:
    def __init__(self, ax, labels, activecolor):
        self.ax = ax
        self.all_labels = labels
        self.activecolor = activecolor
        self.value_selected = labels[0] if labels else None
        self.on_clicked_cbs = []
        self.page_size = 8  
        self.current_page = 0
        self.total_pages = (len(labels) - 1) // self.page_size + 1 if labels else 1
        self.ax_prev = plt.axes([0.41, 0.95, 0.04, 0.03])
        self.ax_next = plt.axes([0.46, 0.95, 0.04, 0.03])
        self.btn_prev = Button(self.ax_prev, '▲', color='#2D3748', hovercolor='#4A5568')
        self.btn_next = Button(self.ax_next, '▼', color='#2D3748', hovercolor='#4A5568')
        self.btn_prev.label.set_color('white'); self.btn_next.label.set_color('white')
        self.btn_prev.on_clicked(self._prev_page)
        self.btn_next.on_clicked(self._next_page)
        self._setup_ui()
        self.ax.figure.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_ui(self):
        self.ax.clear()
        self.ax.set_facecolor('#0D1117')
        for spine in self.ax.spines.values():
            spine.set_visible(True); spine.set_color('#00FFFF')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.all_labels))
        page_labels = self.all_labels[start_idx:end_idx]
        step = 0.10  
        start_y = 0.85 
        for i, text in enumerate(page_labels):
            y = start_y - i * step
            is_active = (text == self.value_selected)
            circle = plt.Circle((0.05, y), 0.018, transform=self.ax.transAxes,
                                edgecolor='white', facecolor=self.activecolor if is_active else 'none')
            self.ax.add_patch(circle)
            display_text = text if len(text) < 40 else text[:37] + "..."
            self.ax.text(0.12, y, display_text, transform=self.ax.transAxes,
                         color='white', fontsize=9, va='center')
        self.ax.figure.canvas.draw_idle()

    def _prev_page(self, event):
        if self.current_page > 0: self.current_page -= 1; self._setup_ui()
    def _next_page(self, event):
        if self.current_page < self.total_pages - 1: self.current_page += 1; self._setup_ui()
    def _on_click(self, event):
        if event.inaxes != self.ax: return
        y_click = event.ydata
        idx_on_page = int(round((0.85 - y_click) / 0.10))
        global_idx = self.current_page * self.page_size + idx_on_page
        if 0 <= global_idx < len(self.all_labels):
            self.value_selected = self.all_labels[global_idx]
            self._setup_ui()
            for cb in self.on_clicked_cbs: cb(self.value_selected)
    def on_clicked(self, func): self.on_clicked_cbs.append(func)

class Lab3View:
    def __init__(self, fig, file_labels):
        self.fig = fig
        UIColors.apply_style(plt)
        fig.patch.set_facecolor(UIColors.BG_DARK)
        self.ax_p1_1 = fig.add_subplot(2, 2, 1)
        self.ax_p1_2 = fig.add_subplot(2, 2, 2)
        self.ax_p1_3 = fig.add_subplot(2, 1, 2)
        self.page1_axes = [self.ax_p1_1, self.ax_p1_2, self.ax_p1_3]
        self.ax_p2_1 = fig.add_subplot(2, 2, 1, visible=False)
        self.ax_p2_2 = fig.add_subplot(2, 2, 2, visible=False)
        self.ax_p2_3 = fig.add_subplot(2, 1, 2, visible=False)
        self.page2_axes = [self.ax_p2_1, self.ax_p2_2, self.ax_p2_3]
        plt.subplots_adjust(left=0.53, right=0.98, bottom=0.08, top=0.94, hspace=0.4, wspace=0.25)
        self.ax_files = plt.axes([0.01, 0.58, 0.49, 0.37])
        self.radio = CustomRadioButtons(self.ax_files, file_labels, UIColors.RADIO_ACTIVE)
        self.ax_snr = plt.axes([0.05, 0.50, 0.43, 0.025], facecolor=UIColors.BG_PANEL)
        self.snr_slider = Slider(self.ax_snr, '', 8, 24, valinit=10, valstep=2, color=UIColors.TEXT_ACCENT)
        self.snr_label = fig.text(0.26, 0.535, "SNR: 10 дБ", ha='center', weight='bold', color=UIColors.TEXT_ACCENT, fontsize=12)
        self.btn_page1 = Button(plt.axes([0.01, 0.45, 0.24, 0.04]), 'P1: АНАЛИЗ', color=UIColors.BTN_PLAY)
        self.btn_page2 = Button(plt.axes([0.26, 0.45, 0.24, 0.04]), 'P2: ФИЛЬТР', color='#2D3748')
        self.btn_run = Button(plt.axes([0.01, 0.36, 0.49, 0.07]), 'ЗАПУСТИТЬ ОБРАБОТКУ', color=UIColors.BTN_RUN)
        self.btn_orig = Button(plt.axes([0.01, 0.31, 0.15, 0.04]), 'ОРИГИНАЛ', color=UIColors.BTN_PLAY)
        self.btn_noisy = Button(plt.axes([0.17, 0.31, 0.16, 0.04]), 'ШУМ', color='#D32F2F')
        self.btn_proc = Button(plt.axes([0.34, 0.31, 0.16, 0.04]), 'ОЧИЩЕН', color='#388E3C')
        self.btn_stop = Button(plt.axes([0.01, 0.27, 0.49, 0.03]), 'СТОП', color='#B71C1C')
        self.ax_info = plt.axes([0.01, 0.04, 0.49, 0.21])
        self.ax_info.axis('off')
        self.info_text_res = self.ax_info.text(0.0, 1.0, "Готов к работе...", va='top', fontsize=9, color='white', family='monospace')
        self.info_text_feat = self.ax_info.text(0.52, 1.0, "", va='top', fontsize=9, color='white', family='monospace')
        self.status_text = fig.text(0.01, 0.015, "● Готов", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=10)
        self.btn_page1.on_clicked(lambda e: self.set_page(1))
        self.btn_page2.on_clicked(lambda e: self.set_page(2))
        self.snr_slider.on_changed(lambda v: self.snr_label.set_text(f"SNR: {int(v)} дБ"))
        self.current_page = 1

    def set_page(self, page_num):
        self.current_page = page_num
        is_p1 = (page_num == 1)
        for ax in self.page1_axes: ax.set_visible(is_p1)
        for ax in self.page2_axes: ax.set_visible(not is_p1)
        self.btn_page1.color = UIColors.BTN_PLAY if is_p1 else '#2D3748'
        self.btn_page2.color = UIColors.BTN_PLAY if not is_p1 else '#2D3748'
        self.fig.canvas.draw_idle()

    def clear_gui(self):
        for ax in self.page1_axes + self.page2_axes: ax.clear(); ax.set_facecolor('black')
        self.info_text_res.set_text("ОБРАБОТКА..."); self.info_text_feat.set_text("")
        self.fig.canvas.draw_idle(); plt.pause(0.01)

    def set_status(self, msg, color=None):
        self.status_text.set_text(f"● {msg}")
        if color: self.status_text.set_color(color)
        self.fig.canvas.draw_idle()

    def update(self, res, snr_val):
        if res is None: return
        sr = res['sr']
        f, m = res['features'], res['metrics']
        
        # Графики
        UIColors.setup_axis(self.ax_p1_1, "1. РУЧНАЯ МЕЛ-СПЕКТР.", "Время (сек)", "Гц")
        extent = [0, len(res['clean'])/sr, f['mel_my_freqs'][0], f['mel_my_freqs'][-1]]
        self.ax_p1_1.imshow(f['mel_my'], aspect='auto', origin='lower', cmap='magma', extent=extent)
        
        UIColors.setup_axis(self.ax_p1_2, "2. LIBROSA МЕЛ-СПЕКТР", "Время (сек)", "Гц")
        librosa.display.specshow(f['mel_lib'], ax=self.ax_p1_2, sr=sr, x_axis='time', y_axis='mel')
        
        UIColors.setup_axis(self.ax_p1_3, "3. STFT (ЛОГ. МАСШТАБ)", "Время (сек)", "Гц")
        librosa.display.specshow(f['stft_lib'], ax=self.ax_p1_3, sr=sr, x_axis='time', y_axis='log')
        
        UIColors.setup_axis(self.ax_p2_1, "4. MFCC (ТЕМБР)", "Индекс", "дБ")
        self.ax_p2_1.bar(range(len(f['mfcc'])), f['mfcc'], color=UIColors.TEXT_ACCENT)
        
        UIColors.setup_axis(self.ax_p2_2, "5. ВХОД (ШУМ)", "Время (сек)", "Гц")
        librosa.display.specshow(f['spec_noisy'], ax=self.ax_p2_2, sr=sr, x_axis='time', y_axis='mel')
        
        UIColors.setup_axis(self.ax_p2_3, "6. ВЫХОД (ОЧИЩЕН)", "Время (сек)", "Гц")
        librosa.display.specshow(f['spec_enh'], ax=self.ax_p2_3, sr=sr, x_axis='time', y_axis='mel')

        # Таблица результатов
        res_report = (f"ОБЪЕКТИВНАЯ ОЦЕНКА:\n"
                      f"----------------------\n"
                      f"SNR (in): {m['snr_in']:>7.2f} дБ\n"
                      f"SDR:      {m['sdr']:>7.2f} дБ\n"
                      f"SI-SDR:   {m['si_sdr']:>7.2f} дБ\n"
                      f"PESQ:     {m['pesq']:>7.2f}\n"
                      f"NISQA:    {m['nisqa']:>7.2f}\n"
                      f"DNSMOS:   {m['dnsmos']:>7.2f}\n"
                      f"Субъект.: [ ОПРОС ]")
        
        feat_report = (f"СПЕКТР. ПРИЗНАКИ:\n"
                       f"----------------------\n"
                       f"Rolloff:  {f['rolloff']:>7.0f} Гц\n"
                       f"Центроид: {f['centroid']:>7.0f} Гц\n"
                       f"Ширина:   {f['bandwidth']:>7.0f} Гц\n"
                       f"Chroma:   {f['chroma']:>9.4f}\n"
                       f"ZCR:      {f['zcr']:>9.4f}")
        
        self.info_text_res.set_text(res_report)
        self.info_text_feat.set_text(feat_report)
        self.set_page(self.current_page)
