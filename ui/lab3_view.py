import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from core.utils.themes import UIColors
import librosa.display

class CustomRadioButtons:
    """Стабильный аналог RadioButtons с поддержкой страниц (пагинации) и крупным шрифтом."""
    def __init__(self, ax, labels, activecolor):
        self.ax = ax
        self.all_labels = labels
        self.activecolor = activecolor
        self.value_selected = labels[0] if labels else None
        self.on_clicked_cbs = []
        
        self.page_size = 8  
        self.current_page = 0
        self.total_pages = (len(labels) - 1) // self.page_size + 1 if labels else 1
        
        self.circles = []
        self.labels = [] 
        
        # Кнопки навигации (справа сверху)
        self.ax_prev = plt.axes([0.41, 0.95, 0.04, 0.03])
        self.ax_next = plt.axes([0.46, 0.95, 0.04, 0.03])
        self.btn_prev = Button(self.ax_prev, '▲', color='#2D3748', hovercolor='#4A5568')
        self.btn_next = Button(self.ax_next, '▼', color='#2D3748', hovercolor='#4A5568')
        self.btn_prev.label.set_color('white'); self.btn_next.label.set_color('white')
        self.btn_prev.label.set_fontsize(10); self.btn_next.label.set_fontsize(10)
        
        self.btn_prev.on_clicked(self._prev_page)
        self.btn_next.on_clicked(self._next_page)
        
        self._setup_ui()
        self.ax.figure.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_ui(self):
        self.ax.clear()
        self.ax.set_facecolor('#0D1117')
        for spine in self.ax.spines.values():
            spine.set_visible(True); spine.set_color('#00FFFF'); spine.set_linewidth(1.5)
        
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        
        page_info = f"СТРАНИЦА {self.current_page + 1} / {self.total_pages}"
        self.ax.text(0.03, 0.95, page_info, transform=self.ax.transAxes, color=UIColors.TEXT_ACCENT, fontsize=9, weight='bold')

        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.all_labels))
        page_labels = self.all_labels[start_idx:end_idx]
        
        self.circles = []
        self.labels = []
        
        step = 0.10  
        start_y = 0.85 
        
        for i, text in enumerate(page_labels):
            y = start_y - i * step
            is_active = (text == self.value_selected)
            
            circle = plt.Circle((0.05, y), 0.018, transform=self.ax.transAxes,
                                edgecolor='white', facecolor=self.activecolor if is_active else 'none',
                                linewidth=1.2, zorder=10)
            self.ax.add_patch(circle)
            self.circles.append(circle)
            
            display_text = text if len(text) < 40 else text[:37] + "..."
            t = self.ax.text(0.12, y, display_text, transform=self.ax.transAxes,
                             color='white', fontsize=9, va='center', zorder=10)
            
            t.get_text = lambda s=text: s 
            self.labels.append(t)
            
            circle.set_clip_on(True); circle.set_clip_box(self.ax.bbox)
            t.set_clip_on(True); t.set_clip_box(self.ax.bbox)

        self.ax.figure.canvas.draw_idle()

    def _prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self._setup_ui()

    def _next_page(self, event):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._setup_ui()

    def _on_click(self, event):
        if event.inaxes != self.ax: return
        y_click = event.ydata
        step = 0.10
        start_y = 0.85
        idx_on_page = int(round((start_y - y_click) / step))
        
        start_idx = self.current_page * self.page_size
        global_idx = start_idx + idx_on_page
        
        if start_idx <= global_idx < min(start_idx + self.page_size, len(self.all_labels)):
            if idx_on_page >= 0 and idx_on_page < self.page_size:
                self.value_selected = self.all_labels[global_idx]
                self._setup_ui()
                for cb in self.on_clicked_cbs: cb(self.value_selected)

    def on_clicked(self, func):
        self.on_clicked_cbs.append(func)

class Lab3View:
    def __init__(self, fig, file_labels):
        self.fig = fig
        UIColors.apply_style(plt)
        fig.patch.set_facecolor(UIColors.BG_DARK)
        
        # 1. ГРАФИКИ (Сдвинуты вправо)
        self.ax_p1_1 = fig.add_subplot(2, 2, 1)
        self.ax_p1_2 = fig.add_subplot(2, 2, 2)
        self.ax_p1_3 = fig.add_subplot(2, 1, 2)
        self.page1_axes = [self.ax_p1_1, self.ax_p1_2, self.ax_p1_3]
        
        self.ax_p2_1 = fig.add_subplot(2, 2, 1, visible=False)
        self.ax_p2_2 = fig.add_subplot(2, 2, 2, visible=False)
        self.ax_p2_3 = fig.add_subplot(2, 1, 2, visible=False)
        self.page2_axes = [self.ax_p2_1, self.ax_p2_2, self.ax_p2_3]
        
        # Освобождаем больше места слева (left=0.53)
        plt.subplots_adjust(left=0.53, right=0.98, bottom=0.08, top=0.94, hspace=0.4, wspace=0.25)
        
        # --- САЙДБАР (РАСШИРЕН до 0.51) ---
        self.fig.add_artist(plt.Rectangle((0, 0), 0.51, 1, facecolor=UIColors.BG_PANEL, zorder=-1, alpha=0.4))
        fig.text(0.015, 0.96, "СПИСОК АУДИОФАЙЛОВ:", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=11)

        # 2. СПИСОК ФАЙЛОВ (Wider)
        self.ax_files = plt.axes([0.01, 0.58, 0.49, 0.37])
        self.radio = CustomRadioButtons(self.ax_files, file_labels, UIColors.RADIO_ACTIVE)

        # 3. УПРАВЛЕНИЕ (Wider)
        self.ax_snr = plt.axes([0.05, 0.50, 0.43, 0.025], facecolor=UIColors.BG_PANEL)
        self.snr_slider = Slider(self.ax_snr, '', 8, 24, valinit=10, valstep=2, color=UIColors.TEXT_ACCENT)
        self.snr_slider.valtext.set_visible(False)
        self.snr_label = fig.text(0.26, 0.535, "SNR: 10 дБ", ha='center', weight='bold', color=UIColors.TEXT_ACCENT, fontsize=14)

        self.btn_page1 = Button(plt.axes([0.01, 0.45, 0.24, 0.04]), 'P1: АНАЛИЗ', color=UIColors.BTN_PLAY)
        self.btn_page2 = Button(plt.axes([0.26, 0.45, 0.24, 0.04]), 'P2: ФИЛЬТР', color='#2D3748')
        self.btn_page1.label.set_color('white'); self.btn_page2.label.set_color('white')

        self.btn_run = Button(plt.axes([0.01, 0.36, 0.49, 0.07]), 'ЗАПУСТИТЬ ОБРАБОТКУ', color=UIColors.BTN_RUN)
        self.btn_run.label.set_weight('bold'); self.btn_run.label.set_color('white'); self.btn_run.label.set_fontsize(11)
        
        self.btn_orig = Button(plt.axes([0.01, 0.31, 0.15, 0.04]), 'ОРИГИНАЛ', color=UIColors.BTN_PLAY)
        self.btn_noisy = Button(plt.axes([0.17, 0.31, 0.16, 0.04]), 'ШУМ', color='#D32F2F')
        self.btn_proc = Button(plt.axes([0.34, 0.31, 0.16, 0.04]), 'ОЧИЩЕН', color='#388E3C')
        self.btn_stop = Button(plt.axes([0.01, 0.27, 0.49, 0.03]), 'СТОП', color='#B71C1C')
        
        for b in [self.btn_orig, self.btn_noisy, self.btn_proc, self.btn_stop]: 
            b.label.set_color('white'); b.label.set_fontsize(9); b.label.set_weight('bold')

        # 4. ИНФОРМАЦИОННАЯ ПАНЕЛЬ (ДВЕ КОЛОНКИ)
        self.ax_info = plt.axes([0.01, 0.04, 0.49, 0.20])
        self.ax_info.axis('off')
        # Колонка 1: Результаты
        self.info_text_res = self.ax_info.text(0.0, 1.0, "Готов к работе...", va='top', fontsize=10, 
                                               color='white', family='monospace', fontweight='bold')
        # Колонка 2: Признаки
        self.info_text_feat = self.ax_info.text(0.55, 1.0, "", va='top', fontsize=10, 
                                                color='white', family='monospace', fontweight='bold')
        
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

    def update(self, res, snr_val):
        if res is None: return
        sr = res['sr']
        UIColors.setup_axis(self.ax_p1_1, "1. РУЧНАЯ МЕЛ-СПЕКТРОГРАММА", "Время (кадры)", "Мел-фильтры")
        self.ax_p1_1.imshow(res['features']['mel_my'], aspect='auto', origin='lower', cmap='magma')
        UIColors.setup_axis(self.ax_p1_2, "2. LIBROSA МЕЛ-СПЕКТР", "Время (сек)", "Гц")
        librosa.display.specshow(res['features']['mel_lib'], ax=self.ax_p1_2, sr=sr, x_axis='time', y_axis='mel')
        UIColors.setup_axis(self.ax_p1_3, "3. LIBROSA STFT (ЛОГ)", "Время (сек)", "Гц")
        librosa.display.specshow(res['features']['stft_lib'], ax=self.ax_p1_3, sr=sr, x_axis='time', y_axis='log')
        UIColors.setup_axis(self.ax_p2_1, "4. MFCC КОЭФФИЦИЕНТЫ", "Индекс", "Значение")
        self.ax_p2_1.bar(range(len(res['features']['mfcc'])), res['features']['mfcc'], color=UIColors.TEXT_ACCENT)
        UIColors.setup_axis(self.ax_p2_2, f"5. ВХОД (ШУМ: {res['metrics']['snr_in']:.1f} дБ)", "Время (сек)", "Гц")
        librosa.display.specshow(res['features']['spec_noisy'], ax=self.ax_p2_2, sr=sr, x_axis='time', y_axis='mel')
        UIColors.setup_axis(self.ax_p2_3, f"6. ВЫХОД (ОЧИЩЕН: {res['metrics']['snr_out']:.1f} дБ)", "Время (сек)", "Гц")
        librosa.display.specshow(res['features']['spec_enh'], ax=self.ax_p2_3, sr=sr, x_axis='time', y_axis='mel')
        
        f, m = res['features'], res['metrics']
        
        res_report = (f"РЕЗУЛЬТАТЫ (SNR {snr_val}дБ):\n"
                      f"----------------------\n"
                      f"Усиление: {m['snr_out']-m['snr_in']:>6.2f} дБ\n"
                      f"PESQ:     {m['pesq']:>6.2f}\n"
                      f"SI-SDR:   {m['si_sdr_out']:>6.2f} дБ")
        
        feat_report = (f"ПРИЗНАКИ:\n"
                       f"----------------------\n"
                       f"Rolloff:  {f['rolloff']:>7.0f} Гц\n"
                       f"Центроид: {f['centroid']:>7.0f} Гц\n"
                       f"ZCR:      {f['zcr']:>9.4f}")
        
        self.info_text_res.set_text(res_report)
        self.info_text_feat.set_text(feat_report)
        self.set_page(self.current_page)

    def set_status(self, msg, color=None):
        self.status_text.set_text(f"● {msg}")
        if color: self.status_text.set_color(color)
        self.fig.canvas.draw_idle()
