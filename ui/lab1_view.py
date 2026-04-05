import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import mplcursors
from core.utils.themes import UIColors

class Lab1View:
    def __init__(self, fig, cfg):
        self.fig = fig
        self.cfg = cfg
        UIColors.apply_style(plt)
        plt.rcParams['toolbar'] = 'None'
        
        # Основные оси
        self.main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
        
        # left=0.45 (было 0.42), bottom=0.18 (было 0.15)
        plt.subplots_adjust(left=0.45, right=0.95, top=0.90, bottom=0.18, hspace=0.7)
        
        self.menu_groups = {
            '1. Восстановление X': [0, 4, 7],
            '2. Восстановление Y': [1, 10, 13],
            '3. Спектр Амп. X': [2, 5, 18],
            '4. Спектр Фаз. X': [3, 6, 19],
            '5. Спектр Амп. Y': [8, 11, 20],
            '6. Спектр Фаз. Y': [9, 12, 21],
            '7. Свертка X * Y': [14, 15, 22],
            '8. Корреляция X & Y': [16, 17, 23],
            '9. Гистограмма ошибок': []
        }
        self.menu_labels = list(self.menu_groups.keys())
        
        self.ax_menu = plt.axes([0.02, 0.45, 0.35, 0.4], facecolor=UIColors.BG_PANEL)
        self.radio = RadioButtons(self.ax_menu, self.menu_labels, active=0, activecolor=UIColors.RADIO_ACTIVE)
        for lbl in self.radio.labels:
            lbl.set_color(UIColors.TEXT_MAIN)
            lbl.set_fontsize(9)
            lbl.set_weight('bold')
        
        self.ax_play = plt.axes([0.02, 0.3, 0.35, 0.06])
        self.btn_play = Button(self.ax_play, 'PLAY AUDIO', color=UIColors.BTN_PLAY)
        self.btn_play.label.set_color('white')
        
        self.ax_save_wav = plt.axes([0.02, 0.22, 0.35, 0.06])
        self.btn_save_wav = Button(self.ax_save_wav, 'SAVE WAV', color=UIColors.BTN_PLAY)
        self.btn_save_wav.label.set_color('white')
        
        self.ax_save_res = plt.axes([0.02, 0.14, 0.35, 0.06])
        self.btn_save_res = Button(self.ax_save_res, 'SAVE RESULTS', color=UIColors.BTN_RUN)
        self.btn_save_res.label.set_color('black')
        
        self.status_text = fig.text(0.02, 0.02, f"Ready | Variant {cfg.variant}", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=12)
        self.current_cursor = None

    def update_plots(self, label, res):
        if res is None: return
        for ax in self.main_axes: 
            ax.clear()
            ax.set_axis_on()
        
        if 'Гистограмма' in label:
            ax = self.main_axes[0]
            errs = res['errors']
            ax.bar(['DFT', 'FFT', 'Lib', 'Conv', 'Corr'], errs, color=[UIColors.SIG_X, UIColors.SIG_Y, UIColors.SIG_REC, UIColors.SIG_ERR, '#F472B6'])
            ax.set_yscale('log')
            UIColors.setup_axis(ax, "ПОГРЕШНОСТЬ РЕАЛИЗАЦИИ", "Алгоритм", "Ошибка")
            self.main_axes[1].axis('off'); self.main_axes[2].axis('off')
        else:
            indices = self.menu_groups.get(label, [])
            for i, p_idx in enumerate(indices):
                x, y, title, p_type, color, unit = res['plots_data'][p_idx]
                y_label = "Ампл." if "Фаза" not in title else "Рад."
                UIColors.setup_axis(self.main_axes[i], f"ГРАФИК №{p_idx+1}: {title}", unit, y_label)
                if p_type == "plot": self.main_axes[i].plot(x, y, color=color, lw=2)
                else: self.main_axes[i].stem(x, y, linefmt=color, markerfmt='o', basefmt=" ")

        if self.current_cursor: self.current_cursor.remove()
        self.current_cursor = mplcursors.cursor(self.main_axes, hover=True)
        self.fig.canvas.draw_idle()

    def set_status(self, msg, color=None):
        self.status_text.set_text(f"● {msg}")
        self.status_text.set_color(color if color else UIColors.TEXT_ACCENT)
        self.fig.canvas.draw_idle()
