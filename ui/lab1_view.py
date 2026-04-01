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
        
        self.main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
        plt.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.08, hspace=0.55)
        
        # Индексы графиков в массиве plots_data
        self.menu_groups = {
            '1. Восстановление X': [0, 4, 7],
            '2. Восстановление Y': [1, 10, 13],
            '3. Спектр Ампл. X': [2, 5, 18],
            '4. Спектр Фаза X': [3, 6, 19],
            '5. Спектр Ампл. Y': [8, 11, 20],
            '6. Спектр Фаза Y': [9, 12, 21],
            '7. Свертка X * Y': [14, 15, 22],
            '8. Корреляция X & Y': [16, 17, 23],
            '9. Гистограмма ошибок': []
        }
        self.menu_labels = list(self.menu_groups.keys())
        
        # Панель меню
        self.ax_menu = plt.axes([0.02, 0.45, 0.18, 0.4], facecolor=UIColors.BG_PANEL)
        self.radio = RadioButtons(self.ax_menu, self.menu_labels, active=0, activecolor=UIColors.RADIO_ACTIVE)
        for lbl in self.radio.labels:
            lbl.set_color(UIColors.TEXT_MAIN)
            lbl.set_fontsize(8)
        
        # Кнопки (без эмодзи для совместимости)
        self.ax_play = plt.axes([0.02, 0.35, 0.18, 0.05])
        self.btn_play = Button(self.ax_play, 'PLAY AUDIO', color=UIColors.BTN_PLAY)
        self.btn_play.label.set_color(UIColors.TEXT_MAIN)
        
        self.ax_save_wav = plt.axes([0.02, 0.28, 0.18, 0.05])
        self.btn_save_wav = Button(self.ax_save_wav, 'SAVE WAV', color='#1e3a8a')
        self.btn_save_wav.label.set_color(UIColors.TEXT_MAIN)
        
        self.ax_save_res = plt.axes([0.02, 0.21, 0.18, 0.05])
        self.btn_save_res = Button(self.ax_save_res, 'SAVE GRAPH', color='#1e3a8a')
        self.btn_save_res.label.set_color(UIColors.TEXT_MAIN)
        
        self.status_text = fig.text(0.02, 0.02, f"Ready | Variant {cfg.variant}", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=9)
        self.current_cursor = None

    def update_plots(self, label, res):
        if res is None: return
        for ax in self.main_axes: 
            ax.clear()
            ax.set_axis_on()
        
        if 'Гистограмма' in label:
            ax = self.main_axes[0]
            errs = res['errors']
            names = ['DFT', 'FFT', 'Lib', 'Conv', 'Corr']
            ax.bar(names, errs, color=['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'])
            ax.set_yscale('log')
            UIColors.setup_axis(ax, "Погрешность восстановления (Log Scale)")
            for i, v in enumerate(errs):
                ax.text(i, v, f"{v:.1e}", color=UIColors.TEXT_MAIN, ha='center', va='bottom', fontsize=8)
            self.main_axes[1].axis('off')
            self.main_axes[2].axis('off')
        else:
            indices = self.menu_groups[label]
            for i, p_idx in enumerate(indices):
                x_axis, y_data, title, p_type, color, unit = res['plots_data'][p_idx]
                is_stem = (p_type == "stem")
                self._draw(i, x_axis, y_data, f"График {p_idx + 1}: {title}", color, unit, "Ампл.", is_stem)

        if self.current_cursor: self.current_cursor.remove()
        self.current_cursor = mplcursors.cursor(self.main_axes, hover=True)
        self.fig.canvas.draw_idle()

    def _draw(self, ax_idx, x, y, title, color, xl, yl, is_stem=False):
        ax = self.main_axes[ax_idx]
        UIColors.setup_axis(ax, title, xl, yl)
        if is_stem:
            ax.stem(x, y, linefmt=color, markerfmt='o', basefmt=" ")
        else:
            ax.plot(x, y, color=color, lw=1.5)

    def set_status(self, msg, color=None):
        self.status_text.set_text(f"● {msg}")
        if color: self.status_text.set_color(color)
        self.fig.canvas.draw_idle()
