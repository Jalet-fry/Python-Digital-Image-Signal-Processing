import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import mplcursors
from core.utils.themes import UIColors

class Lab1View:
    def __init__(self, fig, cfg):
        self.fig = fig
        UIColors.apply_style(plt)
        
        self.main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
        plt.subplots_adjust(left=0.22, right=0.96, top=0.92, bottom=0.1, hspace=0.6)
        
        # Группировка 24 графиков в 9 меню (8 групп + 1 ошибки)
        self.menu_groups = {
            '1. Восстановление X (1,5,8)': ['01', '05', '08'],
            '2. Восстановление Y (2,11,14)': ['02', '11', '14'],
            '3. Спектр Ампл. X (3,6,19)': ['03', '06', '19'],
            '4. Спектр Фаза X (4,7,20)': ['04', '07', '20'],
            '5. Спектр Ампл. Y (9,12,21)': ['09', '12', '21'],
            '6. Спектр Фаза Y (10,13,22)': ['10', '13', '22'],
            '7. Свертка (15,16,23)': ['15', '16', '23'],
            '8. Корреляция (17,18,24)': ['17', '18', '24'],
            '9. Гистограмма ошибок': []
        }
        self.menu_labels = list(self.menu_groups.keys())
        
        self.ax_menu = plt.axes([0.02, 0.45, 0.18, 0.4], facecolor=UIColors.BG_PANEL)
        self.radio = RadioButtons(self.ax_menu, self.menu_labels, active=0, activecolor=UIColors.RADIO_ACTIVE)
        for lbl in self.radio.labels:
            lbl.set_color(UIColors.TEXT_MAIN)
            lbl.set_fontsize(8)
        
        self.ax_play = plt.axes([0.02, 0.35, 0.18, 0.05])
        self.btn_play = Button(self.ax_play, '▶ PLAY AUDIO', color=UIColors.BTN_PLAY)
        self.btn_play.label.set_color(UIColors.TEXT_MAIN)
        
        self.status_text = fig.text(0.02, 0.02, f"Var {cfg.variant} | Ready", color=UIColors.TEXT_ACCENT, weight='bold')
        self.current_cursor = None

    def update_plots(self, label, res):
        if res is None: return
        for ax in self.main_axes: 
            ax.clear()
            ax.set_axis_on()
        
        if 'Гистограмма' in label:
            ax = self.main_axes[0]
            errs = res['errors']
            names = ['DFT Error', 'FFT Error']
            ax.bar(names, errs, color=[UIColors.LAB1['err'], UIColors.LAB1['rec']])
            ax.set_yscale('log')
            UIColors.setup_axis(ax, "Погрешность восстановления (Log Scale)")
            for i, v in enumerate(errs):
                ax.text(i, v, f"{v:.1e}", color=UIColors.TEXT_MAIN, ha='center', va='bottom')
            self.main_axes[1].axis('off')
            self.main_axes[2].axis('off')
        else:
            keys = self.menu_groups[label]
            c_map = UIColors.LAB1
            
            for i, key in enumerate(keys):
                data_tuple = res['data'][key]
                y_data, title, x_label = data_tuple
                
                # Определяем ось X
                if "Частота" in x_label: x_axis = res['f_axis']
                elif "мс" in x_label: x_axis = res['t']
                elif "n" in x_label: x_axis = range(len(y_data))
                else: x_axis = range(len(y_data))

                # Определяем стиль (stem для спектров)
                is_stem = "Спектр" in title or "Lib Ампл" in title or "Lib Фаза" in title
                
                # Цвет (X - синий, Y - бирюзовый, восст - розовый)
                color = c_map['x'] if 'x(t)' in title else c_map['y']
                if 'Восст' in title: color = c_map['rec']
                
                self._draw(i, x_axis, y_data, title, color, x_label, "Ампл.", is_stem)

        if self.current_cursor: self.current_cursor.remove()
        self.current_cursor = mplcursors.cursor(self.main_axes, hover=True)
        self.fig.canvas.draw_idle()

    def _draw(self, ax_idx, x, y, title, color, xl, yl, is_stem=False):
        ax = self.main_axes[ax_idx]
        UIColors.setup_axis(ax, title, xl, yl)
        if is_stem:
            # Обрезаем данные под f_axis если нужно
            x_len = len(x)
            ax.stem(x, y[:x_len], linefmt=color, markerfmt='o', basefmt=" ")
        else:
            ax.plot(x, y, color=color, lw=1.2)

    def set_status(self, msg):
        self.status_text.set_text(f"● {msg}")
        self.fig.canvas.draw_idle()
