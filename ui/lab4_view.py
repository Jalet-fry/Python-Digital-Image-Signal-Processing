import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider
from core.utils.themes import UIColors
import librosa.display
import os

class CustomRadioButtons:
    def __init__(self, ax, labels, activecolor, title=""):
        self.ax = ax
        self.all_labels = labels
        self.activecolor = activecolor
        self.value_selected = labels[0] if labels else None
        self.on_clicked_cbs = []
        self.page_size = 7
        self.current_page = 0
        self.total_pages = (len(labels) - 1) // self.page_size + 1 if labels else 1
        self.title = title
        
        pos = ax.get_position()
        self.ax_prev = plt.axes([pos.x0 + 0.01, pos.y1 + 0.005, 0.03, 0.02])
        self.ax_next = plt.axes([pos.x0 + 0.045, pos.y1 + 0.005, 0.03, 0.02])
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
        self.ax.set_title(f"{self.title} ({self.current_page+1}/{self.total_pages})", 
                         color='white', fontsize=8, pad=10)
        for spine in self.ax.spines.values():
            spine.set_visible(True); spine.set_color('#30363D')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.all_labels))
        page_labels = self.all_labels[start_idx:end_idx]
        
        step = 0.13
        start_y = 0.85
        for i, text in enumerate(page_labels):
            y = start_y - i * step
            is_active = (text == self.value_selected)
            circle = plt.Circle((0.1, y), 0.03, transform=self.ax.transAxes,
                                edgecolor='white', facecolor=self.activecolor if is_active else 'none')
            self.ax.add_patch(circle)
            display_text = text if len(text) < 18 else text[:15] + "..."
            self.ax.text(0.22, y, display_text, transform=self.ax.transAxes,
                         color='white', fontsize=7, va='center')
        self.ax.figure.canvas.draw_idle()

    def _prev_page(self, event):
        if self.current_page > 0: self.current_page -= 1; self._setup_ui()
    def _next_page(self, event):
        if self.current_page < self.total_pages - 1: self.current_page += 1; self._setup_ui()
    def _on_click(self, event):
        if event.inaxes != self.ax: return
        idx_on_page = int(round((0.85 - event.ydata) / 0.13))
        global_idx = self.current_page * self.page_size + idx_on_page
        if 0 <= global_idx < len(self.all_labels):
            self.value_selected = self.all_labels[global_idx]
            self._setup_ui()
            for cb in self.on_clicked_cbs: cb(self.value_selected)
    def on_clicked(self, func): self.on_clicked_cbs.append(func)

class Lab4View:
    def __init__(self, fig, wav_files):
        self.fig = fig
        UIColors.apply_style(plt)
        fig.patch.set_facecolor(UIColors.BG_DARK)
        
        self.ax_list_src = plt.axes([0.02, 0.70, 0.21, 0.22])
        self.radio_src = CustomRadioButtons(self.ax_list_src, wav_files, UIColors.RADIO_ACTIVE, "SOURCE")
        
        self.ax_list_tgt = plt.axes([0.25, 0.70, 0.21, 0.22])
        self.radio_tgt = CustomRadioButtons(self.ax_list_tgt, wav_files, UIColors.RADIO_ACTIVE, "TARGET")
        
        self.ax_k = plt.axes([0.05, 0.64, 0.38, 0.015], facecolor=UIColors.BG_PANEL)
        self.k_slider = Slider(self.ax_k, '', 1, 12, valinit=4, valstep=1, color=UIColors.TEXT_ACCENT)
        self.k_label = fig.text(0.24, 0.66, "k-Neighbors: 4", ha='center', weight='bold', color=UIColors.TEXT_ACCENT, fontsize=9)
        
        self.btn_run_vc = Button(plt.axes([0.02, 0.54, 0.44, 0.07]), 'CONVERT VOICE (kNN-VC)', color=UIColors.BTN_RUN)
        
        self.btn_play_src = Button(plt.axes([0.02, 0.49, 0.14, 0.035]), 'SRC', color='#2D3748')
        self.btn_play_tgt = Button(plt.axes([0.17, 0.49, 0.14, 0.035]), 'TGT', color='#2D3748')
        self.btn_play_res = Button(plt.axes([0.32, 0.49, 0.14, 0.035]), 'RESULT', color='#388E3C')
        
        self.ax_box = plt.axes([0.02, 0.42, 0.34, 0.035], facecolor='#161B22')
        self.text_box = TextBox(self.ax_box, '', initial="Hello BSUIR", color='#21262D', hovercolor='#30363D')
        self.text_box.label.set_color('white')
        self.text_box.text_disp.set_color('white')
        self.btn_run_tts = Button(plt.axes([0.37, 0.42, 0.09, 0.035]), 'TTS', color='#238636')
        
        self.ax_info = plt.axes([0.02, 0.08, 0.44, 0.30], facecolor='#0D1117')
        for spine in self.ax_info.spines.values(): spine.set_visible(True); spine.set_color('#30363D')
        self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        self.info_text = self.ax_info.text(0.02, 0.95, "Ready for processing...", 
                                         va='top', fontsize=8, color='white', family='monospace')
        
        self.ax_src = plt.axes([0.52, 0.68, 0.20, 0.24])
        self.ax_tgt = plt.axes([0.76, 0.68, 0.20, 0.24])
        self.ax_sim = plt.axes([0.52, 0.08, 0.44, 0.52])
        
        self.status_text = fig.text(0.02, 0.02, "● Ready", color=UIColors.TEXT_ACCENT, weight='bold', fontsize=9)
        self.k_slider.on_changed(lambda v: self.k_label.set_text(f"k-Neighbors: {int(v)}"))

    def clear_gui(self):
        self.ax_src.clear(); self.ax_tgt.clear(); self.ax_sim.clear()
        self.info_text.set_text("Processing... Please wait.")
        self.fig.canvas.draw_idle()

    def set_status(self, text, color="white"):
        self.status_text.set_text(f"● {text}")
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()

    def update_plots(self, src_wav, tgt_wav, sim_matrix=None):
        self.ax_src.clear(); self.ax_tgt.clear(); self.ax_sim.clear()
        
        UIColors.setup_axis(self.ax_src, "SOURCE MEL", "", "")
        if src_wav is not None:
            S = librosa.feature.melspectrogram(y=src_wav, sr=16000)
            librosa.display.specshow(librosa.power_to_db(S), ax=self.ax_src, cmap='magma')
            
        UIColors.setup_axis(self.ax_tgt, "TARGET MEL", "", "")
        if tgt_wav is not None:
            S = librosa.feature.melspectrogram(y=tgt_wav, sr=16000)
            librosa.display.specshow(librosa.power_to_db(S), ax=self.ax_tgt, cmap='magma')

        UIColors.setup_axis(self.ax_sim, "SIMILARITY MATRIX")
        if sim_matrix is not None:
            data = sim_matrix.cpu().numpy()
            im = self.ax_sim.imshow(data, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im, ax=self.ax_sim, fraction=0.046, pad=0.04)
        self.fig.canvas.draw_idle()
