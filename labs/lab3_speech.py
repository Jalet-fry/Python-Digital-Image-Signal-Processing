import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
import sounddevice as sd
from scipy.io import wavfile
import librosa

# Добавляем корень проекта в путь
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.utils.themes import UIColors
from core.signals.speech_processor import SpeechProcessor
from core.config_variants import get_lab3_config
from core.utils.aspects import DSPContext

# Попытка инициализации DeepFilterNet
DF_AVAILABLE = False
df_model, df_state = None, None
try:
    import torch
    from df.enhance import init_df
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_model, df_state, _ = init_df()
    df_model = df_model.to(device)
    DF_AVAILABLE = True
    print(f">>> [OK] DeepFilterNet Loaded on {device}")
except Exception as e:
    device = 'cpu'
    print(f">>> [WARN] DeepFilterNet not available: {e}")

# ==========================================================
# 1. ИНИЦИАЛИЗАЦИЯ
# ==========================================================
VARIANT = 10 # Можно брать из sys.argv
DSPContext.variant = VARIANT
DSPContext.current_lab = "lab3"
cfg = get_lab3_config(VARIANT)

processor = SpeechProcessor(df_model, df_state, device)

SOURCE_DIR = os.path.join(BASE_DIR, "source_audio_lab3")
audio_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.wav')]) if os.path.exists(SOURCE_DIR) else []
ui_file_labels = [f[:20] for f in audio_files] or ["No files found"]
file_map = dict(zip(ui_file_labels, audio_files))

# ==========================================================
# 2. UI КЛАСС
# ==========================================================
class SpeechLabUI:
    def __init__(self, fig):
        self.fig = fig
        self.ax1 = fig.add_subplot(2, 2, 1)
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 3)
        self.ax4 = fig.add_subplot(2, 2, 4)
        
        plt.subplots_adjust(left=0.25, bottom=0.15, hspace=0.3, wspace=0.25)
        
        # Виджеты
        self.ax_files = plt.axes([0.02, 0.55, 0.18, 0.35], facecolor=UIColors.BG_PANEL)
        self.radio = RadioButtons(self.ax_files, ui_file_labels, activecolor=UIColors.RADIO_ACTIVE)
        
        self.ax_snr = plt.axes([0.04, 0.48, 0.14, 0.02])
        self.snr_slider = Slider(self.ax_snr, 'SNR', -10, 30, valinit=10, valstep=5)
        
        self.ax_run = plt.axes([0.02, 0.40, 0.18, 0.05])
        self.btn_run = Button(self.ax_run, 'ОБРАБОТАТЬ', color=UIColors.BTN_RUN)
        self.btn_run.on_clicked(self.on_run)
        
        self.ax_play_orig = plt.axes([0.02, 0.32, 0.08, 0.04])
        self.btn_orig = Button(self.ax_play_orig, 'Ориг.', color=UIColors.BTN_PLAY)
        self.btn_orig.on_clicked(lambda e: self.play('clean'))
        
        self.ax_play_proc = plt.axes([0.12, 0.32, 0.08, 0.04])
        self.btn_proc = Button(self.ax_play_proc, 'Результ.', color=UIColors.BTN_PLAY)
        self.btn_proc.on_clicked(lambda e: self.play('enhanced'))

        self.status_text = fig.text(0.02, 0.02, "Готов", color="white")
        self.res = None

    def on_run(self, event):
        fname = file_map[self.radio.value_selected]
        path = os.path.join(SOURCE_DIR, fname)
        self.status_text.set_text(f"Обработка {fname}...")
        self.fig.canvas.draw_idle()
        
        self.res = processor.process(path, self.snr_slider.val, use_df=DF_AVAILABLE)
        self.update_plots()
        self.status_text.set_text("Готово. Проверьте метрики.")

    def play(self, key):
        if self.res and key in self.res:
            sd.stop()
            sd.play(self.res[key], self.res['sr'])

    def update_plots(self):
        if not self.res: return
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]: ax.clear()
        
        # 1. Сравнение спектрограмм (Пункт 2 задания)
        self.ax1.imshow(self.res['features']['spec_my'], aspect='auto', origin='lower')
        self.ax1.set_title("Своя Мел-спектрограмма")
        
        self.ax2.imshow(self.res['features']['spec_lib'], aspect='auto', origin='lower')
        self.ax2.set_title("Librosa Мел-спектрограмма")
        
        # 2. Признаки (Пункт 3 задания)
        f = self.res['features']
        self.ax3.plot(f['mfcc_my'], 'ro-', label='MFCC (Manual)')
        self.ax3.set_title(f"MFCC | Centroid: {f['centroid_my']:.0f} Hz")
        self.ax3.legend()
        
        # 3. Таблица метрик (Пункт 5 задания)
        m = self.res['metrics']
        metric_str = (
            f"ОЦЕНКА КАЧЕСТВА (SNR={self.snr_slider.val}):\n\n"
            f"SNR In:  {m['snr_in']:.2f} dB\n"
            f"SNR Out: {m['snr_out']:.2f} dB\n"
            f"SI-SDR:  {m['si_sdr']:.2f} dB\n"
            f"PESQ (Manual): {m['pesq_my']:.2f}\n\n"
            f"ZCR: {f['zcr_my']:.4f}\n"
            f"Rolloff: {f['rolloff_my']:.0f} Hz"
        )
        self.ax4.text(0.1, 0.5, metric_str, color='white', fontsize=11, family='monospace', va='center')
        self.ax4.axis('off')
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor=UIColors.BG_DARK)
    app = SpeechLabUI(fig)
    plt.show()
