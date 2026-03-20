import sys
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import numpy as np
import mplcursors
from scipy.io import wavfile
from scipy.signal import fftconvolve

try:
    import sounddevice as sd
except ImportError:
    sd = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.signals.fourier import dft, idft, fft, ifft
from core.signals.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation
from core.signals.generator import generate_instrument_signal
from core.config_variants import get_lab1_config 
from core.utils.aspects import DSPContext

# --- Инициализация ---
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

DSPContext.variant = VARIANT
DSPContext.current_lab = "lab1"

cfg = get_lab1_config(VARIANT)

plt.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-v0_8-muted')

N, sr = cfg.N, cfg.sr
sr_a, dur_a = cfg.sr_audio, cfg.duration_audio

# --- Расчеты ---
t, x_raw = generate_instrument_signal(cfg.x.amplitudes, cfg.x.f0, cfg.x.harmonics, cfg.x.phi, duration=N/sr, sr=sr)
_, y_raw = generate_instrument_signal(cfg.y.amplitudes, cfg.y.f0, cfg.y.harmonics, cfg.y.phi, duration=N/sr, sr=sr)

x_s, y_s, t = x_raw[:N], y_raw[:N], t[:N]
dt = t[1] - t[0]

d_x, f_x, l_fx = dft(x_s), fft(x_s), np.fft.fft(x_s)
d_y, f_y, l_fy = dft(y_s), fft(y_s), np.fft.fft(y_s)

id_x, if_x = idft(d_x).real, ifft(f_x).real[:N]
id_y, if_y = idft(d_y).real, ifft(f_y).real[:N]

c_m, c_f, c_l = linear_convolution(x_s, y_s), fft_convolution(x_s, y_s).real, np.convolve(x_s, y_s)
cr_m, cr_f, cr_l = correlation(x_s, y_s), fft_correlation(x_s, y_s).real, np.correlate(x_s, y_s, mode='full')

_, x_audio = generate_instrument_signal(cfg.x.amplitudes, cfg.x.f0, cfg.x.harmonics, cfg.x.phi, duration=dur_a, sr=sr_a)
_, y_audio = generate_instrument_signal(cfg.y.amplitudes, cfg.y.f0, cfg.y.harmonics, cfg.y.phi, duration=dur_a, sr=sr_a)

# --- Визуализация ---
def get_clean(data, max_f=600):
    mag = np.abs(data) / (N / 2)
    ph = np.angle(data)
    ph[mag < 0.001 * np.max(mag)] = 0
    freqs = np.fft.fftfreq(N, d=dt)
    idx = np.where(freqs[:N//2] <= max_f)[0][-1]
    return freqs[:idx], mag[:idx], ph[:idx]

f_dx, m_dx, p_dx = get_clean(d_x); f_fx, m_fx, p_fx = get_clean(f_x); f_lx, m_lx, p_lx = get_clean(l_fx)
f_dy, m_dy, p_dy = get_clean(d_y); f_fy, m_fy, p_fy = get_clean(f_y); f_ly, m_ly, p_ly = get_clean(l_fy)

errors = [
    np.max(np.abs(x_s - id_x)), 
    np.max(np.abs(x_s - if_x)), 
    np.max(np.abs(x_s - np.fft.ifft(l_fx).real)), 
    np.max(np.abs(c_m - c_l)), 
    np.max(np.abs(cr_m - cr_l))
]

plots_data = [
    (t*1000, x_s, "x(t) Исходный", "plot", "blue", "ms"),
    (t*1000, y_s, "y(t) Исходный", "plot", "orange", "ms"),
    (f_dx, m_dx, "x(t) ДПФ: амплитуда", "stem", "blue", "Hz"),
    (f_dx, p_dx, "x(t) ДПФ: фаза", "stem", "blue", "Hz"),
    (t*1000, id_x, "x(t) Восст. ОДПФ", "plot", "green", "ms"),
    (f_fx, m_fx, "x(t) БПФ: амплитуда", "stem", "blue", "Hz"),
    (f_fx, p_fx, "x(t) БПФ: фаза", "stem", "blue", "Hz"),
    (t*1000, if_x, "x(t) Восст. ОБПФ", "plot", "purple", "ms"),
    (f_dy, m_dy, "y(t) ДПФ: амплитуда", "stem", "orange", "Hz"),
    (f_dy, p_dy, "y(t) ДПФ: фаза", "stem", "orange", "Hz"),
    (t*1000, id_y, "y(t) Восст. ОДПФ", "plot", "red", "ms"),
    (f_fy, m_fy, "y(t) БПФ: амплитуда", "stem", "orange", "Hz"),
    (f_fy, p_fy, "y(t) БПФ: фаза", "stem", "orange", "Hz"),
    (t*1000, if_y, "y(t) Восст. ОБПФ", "plot", "brown", "ms"),
    (range(len(c_m)), c_m, "x*y Свертка (Лин)", "plot", "blue", "pts"),
    (range(len(c_f)), c_f, "x*y Свертка (FFT)", "plot", "cyan", "idx"),
    (range(len(cr_m)), cr_m, "x&y Корр (Лин)", "plot", "gray", "idx"),
    (range(len(cr_f)), cr_f, "x&y Корр (FFT)", "plot", "black", "idx"),
    (f_lx, m_lx, "x(t) Спектр (Lib)", "stem", "blue", "Hz"),
    (f_lx, p_lx, "x(t) Фаза (Lib)", "stem", "blue", "Hz"),
    (f_ly, m_ly, "y(t) Спектр (Lib)", "stem", "orange", "Hz"),
    (f_ly, p_ly, "y(t) Фаза (Lib)", "stem", "orange", "Hz"),
    (range(len(c_l)), c_l, "x*y Свертка (Lib)", "plot", "green", "idx"),
    (range(len(cr_l)), cr_l, "x&y Корр (Lib)", "plot", "red", "idx"),
]

group_mapping = [
    [0, 4, 7], [1, 10, 13], [2, 5, 18], [3, 6, 19], 
    [8, 11, 20], [9, 12, 21], [14, 15, 22], [16, 17, 23], []
]

fig = plt.figure(figsize=(15, 10))
fig.canvas.manager.set_window_title(f'Лабораторная работа №1 - Вариант {VARIANT} ({cfg.x.name})')

ax_menu = plt.axes([0.02, 0.45, 0.16, 0.4], facecolor='#f0f0f0')
menu_labels = [
    '1. Восстановление X', '2. Восстановление Y', '3. Спектр Амп. X', '4. Спектр Фаз. X', 
    '5. Спектр Амп. Y', '6. Спектр Фаз. Y', '7. Свертка X * Y', '8. Корреляция X & Y', '9. Гистограмма ошибок'
]
radio = RadioButtons(ax_menu, menu_labels, active=0, activecolor='royalblue')

ax_play = plt.axes([0.02, 0.3, 0.16, 0.06]); btn_play = Button(ax_play, '▶ Play Audio', color='lightgreen', hovercolor='lime')
ax_save_wav = plt.axes([0.02, 0.22, 0.16, 0.06]); btn_save_wav = Button(ax_save_wav, '💾 Save WAV', color='lightblue', hovercolor='deepskyblue')
ax_save_graphs = plt.axes([0.02, 0.14, 0.16, 0.06]); btn_save_graphs = Button(ax_save_graphs, '📊 Save Results', color='peachpuff', hovercolor='orange')

status_text = fig.text(0.02, 0.02, f"Ready. Variant {VARIANT}", fontsize=10, color='darkblue', fontweight='bold')

def set_status(msg, color='darkblue'):
    status_text.set_text(msg); status_text.set_color(color); fig.canvas.draw_idle()

def play_audio(event):
    if sd is None: return
    set_status("Processing audio...", "orange"); sd.stop()
    label = radio.value_selected
    sig = x_audio if 'X' in label else y_audio
    if 'Свертка' in label: sig = fftconvolve(x_audio, y_audio, mode='full')
    sd.play(sig / (np.max(np.abs(sig)) + 1e-9), sr_a)
    set_status(f"Playing: {label}", "green")

def save_wav_files(event):
    set_status("Saving WAV...", "orange")
    audio_dir = os.path.join(BASE_DIR, "results", "audio"); os.makedirs(audio_dir, exist_ok=True)
    for n, s in [(f"x_{cfg.x.name}", x_audio), (f"y_{cfg.y.name}", y_audio)]:
        wavfile.write(os.path.join(audio_dir, f"{n}.wav"), sr_a, np.int16(s/np.max(np.abs(s)) * 32767))
    set_status("WAV files saved!", "darkgreen")

def save_results(event):
    set_status("Saving graph...", "orange")
    plots_dir = os.path.join(BASE_DIR, "results", "graphs"); os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"var{VARIANT}_{radio.value_selected[:2]}.png"), dpi=150)
    set_status("Graph saved!", "darkgreen")

btn_play.on_clicked(play_audio); btn_save_wav.on_clicked(save_wav_files); btn_save_graphs.on_clicked(save_results)

main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
plt.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.08, hspace=0.35)
current_cursor = None

def update_plots(label):
    global current_cursor
    set_status(f"Selected: {label}")
    if 'Гистограмма' in label:
        for i, ax in enumerate(main_axes):
            ax.clear()
            if i == 0:
                ax.bar(['DFT', 'FFT', 'Lib', 'Conv', 'Corr'], errors, color=['red', 'orange', 'green', 'blue', 'purple'])
                ax.set_yscale('log'); ax.set_title("Погрешность реализации")
                for j, v in enumerate(errors): ax.text(j, v, f"{v:.1e}", ha='center', va='bottom')
            else: ax.axis('off')
    else:
        indices = group_mapping[menu_labels.index(label)]
        for i, p_idx in enumerate(indices):
            ax = main_axes[i]; ax.set_axis_on(); ax.clear()
            x_data, y_data, title, p_type, color, unit = plots_data[p_idx]
            ax.set_title(f"График №{p_idx + 1}: {title}", fontsize=11, fontweight='bold')
            if p_type == "plot": ax.plot(x_data, y_data, color=color, lw=1.5)
            else: ax.stem(x_data, y_data, linefmt=color, markerfmt='o', basefmt=" ")
            ax.set_xlabel(unit); ax.grid(True, alpha=0.3)
    
    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor(main_axes, hover=True)
    fig.canvas.draw_idle()

radio.on_clicked(update_plots); update_plots(menu_labels[0]); plt.show()
