import sys
import os
import time
import shutil
import argparse  # 1. Добавлен импорт
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

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.signals.fourier import dft, idft, fft, ifft
from core.signals.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation
from core.signals.generator import generate_instrument_signal
from core.config_variants import get_lab_config  # 2. Добавлен импорт конфига
from core.utils.aspects import DSPContext # 3. Добавлен импорт аспекта для логов

# ==========================================================
# 0. ЧТЕНИЕ ВАРИАНТА ИЗ АРГУМЕНТОВ (Универсальность)
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

# Активация контекста логов (AOP)
DSPContext.variant = VARIANT
DSPContext.current_lab = "lab1"

# Получаем данные динамически
cfg = get_lab_config(1, VARIANT)

# Настройка стиля
plt.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-v0_8-muted')

# ==========================================================
# 1. ПАРАМЕТРЫ (ЗАМЕНЕНЫ НА ЗНАЧЕНИЯ ИЗ CFG)
# ==========================================================
config = {
    'x': cfg['x'], # Теперь берется из выбранного варианта
    'y': cfg['y'], # Теперь берется из выбранного варианта
    'N': 1024,
    'sr': 8000,
    'sr_audio': 44100,
    'duration_audio': 3.0
}

N, sr = config['N'], config['sr']
sr_a, dur_a = config['sr_audio'], config['duration_audio']

# ==========================================================
# 2. РАСЧЕТЫ (БЕЗ ИЗМЕНЕНИЙ ЛОГИКИ)
# ==========================================================
t, x_raw = generate_instrument_signal(config['x']['A'], config['x']['f0'], config['x']['h'], config['x']['phi'], duration=N/sr, sr=sr)
_, y_raw = generate_instrument_signal(config['y']['A'], config['y']['f0'], config['y']['h'], config['y']['phi'], duration=N/sr, sr=sr)

x_s, y_s, t = x_raw[:N], y_raw[:N], t[:N]
dt = t[1] - t[0]

# Прямое/Обратное Фурье
d_x, f_x, l_fx = dft(x_s), fft(x_s), np.fft.fft(x_s)
d_y, f_y, l_fy = dft(y_s), fft(y_s), np.fft.fft(y_s)
id_x, if_x = idft(d_x).real, ifft(f_x).real[:N]
id_y, if_y = idft(d_y).real, ifft(f_y).real[:N]

# Операции
c_m, c_f, c_l = linear_convolution(x_s, y_s), fft_convolution(x_s, y_s).real, np.convolve(x_s, y_s)
cr_m, cr_f, cr_l = correlation(x_s, y_s), fft_correlation(x_s, y_s).real, np.correlate(x_s, y_s, mode='full')

# ==========================================================
# 2.1 ГЕНЕРАЦИЯ ДЛИННОГО ЗВУКА
# ==========================================================
_, x_audio = generate_instrument_signal(config['x']['A'], config['x']['f0'], config['x']['h'], config['x']['phi'], duration=dur_a, sr=sr_a)
_, y_audio = generate_instrument_signal(config['y']['A'], config['y']['f0'], config['y']['h'], config['y']['phi'], duration=dur_a, sr=sr_a)

def get_clean(data, max_f=600):
    mag = np.abs(data) / (N / 2)
    ph = np.angle(data)
    ph[mag < 0.001 * np.max(mag)] = 0
    freqs = np.fft.fftfreq(N, d=dt)
    idx = np.where(freqs[:N//2] <= max_f)[0][-1]
    return freqs[:idx], mag[:idx], ph[:idx]

f_dx, m_dx, p_dx = get_clean(d_x); f_fx, m_fx, p_fx = get_clean(f_x); f_lx, m_lx, p_lx = get_clean(l_fx)
f_dy, m_dy, p_dy = get_clean(d_y); f_fy, m_fy, p_fy = get_clean(f_y); f_ly, m_ly, p_ly = get_clean(l_fy)

errors = [np.max(np.abs(x_s - id_x)), np.max(np.abs(x_s - if_x)), np.max(np.abs(x_s - np.fft.ifft(l_fx).real)), np.max(np.abs(c_m - c_l)), np.max(np.abs(cr_m - cr_l))]

# ==========================================================
# 3. ВСЕ 24 ГРАФИКА (КОНСТРУКЦИЯ ПОЛНОСТЬЮ СОХРАНЕНА)
# ==========================================================
plots_data = [
    (t*1000, x_s, "x(t)", "plot", "blue", "ms"),                                  # 1
    (t*1000, y_s, "y(t)", "plot", "orange", "ms"),                                # 2
    (f_dx, m_dx, "x(t) ДПФ: амплитудный спектр", "stem", "blue", "Hz"),           # 3
    (f_dx, p_dx, "x(t) ДПФ: фазовый спектр", "stem", "blue", "Hz"),               # 4
    (t*1000, id_x, "x(t) ОДПФ", "plot", "green", "ms"),                           # 5
    (f_fx, m_fx, "x(t) БПФ: амплитудный спектр", "stem", "blue", "Hz"),           # 6
    (f_fx, p_fx, "x(t) БПФ: фазовый спектр", "stem", "blue", "Hz"),               # 7
    (t*1000, if_x, "x(t) ОБПФ", "plot", "purple", "ms"),                          # 8
    (f_dy, m_dy, "y(t) ДПФ: амплитудный спектр", "stem", "orange", "Hz"),         # 9
    (f_dy, p_dy, "y(t) ДПФ: фазовый спектр", "stem", "orange", "Hz"),             # 10
    (t*1000, id_y, "y(t) ОДПФ", "plot", "red", "ms"),                             # 11
    (f_fy, m_fy, "y(t) БПФ: амплитудный спектр", "stem", "orange", "Hz"),         # 12
    (f_fy, p_fy, "y(t) БПФ: фазовый спектр", "stem", "orange", "Hz"),             # 13
    (t*1000, if_y, "y(t) ОБПФ", "plot", "brown", "ms"),                           # 14
    (range(len(c_m)), c_m, "x(t) y(t) Свертка", "plot", "blue", "отсчеты"),       # 15
    (range(len(c_f)), c_f, "x(t) y(t) Свертка через БПФ", "plot", "cyan", "idx"), # 16
    (range(len(cr_m)), cr_m, "x(t) y(t) Корреляция", "plot", "gray", "idx"),      # 17
    (range(len(cr_f)), cr_f, "x(t) y(t) Корреляция через БПФ", "plot", "black", "idx"), # 18
    (f_lx, m_lx, "x(t) БПФ: амплитудный спектр (Lib)", "stem", "blue", "Hz"),     # 19
    (f_lx, p_lx, "x(t) БПФ: фазовый спектр (Lib)", "stem", "blue", "Hz"),         # 20
    (f_ly, m_ly, "y(t) БПФ: амплитудный спектр (Lib)", "stem", "orange", "Hz"),   # 21
    (f_ly, p_ly, "y(t) БПФ: фазовый спектр (Lib)", "stem", "orange", "Hz"),       # 22
    (range(len(c_l)), c_l, "x(t) y(t) Свертка (Lib)", "plot", "green", "idx"),    # 23
    (range(len(cr_l)), cr_l, "x(t) y(t) Корреляция (Lib)", "plot", "red", "idx"), # 24
]

group_mapping = [[0, 4, 7], [1, 10, 13], [2, 5, 18], [3, 6, 19], [8, 11, 20], [9, 12, 21], [14, 15, 22], [16, 17, 23]]

# ==========================================================
# 4. ИНТЕРФЕЙС (БЕЗ ИЗМЕНЕНИЙ)
# ==========================================================
fig = plt.figure(figsize=(15, 10))
# Заголовок теперь учитывает номер варианта
fig.canvas.manager.set_window_title(f'Лабораторная работа №1 - Вариант {VARIANT} ({cfg["x"]["name"]})')

ax_menu = plt.axes([0.02, 0.45, 0.16, 0.4], facecolor='#f0f0f0')
menu_labels = ['1. Восстановление X', '2. Восстановление Y', '3. Спектр Амп. X', '4. Спектр Фаз. X', '5. Спектр Амп. Y', '6. Спектр Фаз. Y', '7. Свертка X * Y', '8. Корреляция X & Y', '9. Гистограмма ошибок']
radio = RadioButtons(ax_menu, menu_labels, active=0, activecolor='royalblue')

ax_play = plt.axes([0.02, 0.3, 0.16, 0.06]); btn_play = Button(ax_play, 'Play Audio', color='lightgreen', hovercolor='lime')
ax_save_wav = plt.axes([0.02, 0.22, 0.16, 0.06]); btn_save_wav = Button(ax_save_wav, 'Save WAV', color='lightblue', hovercolor='deepskyblue')
ax_save_graphs = plt.axes([0.02, 0.14, 0.16, 0.06]); btn_save_graphs = Button(ax_save_graphs, 'Save Results', color='peachpuff', hovercolor='orange')

status_text = fig.text(0.02, 0.02, f"Ready. Variant {VARIANT}", fontsize=10, color='darkblue', fontweight='bold')

def set_status(msg, color='darkblue'):
    status_text.set_text(msg)
    status_text.set_color(color)
    fig.canvas.draw_idle()

def get_audio_signal(label):
    if 'X' in label: return x_audio
    if 'Y' in label: return y_audio
    if 'Свертка' in label or '7.' in label: 
        return fftconvolve(x_audio, y_audio, mode='full')[:int(sr_a * dur_a)]
    if 'Корреляция' in label or '8.' in label: 
        return fftconvolve(x_audio, y_audio[::-1], mode='full')
    return x_audio

def play_audio(event):
    if sd is None: return
    set_status("Processing audio...", "orange")
    sd.stop()
    label = radio.value_selected
    sig = get_audio_signal(label)
    norm_sig = sig / (np.max(np.abs(sig)) + 1e-9)
    set_status(f"Playing: {label}", "green")
    sd.play(norm_sig, sr_a)

def save_wav_files(event):
    set_status("Saving WAV files...", "orange")
    audio_dir = os.path.join(BASE_DIR, "results", "audio"); os.makedirs(audio_dir, exist_ok=True)
    conv_audio = fftconvolve(x_audio, y_audio, mode='full')[:int(sr_a*dur_a)]
    
    # Имена файлов теперь зависят от имен инструментов в конфиге
    name_x = cfg['x']['name'].replace(" ", "_").lower()
    name_y = cfg['y']['name'].replace(" ", "_").lower()
    
    for name, sig in [(f"x_{name_x}", x_audio), (f"y_{name_y}", y_audio), ("convolution", conv_audio)]:
        norm_sig = sig / (np.max(np.abs(sig)) + 1e-9)
        wavfile.write(os.path.join(audio_dir, f"{name}.wav"), sr_a, np.int16(norm_sig * 32767))
    
    set_status(f"Saved to: results/audio/", "darkgreen")

def save_results(event):
    set_status("Saving graphs...", "orange")
    plots_dir = os.path.join(BASE_DIR, "results", "graphs"); os.makedirs(plots_dir, exist_ok=True)
    current_label = radio.value_selected.split('.')[0]
    fig.savefig(os.path.join(plots_dir, f"var{VARIANT}_group_{current_label}.png"), dpi=150)
    set_status(f"Saved to: results/graphs/", "darkgreen")

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
                ax.bar(['ДПФ', 'БПФ', 'NumPy', 'Свертка', 'Корреляция'], errors, color=['red', 'orange', 'green', 'blue', 'purple'])
                ax.set_yscale('log'); ax.set_title("Ошибки реализаций (максимальная разность с эталоном)"); ax.grid(True, alpha=0.3, which='both')
                for j, v in enumerate(errors): ax.text(j, v, f"{v:.1e}", ha='center', va='bottom')
            else: ax.axis('off')
    else:
        indices = group_mapping[menu_labels.index(label)]
        for i, p_idx in enumerate(indices):
            ax = main_axes[i]; ax.set_axis_on(); ax.clear()
            x, y, title, p_type, color, unit = plots_data[p_idx]
            ax.set_title(f"График №{p_idx + 1}: {title}", fontsize=11, fontweight='bold')
            if p_type == "plot": ax.plot(x, y, color=color, lw=1.5)
            else: ax.stem(x, y, linefmt=color, markerfmt='o', basefmt=" ")
            ax.set_xlabel(unit, fontsize=9); ax.grid(True, alpha=0.3)
    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor(main_axes, hover=True)
    @current_cursor.connect("add")
    def _(sel):
        sel.annotation.set_text(f"x: {sel.target[0]:.2f}\ny: {sel.target[1]:.4f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8, boxstyle="round")
    fig.canvas.draw_idle()

radio.on_clicked(update_plots); update_plots(menu_labels[0]); plt.show()
