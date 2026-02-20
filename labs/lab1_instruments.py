import sys
import os
import time
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import numpy as np
import mplcursors
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

# Настройка стиля
plt.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-v0_8-muted')

# ==========================================================
# 1. ПАРАМЕТРЫ (ВАРИАНТ №10)
# ==========================================================
config = {
    'x': {'name': 'Виолончель', 'A': [1.0, 0.6, 0.4, 0.2], 'f0': 110, 'h': [1, 2, 3, 4], 'phi': 0},
    'y': {'name': 'Контрабас', 'A': [1.0, 0.7, 0.5], 'f0': 55, 'h': [1, 2, 3], 'phi': 0},
    'N': 1024,
    'sr': 8000
}

N, sr = config['N'], config['sr']
duration = N / sr

# ==========================================================
# 2. РАСЧЕТЫ
# ==========================================================
t, x_raw = generate_instrument_signal(config['x']['A'], config['x']['f0'], config['x']['h'], config['x']['phi'], duration=duration, sr=sr)
_, y_raw = generate_instrument_signal(config['y']['A'], config['y']['f0'], config['y']['h'], config['y']['phi'], duration=duration, sr=sr)

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

def get_clean(data, max_f=600):
    mag = np.abs(data) / (N / 2)
    ph = np.angle(data)
    ph[mag < 0.001 * np.max(mag)] = 0
    freqs = np.fft.fftfreq(N, d=dt)
    idx = np.where(freqs[:N//2] <= max_f)[0][-1]
    return freqs[:idx], mag[:idx], ph[:idx]

f_dx, m_dx, p_dx = get_clean(d_x); f_fx, m_fx, p_fx = get_clean(f_x); f_lx, m_lx, p_lx = get_clean(l_fx)
f_dy, m_dy, p_dy = get_clean(d_y); f_fy, m_fy, p_fy = get_clean(f_y); f_ly, m_ly, p_ly = get_clean(l_fy)

# ==========================================================
# 3. ВСЕ 24 ГРАФИКА
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

group_mapping = [
    [0, 4, 7], [1, 10, 13], [2, 5, 18], [3, 6, 19],
    [8, 11, 20], [9, 12, 21], [14, 15, 22], [16, 17, 23]
]

# ==========================================================
# 4. ИНТЕРФЕЙС
# ==========================================================
fig = plt.figure(figsize=(15, 10))
fig.canvas.manager.set_window_title('Лабораторная работа №1 - Анализ сигналов (Вариант 10)')

ax_menu = plt.axes([0.02, 0.4, 0.16, 0.35], facecolor='#f0f0f0')
menu_labels = [
    '1. Восстановление X', '2. Восстановление Y',
    '3. Спектр Амп. X', '4. Спектр Фаз. X',
    '5. Спектр Амп. Y', '6. Спектр Фаз. Y',
    '7. Свертка X * Y', '8. Корреляция X & Y'
]
radio = RadioButtons(ax_menu, menu_labels, active=0, activecolor='royalblue')

# Кнопка звука (без Emoji для избежания ошибок шрифта)
ax_play = plt.axes([0.02, 0.2, 0.16, 0.08])
btn_play = Button(ax_play, 'Play Audio', color='lightgreen', hovercolor='lime')

def play_audio(event):
    if sd is None: return
    sd.stop()
    label = radio.value_selected
    # Если выбрана группа с X, играем x_s. Если с Y - y_s. 
    # Свертка/Корреляция длинные, для простоты играем x_s (основной сигнал)
    sig = x_s if 'X' in label or 'свертка' in label.lower() or 'корреляция' in label.lower() else y_s
    norm_sig = sig / (np.max(np.abs(sig)) + 1e-9)
    print(f"Воспроизведение: {label}...")
    sd.play(norm_sig, sr)

btn_play.on_clicked(play_audio)

main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
plt.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.06, hspace=0.35)

current_cursor = None

def update_plots(label):
    global current_cursor
    group_idx = menu_labels.index(label)
    indices = group_mapping[group_idx]
    
    for i, p_idx in enumerate(indices):
        ax = main_axes[i]
        ax.clear()
        x, y, title, p_type, color, unit = plots_data[p_idx]
        full_title = f"График №{p_idx + 1}: {title}"
        if p_type == "plot": 
            ax.plot(x, y, color=color, lw=1.5)
        else: 
            ax.stem(x, y, linefmt=color, markerfmt='o', basefmt=" ")
        ax.set_title(full_title, fontsize=11, fontweight='bold')
        ax.set_xlabel(unit, fontsize=9); ax.grid(True, alpha=0.3)
    
    # ПЕРЕСОЗДАЕМ КУРСОР для новых данных на осях
    if current_cursor:
        current_cursor.remove()
    
    current_cursor = mplcursors.cursor(main_axes, hover=True)
    
    @current_cursor.connect("add")
    def _(sel):
        # Форматируем подпись, чтобы она всегда была видна и корректна
        sel.annotation.set_text(f"x: {sel.target[0]:.2f}\ny: {sel.target[1]:.4f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8, boxstyle="round")

    fig.canvas.draw_idle()

radio.on_clicked(update_plots)
update_plots(menu_labels[0])
plt.show()
