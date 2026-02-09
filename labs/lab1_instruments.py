import sys
import os
import time
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import mplcursors

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.signals.generator import generate_instrument_signal
from core.signals.fourier import dft, idft, fft, ifft
from core.signals.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation

plt.rcParams['toolbar'] = 'None'

# ==========================================================
# 1. ПАРАМЕТРЫ (N=1024 ДЛЯ ВСЕХ)
# ==========================================================
config = {
    'x': {'name': 'Виолончель', 'A': [1.0, 0.6, 0.4, 0.2], 'f0': 110, 'h': [1, 2, 3, 4], 'phi': 0},
    'y': {'name': 'Контрабас', 'A': [1.0, 0.7, 0.5], 'f0': 55, 'h': [1, 2, 3], 'phi': 0},
    'N': 1024,
    'sr': 8000
}

N = config['N']
sr = config['sr']
duration = N / sr

# ==========================================================
# 2. ОЧИСТКА СТАРЫХ ГРАФИКОВ
# ==========================================================
plots_dir = os.path.join(BASE_DIR, "results", "plots")
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir) # Удаляем всё, чтобы не было дублей
os.makedirs(plots_dir)

# ==========================================================
# 3. РАСЧЕТЫ
# ==========================================================
print(f"Выполняются тяжелые расчеты для N={N} (около 15 сек)...")
t, x_raw = generate_instrument_signal(config['x']['A'], config['x']['f0'], config['x']['h'], config['x']['phi'], duration=duration, sr=sr)
_, y_raw = generate_instrument_signal(config['y']['A'], config['y']['f0'], config['y']['h'], config['y']['phi'], duration=duration, sr=sr)

x_s, y_s, t = x_raw[:N], y_raw[:N], t[:N]
dt = t[1] - t[0]

# Все считаем на полном N=1024
dft_x, dft_y = dft(x_s), dft(y_s)
fft_x, fft_y = fft(x_s), fft(y_s)
idft_x, idft_y = idft(dft_x).real, idft(dft_y).real
ifft_x, ifft_y = ifft(fft_x).real[:N], ifft(fft_y).real[:N]

conv_man = linear_convolution(x_s, y_s)
conv_fft = fft_convolution(x_s, y_s).real
corr_man = correlation(x_s, y_s)
corr_fft = fft_correlation(x_s, y_s).real

lib_fft_x, lib_fft_y = np.fft.fft(x_s), np.fft.fft(y_s)
lib_conv, lib_corr = np.convolve(x_s, y_s), np.correlate(x_s, y_s, mode='full')

# ==========================================================
# 4. ПОДГОТОВКА И СОХРАНЕНИЕ (PNG)
# ==========================================================
def get_spec_plot(data):
    mag = np.abs(data) / (N / 2)
    phase = np.angle(data)
    phase[mag < 0.05 * np.max(mag)] = 0
    freqs = np.fft.fftfreq(len(data), d=dt)
    return freqs[:N//2], mag[:N//2], phase[:N//2]

f_dx, m_dx, p_dx = get_spec_plot(dft_x); f_fx, m_fx, p_fx = get_spec_plot(fft_x)
f_dy, m_dy, p_dy = get_spec_plot(dft_y); f_fy, m_fy, p_fy = get_spec_plot(fft_y)
f_lx, m_lx, p_lx = get_spec_plot(lib_fft_x); f_ly, m_ly, p_ly = get_spec_plot(lib_fft_y)

plots_data = [
    (t*1000, x_s, "01. x(t) Оригинал", "plot", "blue", "ms"),
    (t*1000, y_s, "02. y(t) Оригинал", "plot", "orange", "ms"),
    (f_dx, m_dx, "03. x DFT Амплитуда", "stem", "blue", "Hz"),
    (f_dx, p_dx, "04. x DFT Фаза", "stem", "blue", "Hz"),
    (t*1000, idft_x, "05. x IDFT Восстановлен", "plot", "green", "ms"),
    (f_fx, m_fx, "06. x FFT Амплитуда", "stem", "blue", "Hz"),
    (f_fx, p_fx, "07. x FFT Фаза", "stem", "blue", "Hz"),
    (t*1000, ifft_x, "08. x IFFT Восстановлен", "plot", "purple", "ms"),
    (f_dy, m_dy, "09. y DFT Амплитуда", "stem", "orange", "Hz"),
    (f_dy, p_dy, "10. y DFT Фаза", "stem", "orange", "Hz"),
    (t*1000, idft_y, "11. y IDFT Восстановлен", "plot", "red", "ms"),
    (f_fy, m_fy, "12. y FFT Амплитуда", "stem", "orange", "Hz"),
    (f_fy, p_fy, "13. y FFT Фаза", "stem", "orange", "Hz"),
    (t*1000, ifft_y, "14. y IFFT Восстановлен", "plot", "brown", "ms"),
    (range(len(conv_man)), conv_man, "15. Свертка Вручную", "plot", "blue", "idx"),
    (range(len(conv_fft)), conv_fft, "16. Свертка через БПФ", "plot", "cyan", "idx"),
    (range(len(corr_man)), corr_man, "17. Корреляция Вручную", "plot", "gray", "idx"),
    (range(len(corr_fft)), corr_fft, "18. Корреляция через БПФ", "plot", "black", "idx"),
    (f_lx, m_lx, "19. x Lib FFT Амплитуда", "stem", "blue", "Hz"),
    (f_lx, p_lx, "20. x Lib FFT Фаза", "stem", "blue", "Hz"),
    (f_ly, m_ly, "21. y Lib FFT Амплитуда", "stem", "orange", "Hz"),
    (f_ly, p_ly, "22. y Lib FFT Фаза", "stem", "orange", "Hz"),
    (range(len(lib_conv)), lib_conv, "23. Lib Свертка", "plot", "green", "idx"),
    (range(len(lib_corr)), lib_corr, "24. Lib Корреляция", "plot", "red", "idx"),
]

for i, (x, y, title, p_type, color, unit) in enumerate(plots_data):
    plt.figure(figsize=(10, 5))
    if p_type == "plot": plt.plot(x, y, color=color, lw=1.2)
    else: plt.stem(x, y, linefmt=color, markerfmt=' ', basefmt=" ")
    plt.title(title); plt.xlabel(unit); plt.grid(True, alpha=0.3)
    if unit == "Hz": plt.xlim(0, 1500)
    plt.savefig(os.path.join(plots_dir, f"{title.replace(' ', '_')}.png"), dpi=120)
    plt.close()

print(f"[УСПЕХ] 24 графика обновлены в {plots_dir}")

# ==========================================================
# 5. ИНТЕРФЕЙС
# ==========================================================
fig = plt.figure(figsize=(13, 9))
n_rows, visible_rows, scroll_pos = 12, 3, 0
axes = []
gs = fig.add_gridspec(n_rows, 2, hspace=0.8)

for i in range(24):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    x, y, title, p_type, color, unit = plots_data[i]
    if p_type == "plot": ax.plot(x, y, color=color, lw=1)
    else: ax.stem(x, y, linefmt=color, markerfmt=' ', basefmt=" ")
    ax.set_title(title, fontsize=9); ax.grid(True, alpha=0.2)
    if unit == "Hz": ax.set_xlim(0, 1500)
    axes.append(ax)

def update():
    for i, ax in enumerate(axes):
        row = i // 2
        if scroll_pos <= row < scroll_pos + visible_rows:
            l_row = row - scroll_pos
            ax.set_position([ax.get_position().x0, 0.88 - (l_row+1)*0.25, ax.get_position().width, 0.18])
            ax.set_visible(True)
        else: ax.set_visible(False)
    fig.canvas.draw_idle()

ax_p, ax_n = plt.axes([0.4, 0.02, 0.05, 0.04]), plt.axes([0.55, 0.02, 0.05, 0.04])
b_p, b_n = Button(ax_p, '<'), Button(ax_n, '>')
b_n.on_clicked(lambda e: change(1)); b_p.on_clicked(lambda e: change(-1))
def change(d):
    global scroll_pos
    scroll_pos = max(0, min(scroll_pos + d, n_rows - visible_rows)); update()

update()
mplcursors.cursor(hover=True)
plt.show()
