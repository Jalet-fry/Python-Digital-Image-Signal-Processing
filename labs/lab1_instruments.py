import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import mplcursors

# Отключаем стандартную панель инструментов
plt.rcParams['toolbar'] = 'None'

# Добавляем путь к ядру
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signals.generator import generate_instrument_signal
from core.signals.fourier import dft, idft, fft, ifft
from core.signals.math_ops import linear_convolution, fft_convolution, correlation, fft_correlation

# --- 1. НАСТРОЙКИ И РАСЧЕТЫ ---
N = 256
sr = 8000
duration = N / sr

t, x_s = generate_instrument_signal([1, 0.3, 0.1], 220, [1, 4, 6], 0, duration=duration, sr=sr)
_, y_s = generate_instrument_signal([1, 0.4, 0.2], 220, [1, 3, 5], 0, duration=duration, sr=sr)
t, x_s, y_s = t[:N], x_s[:N], y_s[:N]
dt = t[1] - t[0]

t_ms = t * 1000 

def get_spec(data):
    n = len(data)
    mag = np.abs(data) / (n / 2)
    phase = np.angle(data)
    phase[mag < 0.05] = 0
    freqs = np.fft.fftfreq(n, d=dt)
    return freqs[:n//2], mag[:n//2], phase[:n//2]

# Вычисления
d_x, f_x = dft(x_s), fft(x_s)
id_x, if_x = idft(d_x), ifft(f_x)[:N]
d_y, f_y = dft(y_s), fft(y_s)
id_y, if_y = idft(d_y), ifft(f_y)[:N]
c_m, c_f = linear_convolution(x_s, y_s), fft_convolution(x_s, y_s).real
cr_m, cr_f = correlation(x_s, y_s), fft_correlation(x_s, y_s).real
l_fx, l_fy = np.fft.fft(x_s), np.fft.fft(y_s)
l_c, l_cr = np.convolve(x_s, y_s), np.correlate(x_s, y_s, mode='full')

f_dx, m_dx, p_dx = get_spec(d_x); f_fx, m_fx, p_fx = get_spec(f_x)
f_dy, m_dy, p_dy = get_spec(d_y); f_fy, m_fy, p_fy = get_spec(f_y)
f_lx, m_lx, p_lx = get_spec(l_fx); f_ly, m_ly, p_ly = get_spec(l_fy)

plots_data = [
    (t_ms, x_s, '1. x(t) Original', 'plot', 'blue', 'Time (ms)', 'Amp'),
    (t_ms, y_s, '2. y(t) Original', 'plot', 'orange', 'Time (ms)', 'Amp'),
    (f_dx, m_dx, '3. x DFT: Amplitude', 'stem', 'blue', 'Freq (Hz)', 'Mag'),
    (f_dx, p_dx, '4. x DFT: Phase', 'stem', 'blue', 'Freq (Hz)', 'Phase (rad)'),
    (t_ms, id_x, '5. x IDFT (Restored)', 'plot', 'green', 'Time (ms)', 'Amp'),
    (f_fx, m_fx, '6. x FFT: Amplitude', 'stem', 'blue', 'Freq (Hz)', 'Mag'),
    (f_fx, p_fx, '7. x FFT: Phase', 'stem', 'blue', 'Freq (Hz)', 'Phase (rad)'),
    (t_ms, if_x, '8. x IFFT (Restored)', 'plot', 'purple', 'Time (ms)', 'Amp'),
    (f_dy, m_dy, '9. y DFT: Amplitude', 'stem', 'orange', 'Freq (Hz)', 'Mag'),
    (f_dy, p_dy, '10. y DFT: Phase', 'stem', 'orange', 'Freq (Hz)', 'Phase (rad)'),
    (t_ms, id_y, '11. y IDFT (Restored)', 'plot', 'red', 'Time (ms)', 'Amp'),
    (f_fy, m_fy, '12. y FFT: Amplitude', 'stem', 'orange', 'Freq (Hz)', 'Mag'),
    (f_fy, p_fy, '13. y FFT: Phase', 'stem', 'orange', 'Freq (Hz)', 'Phase (rad)'),
    (t_ms, if_y, '14. y IFFT (Restored)', 'plot', 'brown', 'Time (ms)', 'Amp'),
    (None, c_m, '15. Conv (Manual)', 'plot', 'blue', 'Index', 'Val'),
    (None, c_f, '16. Conv (FFT)', 'plot', 'cyan', 'Index', 'Val'),
    (None, cr_m, '17. Corr (Manual)', 'plot', 'gray', 'Index', 'Val'),
    (None, cr_f, '18. Corr (FFT)', 'plot', 'black', 'Index', 'Val'),
    (f_lx, m_lx, '19. Lib FFT x (Amp)', 'stem', 'blue', 'Freq (Hz)', 'Mag'),
    (f_lx, p_lx, '20. Lib FFT x (Phase)', 'stem', 'blue', 'Freq (Hz)', 'Phase (rad)'),
    (f_ly, m_ly, '21. Lib FFT y (Amp)', 'stem', 'orange', 'Freq (Hz)', 'Mag'),
    (f_ly, p_ly, '22. Lib FFT y (Phase)', 'stem', 'orange', 'Freq (Hz)', 'Phase (rad)'),
    (None, l_c, '23. Lib Conv', 'plot', 'green', 'Index', 'Val'),
    (None, l_cr, '24. Lib Corr', 'plot', 'red', 'Index', 'Val')
]

# --- 2. ИНТЕРФЕЙС ---
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Лабораторная №1: Инструменты и Фурье', fontsize=14, fontweight='bold')

n_rows = 12
gs = fig.add_gridspec(n_rows, 2, hspace=0.8, wspace=0.3)
axes_list = []

for i in range(24):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    x, y, title, p_type, color, xlabel, ylabel = plots_data[i]
    if p_type == 'plot':
        if x is not None: ax.plot(x, y, color=color, linewidth=1.2)
        else: ax.plot(y, color=color, linewidth=1.2)
    else:
        ax.stem(x, y, linefmt=color, markerfmt=' ', basefmt=" ")
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    axes_list.append(ax)

scroll_pos = 0
visible_rows = 3

def update_view():
    for i, ax in enumerate(axes_list):
        row = i // 2
        if scroll_pos <= row < scroll_pos + visible_rows:
            local_row = row - scroll_pos
            height = 0.75 / visible_rows
            y_pos = 0.9 - (local_row + 1) * height
            ax.set_position([ax.get_position().x0, y_pos + 0.08, ax.get_position().width, height - 0.12])
            ax.set_visible(True)
        else:
            ax.set_visible(False)
    
    status_text.set_text(f"Страница {scroll_pos+1} из {n_rows-visible_rows+1}")
    fig.canvas.draw_idle()

ax_prev = plt.axes([0.35, 0.02, 0.1, 0.04])
ax_next = plt.axes([0.55, 0.02, 0.1, 0.04])
btn_prev = Button(ax_prev, '↑ Назад')
btn_next = Button(ax_next, '↓ Вперед')

status_ax = plt.axes([0.45, 0.02, 0.1, 0.04], frameon=False)
status_ax.set_xticks([]); status_ax.set_yticks([])
status_text = status_ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=9)

def go_next(e):
    global scroll_pos
    if scroll_pos < n_rows - visible_rows: scroll_pos += 1; update_view()

def go_prev(e):
    global scroll_pos
    if scroll_pos > 0: scroll_pos -= 1; update_view()

btn_next.on_clicked(go_next)
btn_prev.on_clicked(go_prev)
update_view()

# Интерактивность
cursor = mplcursors.cursor(hover=True)
@cursor.connect("add")
def on_add(sel):
    x, y = sel.target
    # Самый надежный способ получить оси через аннотацию
    ax = sel.annotation.axes
    title = ax.get_title() if ax else ""

    unit_x = "ms" if "t)" in title or "Restored" in title else "Hz" if "FT" in title else "idx"
    sel.annotation.set(text=f"{x:.2f} {unit_x}\nval: {y:.3f}")
    sel.annotation.get_bbox_patch().set(alpha=0.9, facecolor='white', edgecolor='blue')

plt.show()
