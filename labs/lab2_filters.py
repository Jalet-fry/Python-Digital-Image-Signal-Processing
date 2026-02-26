import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import mplcursors

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.signals.generator import generate_instrument_signal
from core.signals.fourier import fft
from core.signals.filters import moving_average_recursive, fir_window_bandpass, iir_bandpass, apply_iir

try:
    import sounddevice as sd
except ImportError:
    sd = None

# ==========================================================
# 1. ПАРАМЕТРЫ И ГЕНЕРАЦИЯ (Вариант 10)
# ==========================================================
cfg = {
    'name': 'Виолончель', 'A': [1.0, 0.6, 0.4, 0.2], 'f0': 110, 'h': [1, 2, 3, 4],
    'sr': 8000, 'N': 2048, 'dur_audio': 3.0
}

sr, N = cfg['sr'], cfg['N']
t_plot = np.arange(N) / sr

# Чистый сигнал
_, x_clean = generate_instrument_signal(cfg['A'], cfg['f0'], cfg['h'], 0, duration=cfg['dur_audio'], sr=sr)

# Зашумленный сигнал
noise_power = 0.05
noise = np.random.normal(0, np.sqrt(noise_power), len(x_clean))
interference = 0.4 * np.sin(2 * np.pi * 1500 * np.arange(len(x_clean)) / sr) # Наводка 1.5 кГц
x_noisy = x_clean + noise + interference

# ==========================================================
# 2. ПРИМЕНЕНИЕ ФИЛЬТРОВ И ЗАМЕРЫ
# ==========================================================

# 2.1 Однородный (M=79)
t1 = time.time()
y_ma = moving_average_recursive(x_noisy, M=79)
dt_ma = time.time() - t1

# 2.2 КИХ (80-300 Гц, M=151)
t2 = time.time()
h_fir = fir_window_bandpass(80, 300, M=151, sr=sr)
y_fir = np.convolve(x_noisy, h_fir, mode='same')
dt_fir = time.time() - t2

# 2.3 БИХ (f0=200, BW=60)
t3 = time.time()
b_iir, a_iir = iir_bandpass(200, 60, sr=sr)
y_iir = apply_iir(x_noisy, b_iir, a_iir)
dt_iir = time.time() - t3

# Расчет SNR (Signal-to-Noise Ratio)
def calc_snr(clean, processed):
    noise_part = processed - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise_part**2) + 1e-12))

snr_noisy = calc_snr(x_clean, x_noisy)
snr_ma = calc_snr(x_clean, y_ma)
snr_fir = calc_snr(x_clean, y_fir)
snr_iir = calc_snr(x_clean, y_iir)

# ==========================================================
# 3. ЛОГИРОВАНИЕ (ОТЧЕТ)
# ==========================================================
def save_report():
    report_path = os.path.join(BASE_DIR, "results", "lab2_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №2 (ВАРИАНТ 10) ===\n")
        f.write(f"Инструмент: {cfg['name']}\n\n")
        
        f.write("1. ПАРАМЕТРЫ ФИЛЬТРОВ:\n")
        f.write(f"- Однородный: Рекурсивный, M=79\n")
        f.write(f"- КИХ: Полосовой 80-300 Гц, Окно Блэкмана, M=151\n")
        f.write(f"- БИХ: Полосовой f0=200 Гц, BW=60 Гц (Биквадрат)\n\n")
        
        f.write("2. ЭФФЕКТИВНОСТЬ (SNR - чем выше, тем чище звук):\n")
        f.write(f"- Исходный (с шумом): {snr_noisy:.2f} дБ\n")
        f.write(f"- После Однородного: {snr_ma:.2f} дБ\n")
        f.write(f"- После КИХ (FIR):   {snr_fir:.2f} дБ\n")
        f.write(f"- После БИХ (IIR):   {snr_iir:.2f} дБ\n\n")
        
        f.write("3. СКОРОСТЬ ОБРАБОТКИ (время на 3 сек сигнала):\n")
        f.write(f"- Однородный: {dt_ma:.6f} сек\n")
        f.write(f"- КИХ (FIR):   {dt_fir:.6f} сек\n")
        f.write(f"- БИХ (IIR):   {dt_iir:.6f} сек\n\n")
        
        f.write("4. КОЭФФИЦИЕНТЫ (для проверки):\n")
        f.write(f"КИХ (первые 10): {h_fir[:10].tolist()}\n")
        f.write(f"БИХ (b): {b_iir.tolist()}\n")
        f.write(f"БИХ (a): {a_iir.tolist()}\n")

save_report()

# ==========================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ==========================================================
data_map = {
    'Оригинал + Шум': x_noisy,
    'Однородный (M=79)': y_ma,
    'КИХ (Blackman 80-300Гц)': y_fir,
    'БИХ (Bandpass 200Гц)': y_iir
}

fig = plt.figure(figsize=(14, 9))
fig.canvas.manager.set_window_title('Лабораторная работа №2 - Фильтрация')

ax_time = fig.add_subplot(2, 1, 1)
ax_freq = fig.add_subplot(2, 1, 2)
plt.subplots_adjust(left=0.25, bottom=0.1, hspace=0.35)

ax_menu = plt.axes([0.02, 0.6, 0.18, 0.2], facecolor='#f0f0f0')
radio = RadioButtons(ax_menu, list(data_map.keys()), activecolor='royalblue')

ax_play = plt.axes([0.02, 0.45, 0.18, 0.06])
btn_play = Button(ax_play, 'Play Selected', color='lightgreen', hovercolor='lime')

status_text = fig.text(0.02, 0.02, "Ready. Report saved to results/", fontsize=10, color='darkblue')

current_cursor = None
freqs = np.fft.fftfreq(N, 1/sr)[:N//2]

def get_spec(sig):
    win = sig[:N] * np.blackman(N)
    mag = np.abs(fft(win))[:N//2]
    return 20 * np.log10(mag + 1e-9)

def update(label):
    global current_cursor
    sig_full = data_map[label]
    
    ax_time.clear()
    ax_time.plot(t_plot*1000, x_noisy[:N], color='gray', alpha=0.3, label='Noisy')
    ax_time.plot(t_plot*1000, sig_full[:N], color='blue', lw=1, label='Filtered')
    ax_time.set_title(f"Временная область: {label}"); ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc='upper right')
    
    ax_freq.clear()
    ax_freq.plot(freqs, get_spec(x_noisy), color='gray', alpha=0.3, label='Noisy Spec')
    ax_freq.plot(freqs, get_spec(sig_full), color='red', lw=1.2, label='Filtered Spec')
    ax_freq.set_title("Амплитудный спектр (дБ)"); ax_freq.set_ylim([-40, 60]); ax_freq.grid(True, alpha=0.3)
    ax_freq.legend(loc='upper right')
    
    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor([ax_time, ax_freq], hover=True)
    fig.canvas.draw_idle()

radio.on_clicked(update)
btn_play.on_clicked(lambda e: sd.play(data_map[radio.value_selected]/np.max(np.abs(data_map[radio.value_selected])), sr) if sd else None)

update(list(data_map.keys())[0])
plt.show()
