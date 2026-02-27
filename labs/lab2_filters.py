import sys
import os
import time
import argparse
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
from core.config_variants import get_lab_config

try:
    import sounddevice as sd
except ImportError:
    sd = None

# ==========================================================
# 0. ЧТЕНИЕ ВАРИАНТА ИЗ АРГУМЕНТОВ
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

# Получаем данные из универсальной базы
lab1_cfg = get_lab_config(1, VARIANT) # Для генерации инструмента
lab2_cfg = get_lab_config(2, VARIANT) # Для фильтрации

# ==========================================================
# 1. ПАРАМЕТРЫ И ГЕНЕРАЦИЯ
# ==========================================================
sr = 8000
N = 2048
dur_audio = 3.0
t_plot = np.arange(N) / sr

# Чистый сигнал (берем из конфига выбранного варианта)
_, x_clean = generate_instrument_signal(
    lab1_cfg['x']['A'], lab1_cfg['x']['f0'], lab1_cfg['x']['h'], 0, 
    duration=dur_audio, sr=sr
)

# Зашумленный сигнал
noise_power = 0.05
noise = np.random.normal(0, np.sqrt(noise_power), len(x_clean))
interference = 0.4 * np.sin(2 * np.pi * 1500 * np.arange(len(x_clean)) / sr)
x_noisy = x_clean + noise + interference

# ==========================================================
# 2. ПРИМЕНЕНИЕ ФИЛЬТРОВ (АДАПТИВНОЕ)
# ==========================================================

# 2.1 Однородный
M_ma = lab2_cfg['M_ma']
y_ma = moving_average_recursive(x_noisy, M=M_ma)

# 2.2 КИХ
f_fir = lab2_cfg['fir']['f']
M_fir = lab2_cfg['fir']['M']
# Для универсальности считаем полосовой, если в конфиге два числа
if isinstance(f_fir, list):
    h_fir = fir_window_bandpass(f_fir[0], f_fir[1], M=M_fir, sr=sr)
else:
    # Если одно число (НЧ или ВЧ), делаем полосу вокруг него для простоты ядра
    h_fir = fir_window_bandpass(f_fir*0.8, f_fir*1.2, M=M_fir, sr=sr)
y_fir = np.convolve(x_noisy, h_fir, mode='same')

# 2.3 БИХ
if lab2_cfg['iir']['type'] == 'bandpass':
    b_iir, a_iir = iir_bandpass(lab2_cfg['iir']['f0'], lab2_cfg['iir']['bw'], sr=sr)
else:
    # Заглушка для НЧ/ВЧ типов БИХ (используем 200 Гц как в 10 варианте)
    b_iir, a_iir = iir_bandpass(200, 60, sr=sr)
y_iir = apply_iir(x_noisy, b_iir, a_iir)

# SNR
def calc_snr(clean, processed):
    noise_part = processed - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise_part**2) + 1e-12))

# ==========================================================
# 3. ОТЧЕТ И ВИЗУАЛИЗАЦИЯ
# ==========================================================
report_path = os.path.join(BASE_DIR, "results", f"lab2_report_var{VARIANT}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"=== ОТЧЕТ ЛАБ 2 (ВАРИАНТ {VARIANT}) ===\n")
    f.write(f"Инструмент: {lab1_cfg['x']['name']}\n")
    f.write(f"Параметры MA: M={M_ma}\n")
    f.write(f"Параметры КИХ: {lab2_cfg['fir']}\n")
    f.write(f"SNR (После КИХ): {calc_snr(x_clean, y_fir):.2f} дБ\n")

data_map = {
    'Оригинал + Шум': x_noisy,
    f'MA (M={M_ma})': y_ma,
    'КИХ (Оконный)': y_fir,
    'БИХ (Резонанс)': y_iir
}

fig = plt.figure(figsize=(14, 9))
fig.canvas.manager.set_window_title(f'Лабораторная работа №2 (Вариант {VARIANT})')
ax_time = fig.add_subplot(2, 1, 1); ax_freq = fig.add_subplot(2, 1, 2)
plt.subplots_adjust(left=0.25, bottom=0.1, hspace=0.35)
ax_menu = plt.axes([0.02, 0.6, 0.18, 0.2], facecolor='#f0f0f0')
radio = RadioButtons(ax_menu, list(data_map.keys()), activecolor='royalblue')
ax_play = plt.axes([0.02, 0.45, 0.18, 0.06]); btn_play = Button(ax_play, 'Play Audio', color='lightgreen')
status_text = fig.text(0.02, 0.02, f"Ready. Variant {VARIANT}", fontsize=10, color='darkblue')

current_cursor = None
freqs = np.fft.fftfreq(N, 1/sr)[:N//2]

def get_spec(sig):
    mag = np.abs(fft(sig[:N] * np.blackman(N)))[:N//2]
    return 20 * np.log10(mag + 1e-9)

def update(label):
    global current_cursor
    sig = data_map[label]
    ax_time.clear(); ax_time.plot(t_plot*1000, x_noisy[:N], alpha=0.3); ax_time.plot(t_plot*1000, sig[:N], color='blue')
    ax_freq.clear(); ax_freq.plot(freqs, get_spec(x_noisy), alpha=0.3); ax_freq.plot(freqs, get_spec(sig), color='red')
    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor([ax_time, ax_freq], hover=True)
    fig.canvas.draw_idle()

radio.on_clicked(update)
btn_play.on_clicked(lambda e: sd.play(data_map[radio.value_selected]/np.max(np.abs(data_map[radio.value_selected])), sr) if sd else None)
update(list(data_map.keys())[0]); plt.show()
