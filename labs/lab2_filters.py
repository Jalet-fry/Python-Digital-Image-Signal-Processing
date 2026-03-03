import sys
import os
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

# Библиотека для звука (если установлена)
try:
    import sounddevice as sd
except ImportError:
    sd = None

# ==========================================================
# 1. ПОДГОТОВКА ДАННЫХ
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

# Конфиги: lab1 - для звука инструмента, lab2 - для параметров фильтров
lab1_cfg = get_lab_config(1, VARIANT)
lab2_cfg = get_lab_config(2, VARIANT)

sr = 8000 # Частота дискретизации
duration = 3.0 # Генерируем 3 секунды звука, чтобы успеть послушать

# Генерируем "чистую" ноту виолончели/контрабаса
_, x_clean = generate_instrument_signal(
    lab1_cfg['x']['A'], lab1_cfg['x']['f0'], lab1_cfg['x']['h'], 0, 
    duration=duration, sr=sr
)

# ПОРТИМ СИГНАЛ:
# 1. Добавляем белый шум (равномерное шипение)
noise = np.random.normal(0, 0.05, len(x_clean))
# 2. Добавляем наводку (писк на частоте 1500 Гц)
interference = 0.4 * np.sin(2 * np.pi * 1500 * np.arange(len(x_clean)) / sr)
x_noisy = x_clean + noise + interference

# ==========================================================
# 2. ФИЛЬТРАЦИЯ (ТРИ ПОДХОДА)
# ==========================================================

# 2.1. ОДНОРОДНЫЙ ФИЛЬТР (Скользящее среднее)
# Просто усредняет соседние точки. Хорошо давит шум, но "замыливает" сигнал.
y_ma = moving_average_recursive(x_noisy, M=lab2_cfg['M_ma'])

# 2.2. КИХ ФИЛЬТР (Оконный полосовой)
# Пропускает только частоты в заданном диапазоне (например, 80-300 Гц).
# Использует окно Блэкмана для плавного среза.
f_fir = lab2_cfg['fir']['f']
h_fir = fir_window_bandpass(f_fir[0], f_fir[1], M=lab2_cfg['fir']['M'], sr=sr)
y_fir = np.convolve(x_noisy, h_fir, mode='same')

# 2.3. БИХ ФИЛЬТР (Рекурсивный)
# Самый эффективный. Использует обратную связь.
b_iir, a_iir = iir_bandpass(lab2_cfg['iir']['f0'], lab2_cfg['iir']['bw'], sr=sr)
y_iir = apply_iir(x_noisy, b_iir, a_iir)

# ==========================================================
# 3. SNR (ОТНОШЕНИЕ СИГНАЛ/ШУМ)
# ==========================================================
def calc_snr(clean, processed):
    """Считает SNR в децибелах. Чем выше число, тем чище звук."""
    noise_part = processed - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise_part**2) + 1e-12))

# ==========================================================
# 4. ИНТЕРФЕЙС И СРАВНЕНИЕ
# ==========================================================
data_map = {
    'Зашумленный (Input)': x_noisy,
    'MA (Усреднение)': y_ma,
    'КИХ (Полосовой)': y_fir,
    'БИХ (Резонансный)': y_iir
}

fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(left=0.3, hspace=0.4)

# Меню выбора фильтра
ax_radio = plt.axes([0.02, 0.5, 0.2, 0.3], facecolor='#f0f0f0')
radio = RadioButtons(ax_radio, list(data_map.keys()))

# Кнопка проигрывания звука
ax_play = plt.axes([0.02, 0.35, 0.2, 0.08])
btn_play = Button(ax_play, '▶ Play Audio', color='lightgreen')

def update(label):
    sig = data_map[label]
    N_plot = 1000 # Рисуем только кусочек для наглядности
    
    # Временной график
    ax_time.clear()
    ax_time.plot(x_noisy[:N_plot], alpha=0.3, label='Noisy')
    ax_time.plot(sig[:N_plot], color='blue', label='Filtered')
    ax_time.set_title(f"Временная область: {label} (SNR: {calc_snr(x_clean, sig):.2f} dB)")
    ax_time.legend()
    
    # Частотный график (Спектр)
    ax_freq.clear()
    spec_noisy = np.abs(np.fft.fft(x_noisy[:2048]))[:1024]
    spec_filtered = np.abs(np.fft.fft(sig[:2048]))[:1024]
    ax_freq.plot(spec_noisy, alpha=0.3)
    ax_freq.plot(spec_filtered, color='red')
    ax_freq.set_title("Частотный спектр (Видно, как фильтр отрезает шум)")
    
    fig.canvas.draw_idle()

def play_sound(event):
    if sd:
        # Нормализуем звук перед проигрыванием, чтобы не оглохнуть
        s = data_map[radio.value_selected]
        sd.play(s / np.max(np.abs(s)), sr)

radio.on_clicked(update)
btn_play.on_clicked(play_sound)
update(list(data_map.keys())[0])

plt.show()
