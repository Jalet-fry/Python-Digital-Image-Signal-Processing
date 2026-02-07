import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

# Добавляем путь к ядру
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signals.generator import generate_instrument_signal
from core.signals.math_ops import circular_convolution, linear_convolution, fft_convolution, correlation

# --- ПАРАМЕТРЫ ВАРИАНТА 4 ---
# Берем очень короткий сигнал для наглядности свертки (иначе графики будут слишком плотными)
duration = 0.01 
t, x = generate_instrument_signal([1, 0.3, 0.1], 220, [1, 4, 6], 0, duration=duration)
_, y = generate_instrument_signal([1, 0.4, 0.2], 220, [1, 3, 5], 0, duration=duration)

# 1. Свертки
conv_circ = circular_convolution(x, y)
conv_lin = linear_convolution(x, y)
conv_fft = fft_convolution(x, y)

# 2. Корреляция
corr_res = correlation(x, y)

# --- UI И ОТРИСОВКА ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(4, 1, figsize=(11, 12))
plt.subplots_adjust(hspace=0.6)

# График 1: Исходные сигналы
axs[0].plot(x, label='Струнные (x)', color='#1f77b4')
axs[0].plot(y, label='Духовые (y)', color='#ff7f0e', alpha=0.7)
axs[0].set_title('ИСХОДНЫЕ СИГНАЛЫ (ВАРИАНТ 4)', fontweight='bold')
axs[0].legend()

# График 2: Круговая свертка
line_circ, = axs[1].plot(conv_circ, color='green', label='Circular Conv')
axs[1].set_title('КРУГОВАЯ СВЕРТКА (Time Domain)', fontweight='bold')
axs[1].legend()

# График 3: Быстрая свертка (FFT) - проверка на совпадение
line_fft, = axs[2].plot(conv_fft.real, color='purple', linestyle='--', label='FFT Conv')
axs[2].set_title('БЫСТРАЯ СВЕРТКА (через БПФ)', fontweight='bold')
axs[2].legend()

# График 4: Взаимная корреляция
line_corr, = axs[3].plot(corr_res, color='red', label='Correlation')
axs[3].set_title('ВЗАИМНАЯ КОРРЕЛЯЦИЯ', fontweight='bold')
axs[3].legend()

# --- ИНТЕРАКТИВНОСТЬ ---
cursor = mplcursors.cursor([line_circ, line_fft, line_corr], hover=True)

@cursor.connect("add")
def on_add(sel):
    x_idx, y_val = sel.target
    sel.annotation.set(text=f"Index: {int(x_idx)}\nValue: {y_val:.3f}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

print("Лабораторная №2 запущена. Сравните графики 2 и 3 - они должны быть идентичны!")
plt.show()
