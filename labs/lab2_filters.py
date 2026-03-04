import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import mplcursors
from scipy.io import wavfile
from scipy.signal import freqz

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.signals.generator import generate_instrument_signal
from core.signals.filters import moving_average_recursive, fir_window_bandpass, iir_bandpass, apply_iir
from core.config_variants import get_lab_config
from core.utils.aspects import DSPContext

# ==========================================================
# 1. ПОДГОТОВКА ДАННЫХ (Вариант 10)
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

DSPContext.variant = VARIANT
DSPContext.current_lab = "lab2"

lab1_cfg = get_lab_config(1, VARIANT)
lab2_cfg = get_lab_config(2, VARIANT)

sr = 8000
duration = 2.0 
t_axis = np.linspace(0, duration, int(sr * duration), endpoint=False)

# --- ГЕНЕРАЦИЯ ---
_, x_clean = generate_instrument_signal(lab1_cfg['x']['A'], lab1_cfg['x']['f0'], lab1_cfg['x']['h'], 0, duration=duration, sr=sr)
np.random.seed(42)
white_noise = np.random.normal(0, 0.08, len(x_clean))
interference = 0.4 * np.sin(2 * np.pi * 1500 * t_axis)
total_noise = white_noise + interference
x_noisy = x_clean + total_noise

# --- ФИЛЬТРАЦИЯ ---
y_ma = moving_average_recursive(x_noisy, M=lab2_cfg['M_ma'])
f_range = lab2_cfg['fir']['f']
h_fir = fir_window_bandpass(f_range[0], f_range[1], M=lab2_cfg['fir']['M'], sr=sr)
y_fir = np.convolve(x_noisy, h_fir, mode='same')
b_iir, a_iir = iir_bandpass(lab2_cfg['iir']['f0'], lab2_cfg['iir']['bw'], sr=sr)
y_iir = apply_iir(x_noisy, b_iir, a_iir)

def calc_snr(clean, processed):
    noise_part = processed - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise_part**2) + 1e-12))

# Библиотека для звука
try: import sounddevice as sd
except ImportError: sd = None

# Справочник данных для UI
data_map = {
    "1. Чистый сигнал": (x_clean, "blue", None),
    "2. Только Шум": (total_noise, "gray", None),
    "3. Вход (Зашумленный)": (x_noisy, "orange", calc_snr(x_clean, x_noisy)),
    "4. Фильтр MA": (y_ma, "green", calc_snr(x_clean, y_ma)),
    "5. Фильтр КИХ": (y_fir, "red", calc_snr(x_clean, y_fir)),
    "6. Фильтр БИХ": (y_iir, "purple", calc_snr(x_clean, y_iir))
}

# ==========================================================
# 2. ИНТЕРФЕЙС
# ==========================================================
plt.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-v0_8-muted')
fig = plt.figure(figsize=(15, 10))
fig.canvas.manager.set_window_title(f"ЛР №2: Цифровая фильтрация - Вариант {VARIANT}")

ax_menu = plt.axes([0.02, 0.5, 0.16, 0.35], facecolor='#f8f9fa')
radio = RadioButtons(ax_menu, list(data_map.keys()), active=0, activecolor='royalblue')

ax_play = plt.axes([0.02, 0.4, 0.16, 0.06]); btn_play = Button(ax_play, '▶ Play Audio', color='#d4edda')
ax_save_wav = plt.axes([0.02, 0.32, 0.16, 0.06]); btn_save_wav = Button(ax_save_wav, '💾 Save WAV', color='#d1ecf1')
ax_save_res = plt.axes([0.02, 0.24, 0.16, 0.06]); btn_save_res = Button(ax_save_res, '📊 Save Results', color='#fff3cd')

status_text = fig.text(0.02, 0.03, f"Ready. Variant {VARIANT}", fontsize=10, fontweight='bold')
def set_status(msg, col='black'): status_text.set_text(msg); status_text.set_color(col); fig.canvas.draw_idle()

main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
plt.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.08, hspace=0.4)
current_cursor = None

def get_spectrum(sig, n=2048):
    mag = np.abs(np.fft.fft(sig[:n]))[:n//2]
    freqs = np.fft.fftfreq(n, 1/sr)[:n//2]
    return freqs, mag

def update_plots(label):
    global current_cursor
    for ax in main_axes: ax.clear(); ax.set_axis_on()
    sig, col, snr = data_map[label]
    N_pts = 500
    set_status(f"Показ: {label}")

    if label.startswith(("1.", "2.", "3.")):
        # ГРУППА "ВХОД": Просто смотрим на один сигнал
        # 1. Время
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col)
        main_axes[0].set_title(f"Временная область: {label}", fontweight='bold')
        main_axes[0].set_xlabel("ms")
        
        # 2. Спектр
        f, m = get_spectrum(sig)
        main_axes[1].fill_between(f, m, color=col, alpha=0.4)
        main_axes[1].set_title(f"Амплитудный спектр: {label}", fontweight='bold')
        main_axes[1].set_xlim(0, 2500); main_axes[1].set_xlabel("Hz")

        # 3. Доп. инфо (Гистограмма значений)
        main_axes[2].hist(sig, bins=50, color=col, alpha=0.7)
        main_axes[2].set_title("Распределение амплитуд (Гистограмма)", fontweight='bold')
    
    else:
        # ГРУППА "ФИЛЬТРЫ": Сравниваем До и После
        if "MA" in label: 
            w, h_resp = freqz(np.ones(lab2_cfg['M_ma'])/lab2_cfg['M_ma'], 1, worN=2000)
        elif "КИХ" in label: 
            w, h_resp = freqz(h_fir, 1, worN=2000)
        else: 
            w, h_resp = freqz(b_iir, a_iir, worN=2000)

        # 1. Время (До/После)
        main_axes[0].plot(t_axis[:N_pts]*1000, x_noisy[:N_pts], color='orange', alpha=0.3, label='Вход (Грязный)')
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, label='Выход (Чистый)')
        main_axes[0].set_title(f"Очистка во времени (SNR: {snr:.2f} dB)", fontweight='bold')
        main_axes[0].legend(loc='upper right'); main_axes[0].set_xlabel("ms")

        # 2. Спектр (До/После)
        f, m_in = get_spectrum(x_noisy); _, m_out = get_spectrum(sig)
        main_axes[1].fill_between(f, m_in, color='orange', alpha=0.15, label='Вход')
        main_axes[1].plot(f, m_out, color=col, lw=1.5, label='После фильтра')
        main_axes[1].set_title("Спектральная очистка (Видно подавление шума)", fontweight='bold')
        main_axes[1].set_xlim(0, 2500); main_axes[1].legend(); main_axes[1].set_xlabel("Hz")

        # 3. Характеристика фильтра
        main_axes[2].plot(w/np.pi * (sr/2), 20 * np.log10(np.abs(h_resp) + 1e-9), color='black', lw=2)
        main_axes[2].set_title("АЧХ самого фильтра (Frequency Response), дБ", fontweight='bold')
        main_axes[2].set_ylabel("дБ"); main_axes[2].set_ylim(-60, 5); main_axes[2].set_xlabel("Hz")

    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor(main_axes, hover=True)
    fig.canvas.draw_idle()

def play_audio(event):
    if sd:
        set_status("Воспроизведение...", "orange"); sd.stop()
        sig = data_map[radio.value_selected][0]
        sd.play(sig / (np.max(np.abs(sig)) + 1e-9), sr)

def save_wav(event):
    audio_dir = os.path.join(BASE_DIR, "results", "audio", "lab2"); os.makedirs(audio_dir, exist_ok=True)
    sig = data_map[radio.value_selected][0]
    name = radio.value_selected[3:].lower().replace(" ", "_")
    wavfile.write(os.path.join(audio_dir, f"var{VARIANT}_{name}.wav"), sr, np.int16(sig/np.max(np.abs(sig)) * 32767))
    set_status(f"WAV сохранен: results/audio/lab2/", "darkgreen")

def save_res(event):
    plots_dir = os.path.join(BASE_DIR, "results", "graphs", "lab2"); os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"var{VARIANT}_{radio.value_selected[:2]}.png"), dpi=150)
    set_status("График сохранен!", "darkgreen")

radio.on_clicked(update_plots); btn_play.on_clicked(play_audio); btn_save_wav.on_clicked(save_wav); btn_save_res.on_clicked(save_res)
update_plots(list(data_map.keys())[0]); plt.show()
