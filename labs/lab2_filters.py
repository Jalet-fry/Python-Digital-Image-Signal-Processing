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
from core.signals.filters import moving_average_recursive, fir_manual_filter, fir_window_bandpass, iir_design, apply_iir, iir_bandpass
from core.config_variants import get_lab_config
from core.utils.aspects import DSPContext

# ==========================================================
# 1. ПОДГОТОВКА ДАННЫХ И РАСЧЕТЫ
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

# --- Генерация ---
_, x_clean = generate_instrument_signal(lab1_cfg['x']['A'], lab1_cfg['x']['f0'], lab1_cfg['x']['h'], 0, duration=duration, sr=sr)
np.random.seed(42)
white_noise = np.random.normal(0, 0.08, len(x_clean))
interference = 0.4 * np.sin(2 * np.pi * 1500 * t_axis)
total_noise = white_noise + interference
x_noisy = x_clean + total_noise

# --- Фильтрация ---
y_ma = moving_average_recursive(x_noisy, M=lab2_cfg['M_ma'])

fir_config = lab2_cfg['fir']
f_v = fir_config['f']
f_low = f_v[0] if isinstance(f_v, (list, np.ndarray)) else f_v
f_high = f_v[1] if isinstance(f_v, (list, np.ndarray)) else sr/2 - 1
h_fir = fir_window_bandpass(f_low, f_high, M=fir_config['M'], sr=sr)
y_fir = fir_manual_filter(x_noisy, h_fir)

b_iir, a_iir = iir_bandpass(lab2_cfg['iir']['f0'], lab2_cfg['iir']['bw'], sr=sr)
y_iir = apply_iir(x_noisy, b_iir, a_iir)

def calc_snr(clean, processed):
    noise_part = processed - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise_part**2) + 1e-12))

# Справочник данных для UI
data_map = {
    "1. Чистый сигнал (без шума)": {"sig": x_clean, "col": "#2980b9", "snr": None, "group": "clean"},
    "2. Только Шум (белый+писк)": {"sig": total_noise, "col": "#7f8c8d", "snr": None, "group": "noise"},
    "3. Зашумленный сигнал (вход)": {"sig": x_noisy, "col": "#e67e22", "snr": calc_snr(x_clean, x_noisy), "group": "noisy"},
    "4. Сравнение сигналов (все вместе)": {"sig": None, "col": None, "snr": None, "group": "comparison"},
    "5. Фильтр MA (рекурсивный)": {"sig": y_ma, "col": "#27ae60", "snr": calc_snr(x_clean, y_ma), "group": "filter"},
    "6. КИХ-фильтр (Blackman)": {"sig": y_fir, "col": "#c0392b", "snr": calc_snr(x_clean, y_fir), "group": "filter"},
    "7. БИХ-фильтр (резонансный)": {"sig": y_iir, "col": "#8e44ad", "snr": calc_snr(x_clean, y_iir), "group": "filter"}
}

try: import sounddevice as sd
except ImportError: sd = None

# ==========================================================
# 2. ИНТЕРФЕЙС
# ==========================================================
plt.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-v0_8-muted')
fig = plt.figure(figsize=(15, 10))
fig.canvas.manager.set_window_title(f"ЛР №2: Проектирование фильтров - Вариант {VARIANT}")

ax_menu = plt.axes([0.02, 0.5, 0.2, 0.35], facecolor='#f8f9fa')
radio = RadioButtons(ax_menu, list(data_map.keys()), active=0, activecolor='royalblue')
ax_play = plt.axes([0.02, 0.4, 0.2, 0.06]); btn_play = Button(ax_play, '▶ Play Selected', color='#d4edda')
ax_save_wav = plt.axes([0.02, 0.32, 0.2, 0.06]); btn_save_wav = Button(ax_save_wav, '💾 Save WAV', color='#d1ecf1')
ax_save_res = plt.axes([0.02, 0.24, 0.2, 0.06]); btn_save_res = Button(ax_save_res, '📊 Save Results', color='#fff3cd')

filter_info_txt = f"MA(M={lab2_cfg['M_ma']}), FIR({f_low:.0f}-{f_high:.0f}Гц), IIR(f0={lab2_cfg['iir']['f0']}Hz)"
status_text = fig.text(0.02, 0.03, f"Вариант {VARIANT} | {filter_info_txt}", fontsize=9, fontweight='bold', color='darkblue')

def set_status(msg, col='darkblue'):
    status_text.set_text(f"Вариант {VARIANT} | {msg}")
    status_text.set_color(col); fig.canvas.draw_idle()

main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
plt.subplots_adjust(left=0.28, right=0.96, top=0.94, bottom=0.08, hspace=0.45)
current_cursor = None

def get_spectrum(sig, n=2048):
    spectrum = np.fft.fft(sig[:n])
    mag = np.abs(spectrum)[:n//2]
    if np.max(mag) > 1e-12: mag = mag / np.max(mag)
    freqs = np.fft.fftfreq(n, 1/sr)[:n//2]
    return freqs, mag

def save_wav(event):
    audio_dir = os.path.join(BASE_DIR, "results", "audio", "lab2"); os.makedirs(audio_dir, exist_ok=True)
    label = radio.value_selected
    sig = data_map[label]["sig"] if data_map[label]["sig"] is not None else x_noisy
    name = label[3:].lower().replace(" ", "_")
    wavfile.write(os.path.join(audio_dir, f"var{VARIANT}_{name}.wav"), sr, np.int16(sig/np.max(np.abs(sig)) * 32767))
    set_status(f"WAV сохранен!")

def save_res(event):
    plots_dir = os.path.join(BASE_DIR, "results", "graphs", "lab2"); os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"var{VARIANT}_{radio.value_selected[:2]}.png"), dpi=150)
    set_status("График сохранен!")

def update_plots(label):
    global current_cursor
    for ax in main_axes:
        ax.clear()
        ax.set_axis_on()

    data = data_map[label]
    group = data["group"]
    N_pts = 500
    set_status(label)

    if group == "clean":
        sig, col = x_clean, "#2980b9"
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, linewidth=1.5)
        main_axes[0].set_title("Чистый сигнал (виолончель, 110 Гц)", fontweight='bold')
        main_axes[0].set_xlabel("Время, мс"); main_axes[0].set_ylabel("Амплитуда")
        f, m = get_spectrum(sig); main_axes[1].fill_between(f, m, color=col, alpha=0.5)
        for freq_h in [110, 220, 330, 440]: main_axes[1].axvline(x=freq_h, color='red', linestyle='--', alpha=0.5)
        main_axes[1].set_title("Спектр чистого сигнала", fontweight='bold'); main_axes[1].set_xlim(0, 2500)
        main_axes[2].hist(sig, bins=70, color=col, alpha=0.7, edgecolor='black', lw=0.5)
        main_axes[2].set_title("Распределение амплитуд", fontweight='bold')

    elif group == "noise":
        sig, col = total_noise, "#7f8c8d"
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, linewidth=1)
        main_axes[0].set_title("Шум (белый + наводка 1500 Гц)", fontweight='bold')
        f, m = get_spectrum(sig); main_axes[1].fill_between(f, m, color=col, alpha=0.5)
        main_axes[1].axvline(x=1500, color='red', linestyle='--', alpha=0.7); main_axes[1].set_xlim(0, 2500)
        main_axes[2].hist(sig, bins=70, color=col, alpha=0.7, edgecolor='black', lw=0.5)
        main_axes[2].set_title("Гистограмма шума", fontweight='bold')

    elif group == "noisy":
        sig, col, snr = x_noisy, "#e67e22", data["snr"]
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, linewidth=1.5)
        main_axes[0].set_title(f"Зашумленный сигнал (SNR = {snr:.2f} dB)", fontweight='bold')
        f, m = get_spectrum(sig); main_axes[1].fill_between(f, m, color=col, alpha=0.5)
        for freq_h in [110, 220, 330, 440, 1500]: main_axes[1].axvline(x=freq_h, color='red', linestyle='--', alpha=0.3)
        main_axes[1].set_title("Спектр зашумленного сигнала", fontweight='bold'); main_axes[1].set_xlim(0, 2500)
        main_axes[2].hist(sig, bins=70, color=col, alpha=0.7, edgecolor='black', lw=0.5)

    elif group == "comparison":
        # ===== ГРАФИК 1: Временная область (все три сигнала) =====
        main_axes[0].plot(t_axis[:N_pts]*1000, x_clean[:N_pts],
                         color="#2980b9", alpha=0.8, label='Чистый сигнал', linewidth=1.5)
        main_axes[0].plot(t_axis[:N_pts]*1000, total_noise[:N_pts],
                         color="#7f8c8d", alpha=0.5, label='Шум (белый + 1500 Гц)', linewidth=1)
        main_axes[0].plot(t_axis[:N_pts]*1000, x_noisy[:N_pts],
                         color="#e67e22", alpha=0.8, label='Зашумленный (чистый + шум)', linewidth=1.5)
        main_axes[0].set_title("Сравнение сигналов во времени", fontweight='bold', fontsize=11)
        main_axes[0].set_xlabel("Время, мс", fontsize=9)
        main_axes[0].set_ylabel("Амплитуда", fontsize=9)
        main_axes[0].legend(loc='upper right', fontsize=8)
        main_axes[0].grid(True, alpha=0.3, linestyle='--')

        # Добавляем SNR
        snr_val = calc_snr(x_clean, x_noisy)
        main_axes[0].text(0.95, 0.95, f'SNR = {snr_val:.2f} dB',
                         transform=main_axes[0].transAxes, ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                                  edgecolor='#e67e22'))

        # ===== ГРАФИК 2: Спектры всех сигналов =====
        f_clean, m_clean = get_spectrum(x_clean)
        f_noise, m_noise = get_spectrum(total_noise)
        f_noisy, m_noisy = get_spectrum(x_noisy)

        # Чистый сигнал
        main_axes[1].plot(f_clean, m_clean, color="#2980b9", linewidth=1.5,
                         label='Чистый сигнал')
        # Шум
        main_axes[1].fill_between(f_noise, m_noise, color="#7f8c8d", alpha=0.3,
                                  label='Шум')
        # Зашумленный
        main_axes[1].plot(f_noisy, m_noisy, color="#e67e22", linewidth=1.5,
                         label='Зашумленный')

        # Отмечаем гармоники виолончели (как было в хороших графиках)
        for idx, freq_h in enumerate([110, 220, 330, 440]):
            main_axes[1].axvline(x=freq_h, color='red', linestyle='--', alpha=0.5,
                                label=f'{freq_h} Гц' if idx == 0 else '')

        # Отмечаем наводку 1500 Гц
        main_axes[1].axvline(x=1500, color='purple', linestyle='-', alpha=0.7,
                            label='Наводка 1500 Гц', linewidth=2)

        main_axes[1].set_title("Сравнение спектров", fontweight='bold', fontsize=11)
        main_axes[1].set_xlim(0, 2500)
        main_axes[1].set_xlabel("Частота, Гц", fontsize=9)
        main_axes[1].set_ylabel("Нормированная амплитуда", fontsize=9)
        main_axes[1].legend(loc='upper right', fontsize=7)
        main_axes[1].grid(True, alpha=0.3, linestyle='--')

        # ===== ГРАФИК 3: Гистограммы для сравнения =====
        main_axes[2].hist(x_clean, bins=50, alpha=0.5, color='#2980b9',
                         label='Чистый', edgecolor='black', linewidth=0.5)
        main_axes[2].hist(x_noisy, bins=50, alpha=0.5, color='#e67e22',
                         label='Зашумленный', edgecolor='black', linewidth=0.5)
        main_axes[2].set_title("Сравнение распределений амплитуд", fontweight='bold', fontsize=11)
        main_axes[2].set_xlabel("Амплитуда", fontsize=9)
        main_axes[2].set_ylabel("Количество", fontsize=9)
        main_axes[2].legend(loc='upper right', fontsize=7)
        main_axes[2].grid(True, alpha=0.3, linestyle='--')

    elif group == "filter":
        sig, col, snr = data["sig"], data["col"], data["snr"]
        if "MA" in label: w, h_resp = freqz(np.ones(lab2_cfg['M_ma'])/lab2_cfg['M_ma'], 1, worN=2000); p_band = (0, sr/(2*lab2_cfg['M_ma']))
        elif "КИХ" in label: w, h_resp = freqz(h_fir, 1, worN=2000); p_band = (f_low, f_high)
        else: w, h_resp = freqz(b_iir, a_iir, worN=2000); p_band = (lab2_cfg['iir']['f0']-lab2_cfg['iir']['bw']/2, lab2_cfg['iir']['f0']+lab2_cfg['iir']['bw']/2)

        main_axes[0].plot(t_axis[:N_pts]*1000, x_noisy[:N_pts], color='orange', alpha=0.4, label='Вход')
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, label='Выход')
        main_axes[0].set_title(f"Результат фильтрации (SNR: {snr:.2f} dB)", fontweight='bold'); main_axes[0].legend()
        f_in, m_in = get_spectrum(x_noisy); f_out, m_out = get_spectrum(sig)
        main_axes[1].fill_between(f_in, m_in, color='orange', alpha=0.15, label='Вход')
        main_axes[1].plot(f_out, m_out, color=col, lw=2, label='Выход'); main_axes[1].axvspan(p_band[0], p_band[1], alpha=0.2, color='green')
        main_axes[1].set_title("Спектральная очистка", fontweight='bold'); main_axes[1].set_xlim(0, 2500)
        main_axes[2].plot(w/np.pi * (sr/2), 20 * np.log10(np.abs(h_resp) + 1e-9), color='black', lw=2)
        main_axes[2].axhline(y=-3, color='red', linestyle='--'); main_axes[2].set_ylim(-60, 5)
        main_axes[2].set_title("АЧХ фильтра", fontweight='bold')

    for ax in main_axes: ax.grid(True, alpha=0.3, linestyle='--')
    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor(main_axes, hover=True)
    @current_cursor.connect("add")
    def _(sel):
        ax = sel.artist.axes
        title = ax.get_title().lower()
        if "спектр" in title or "ачх" in title: sel.annotation.set_text(f"f = {sel.target[0]:.1f} Гц\nA = {sel.target[1]:.2f}")
        else: sel.annotation.set_text(f"t = {sel.target[0]:.2f} мс\nA = {sel.target[1]:.4f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, ec="black", lw=0.5, boxstyle="round,pad=0.3")
    fig.canvas.draw_idle()

def play_audio(event):
    if sd:
        sd.stop()
        label = radio.value_selected
        sig = data_map[label]["sig"] if data_map[label]["sig"] is not None else x_noisy
        sd.play(sig / (np.max(np.abs(sig)) + 1e-9), sr)

def save_wav(event):
    audio_dir = os.path.join(BASE_DIR, "results", "audio", "lab2"); os.makedirs(audio_dir, exist_ok=True)
    label = radio.value_selected
    sig = data_map[label]["sig"] if data_map[label]["sig"] is not None else x_noisy
    name = label[3:].lower().replace(" ", "_")
    wavfile.write(os.path.join(audio_dir, f"var{VARIANT}_{name}.wav"), sr, np.int16(sig/np.max(np.abs(sig)) * 32767))
    set_status(f"WAV сохранен!")

def save_res(event):
    plots_dir = os.path.join(BASE_DIR, "results", "graphs", "lab2"); os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"var{VARIANT}_{radio.value_selected[:2]}.png"), dpi=150)
    set_status("График сохранен!")

radio.on_clicked(update_plots); btn_play.on_clicked(play_audio); btn_save_wav.on_clicked(save_wav); btn_save_res.on_clicked(save_res)
update_plots(list(data_map.keys())[0]); plt.show()
