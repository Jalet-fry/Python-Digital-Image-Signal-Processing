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
from core.signals.filters import moving_average_recursive, fir_manual_filter, fir_window_design, iir_design, apply_iir, iir_bandpass
from core.config_variants import get_lab1_config, get_lab2_config
from core.utils.aspects import DSPContext, log_dsp_action

# ==========================================================
# 1. ПОДГОТОВКА ДАННЫХ И РАСЧЕТЫ
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

DSPContext.variant = VARIANT
DSPContext.current_lab = "lab2"

cfg1 = get_lab1_config(VARIANT)
cfg2 = get_lab2_config(VARIANT)

sr = 8000
duration = 2.0 
t_axis = np.linspace(0, duration, int(sr * duration), endpoint=False)

# --- Генерация ---
_, x_clean = generate_instrument_signal(cfg1.x.amplitudes, cfg1.x.f0, cfg1.x.harmonics, 0, duration=duration, sr=sr)
np.random.seed(42)
white_noise = np.random.normal(0, 0.08, len(x_clean))
interference = 0.4 * np.sin(2 * np.pi * 1500 * t_axis)
total_noise = white_noise + interference
x_noisy = x_clean + total_noise

# --- Фильтрация ---
y_ma = moving_average_recursive(x_noisy, M=cfg2.M_ma)

f_low = cfg2.fir.f_range[0] if isinstance(cfg2.fir.f_range, (list, np.ndarray)) else cfg2.fir.f_range
f_high = cfg2.fir.f_range[1] if isinstance(cfg2.fir.f_range, (list, np.ndarray)) else sr/2 - 1
h_fir = fir_window_design(f_low, f_high, M=cfg2.fir.M, sr=sr, window_type=cfg2.fir.window)
y_fir = fir_manual_filter(x_noisy, h_fir)

b_iir, a_iir = iir_bandpass(cfg2.iir.f0, cfg2.iir.bw, sr=sr)
y_iir = apply_iir(x_noisy, b_iir, a_iir)

def estimate_delay(signal, filtered):
    """ Оценка задержки через взаимную корреляцию для точного SNR """
    corr = np.correlate(signal, filtered, mode='full')
    return max(0, np.argmax(np.abs(corr)) - len(signal) + 1)

@log_dsp_action
def calc_snr(clean, processed, delay=0):
    """ Расчет SNR с компенсацией задержки """
    c = clean
    p = processed
    if delay > 0:
        c = clean[:-delay]
        p = processed[delay:]
    noise_part = p - c
    return 10 * np.log10(np.sum(c**2) / (np.sum(noise_part**2) + 1e-12))

# Задержки
delay_ma = (cfg2.M_ma - 1) // 2
delay_fir = (cfg2.fir.M - 1) // 2
delay_iir = estimate_delay(x_clean, y_iir)

# Справочник данных для UI
data_map = {
    "1. Чистый сигнал (без шума)": {"sig": x_clean, "col": "#2980b9", "snr": None, "group": "clean"},
    "2. Только Шум (белый+писк)": {"sig": total_noise, "col": "#7f8c8d", "snr": None, "group": "noise"},
    "3. Зашумленный сигнал (вход)": {"sig": x_noisy, "col": "#e67e22", "snr": calc_snr(x_clean, x_noisy), "group": "noisy"},
    "4. Сравнение сигналов (все вместе)": {"sig": None, "col": None, "snr": None, "group": "comparison"},
    "5. Фильтр MA (рекурсивный)": {"sig": y_ma, "col": "#27ae60", "snr": calc_snr(x_clean, y_ma, delay_ma), "group": "filter"},
    "6. КИХ-фильтр (Blackman)": {"sig": y_fir, "col": "#c0392b", "snr": calc_snr(x_clean, y_fir, delay_fir), "group": "filter"},
    "7. БИХ-фильтр (резонансный)": {"sig": y_iir, "col": "#8e44ad", "snr": calc_snr(x_clean, y_iir, delay_iir), "group": "filter"}
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

filter_info_txt = f"MA(M={cfg2.M_ma}), FIR({f_low:.0f}-{f_high:.0f}Гц), IIR(f0={cfg2.iir.f0}Hz)"
status_text = fig.text(0.02, 0.03, f"Вариант {VARIANT} | {filter_info_txt}", fontsize=9, fontweight='bold', color='darkblue')

def set_status(msg, col='darkblue'):
    status_text.set_text(f"Вариант {VARIANT} | {msg}")
    status_text.set_color(col); fig.canvas.draw_idle()

main_axes = [fig.add_subplot(3, 1, i+1) for i in range(3)]
plt.subplots_adjust(left=0.28, right=0.96, top=0.94, bottom=0.08, hspace=0.45)
current_cursor = None

@log_dsp_action
def get_spectrum(sig, n=None, ref_peak=None):
    if n is None: n = len(sig)
    window = np.hanning(n)
    spectrum = np.fft.fft(sig[:n] * window)
    mag = np.abs(spectrum)[:n//2]
    freqs = np.fft.fftfreq(n, 1/sr)[:n//2]
    if ref_peak is not None:
        return freqs, 20 * np.log10(mag/ref_peak + 1e-12)
    return freqs, mag

_, m_lin = get_spectrum(x_noisy, ref_peak=None)
ref_peak = np.max(m_lin) if np.max(m_lin) > 1e-12 else 1.0

def update_plots(label):
    global current_cursor
    for ax in main_axes: ax.clear(); ax.set_axis_on()
    data = data_map[label]
    group, N_pts = data["group"], 500
    set_status(label)

    if group == "clean":
        sig, col = x_clean, "#2980b9"
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, linewidth=1.5)
        main_axes[0].set_title("Чистый сигнал", fontweight='bold')
        main_axes[0].set_xlabel("Время, мс"); main_axes[0].set_ylabel("Амплитуда"); main_axes[0].grid(True, alpha=0.3)
        f, m_db = get_spectrum(sig, ref_peak=ref_peak)
        main_axes[1].fill_between(f, -60, m_db, color=col, alpha=0.5)
        for fh in [cfg1.x.f0 * h for h in cfg1.x.harmonics]:
            main_axes[1].axvline(x=fh, color='red', linestyle='--', alpha=0.4)
        main_axes[1].set_title("Спектр чистого сигнала (дБ)", fontweight='bold'); main_axes[1].set_xlim(0, 2500); main_axes[1].set_ylim(-60, 5)
        main_axes[1].grid(True, alpha=0.3)
        main_axes[2].hist(sig, bins=70, color=col, alpha=0.7); main_axes[2].set_title("Распределение амплитуд")

    elif group == "noise":
        sig, col = total_noise, "#7f8c8d"
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, linewidth=1)
        main_axes[0].set_title("Шум (белый + наводка 1500 Гц)"); main_axes[0].grid(True, alpha=0.3)
        f, m_db = get_spectrum(sig, ref_peak=ref_peak)
        main_axes[1].fill_between(f, -60, m_db, color=col, alpha=0.5)
        main_axes[1].axvline(x=1500, color='red', linestyle='--', alpha=0.7, label='Помеха 1500 Гц')
        main_axes[1].set_title("Спектр шума (дБ)"); main_axes[1].set_xlim(0, 2500); main_axes[1].set_ylim(-60, 5)
        main_axes[1].legend(); main_axes[1].grid(True, alpha=0.3)
        main_axes[2].hist(sig, bins=70, color=col, alpha=0.7)

    elif group == "noisy":
        sig, col, snr = x_noisy, "#e67e22", data["snr"]
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, linewidth=1.5)
        main_axes[0].set_title(f"Зашумленный сигнал (SNR = {snr:.2f} dB)"); main_axes[0].grid(True, alpha=0.3)
        f, m_db = get_spectrum(sig, ref_peak=ref_peak)
        main_axes[1].fill_between(f, -60, m_db, color=col, alpha=0.5)
        main_axes[1].set_title("Спектр (дБ)"); main_axes[1].set_xlim(0, 2500); main_axes[1].set_ylim(-60, 5)
        main_axes[1].grid(True, alpha=0.3)
        main_axes[2].hist(sig, bins=70, color=col, alpha=0.7)

    elif group == "comparison":
        main_axes[0].plot(t_axis[:N_pts]*1000, x_clean[:N_pts], color="#2980b9", label='Чистый')
        main_axes[0].plot(t_axis[:N_pts]*1000, x_noisy[:N_pts], color="#e67e22", alpha=0.6, label='Зашумленный')
        main_axes[0].legend(); main_axes[0].set_title("Сравнение во времени"); main_axes[0].grid(True, alpha=0.3)
        f_c, m_c_db = get_spectrum(x_clean, ref_peak=ref_peak)
        f_n, m_n_db = get_spectrum(x_noisy, ref_peak=ref_peak)
        main_axes[1].plot(f_c, m_c_db, color="#2980b9", label='Чистый'); main_axes[1].plot(f_n, m_n_db, color="#e67e22", alpha=0.6, label='Зашумленный')
        main_axes[1].set_title("Сравнение спектров (дБ)"); main_axes[1].set_xlim(0, 2500); main_axes[1].set_ylim(-60, 5); main_axes[1].legend()
        main_axes[2].hist(x_clean, bins=50, alpha=0.5, color='#2980b9', label='Чистый'); main_axes[2].hist(x_noisy, bins=50, alpha=0.5, color='#e67e22', label='Зашумленный'); main_axes[2].legend()

    elif group == "filter":
        sig, col, snr = data["sig"], data["col"], data["snr"]
        if "MA" in label: b, a = np.ones(cfg2.M_ma)/cfg2.M_ma, [1.0]
        elif "КИХ" in label: b, a = h_fir, [1.0]
        else: b, a = b_iir, a_iir
        w, h_resp = freqz(b, a, worN=2000, fs=sr)
        main_axes[0].plot(t_axis[:N_pts]*1000, x_noisy[:N_pts], color='orange', alpha=0.3, label='Вход')
        main_axes[0].plot(t_axis[:N_pts]*1000, sig[:N_pts], color=col, lw=1.5, label='Выход')
        main_axes[0].set_title(f"Очистка (SNR: {snr:.2f} dB)"); main_axes[0].legend(); main_axes[0].grid(True, alpha=0.3)
        f_in, m_in_db = get_spectrum(x_noisy, ref_peak=ref_peak)
        f_out, m_out_db = get_spectrum(sig, ref_peak=ref_peak)
        main_axes[1].plot(f_in, m_in_db, color='orange', alpha=0.2, label='ДО'); main_axes[1].plot(f_out, m_out_db, color=col, label='ПОСЛЕ')
        main_axes[1].set_title("Спектральная очистка"); main_axes[1].set_xlim(0, 2500); main_axes[1].set_ylim(-60, 5); main_axes[1].legend()
        main_axes[2].plot(w, np.abs(h_resp), color='black', lw=2, label='АЧХ (лин)')
        ax_db = main_axes[2].twinx()
        ax_db.plot(w, 20*np.log10(np.abs(h_resp) + 1e-9), color='gray', alpha=0.5, linestyle='--', label='АЧХ (дБ)')
        main_axes[2].set_title("АЧХ фильтра"); main_axes[2].grid(True, alpha=0.3)
        l1, lb1 = main_axes[2].get_legend_handles_labels(); l2, lb2 = ax_db.get_legend_handles_labels()
        main_axes[2].legend(l1+l2, lb1+lb2, loc='upper right', fontsize=8)

    if current_cursor: current_cursor.remove()
    current_cursor = mplcursors.cursor([line for ax in main_axes for line in ax.get_lines()], hover=True)
    @current_cursor.connect("add")
    def _(sel):
        ax = sel.artist.axes; title = ax.get_title().lower()
        if "спектр" in title or "ачх" in title: sel.annotation.set_text(f"f = {sel.target[0]:.1f} Гц\nA = {sel.target[1]:.2f}")
        else: sel.annotation.set_text(f"t = {sel.target[0]:.2f} мс\nA = {sel.target[1]:.4f}")
    fig.canvas.draw_idle()

def play_audio(event):
    if sd:
        sd.stop(); label = radio.value_selected
        sig = data_map[label]["sig"] if data_map[label]["sig"] is not None else x_noisy
        sd.play(sig / (np.max(np.abs(sig)) + 1e-9), sr)

def save_wav(event):
    audio_dir = os.path.join(BASE_DIR, "results", "audio", "lab2"); os.makedirs(audio_dir, exist_ok=True)
    label = radio.value_selected; sig = data_map[label]["sig"] if data_map[label]["sig"] is not None else x_noisy
    wavfile.write(os.path.join(audio_dir, f"var{VARIANT}_{label[3:10]}.wav"), sr, np.int16(sig/np.max(np.abs(sig)) * 32767))
    set_status("WAV сохранен!")

def save_res(event):
    plots_dir = os.path.join(BASE_DIR, "results", "graphs", "lab2"); os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"var{VARIANT}_{radio.value_selected[:2]}.png"), dpi=150)
    set_status("График сохранен!")

radio.on_clicked(update_plots); btn_play.on_clicked(play_audio); btn_save_wav.on_clicked(save_wav); btn_save_res.on_clicked(save_res)
update_plots(list(data_map.keys())[0]); plt.show()
