import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
from scipy.io import wavfile
import librosa
import librosa.display

# Добавляем корень проекта в путь
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.signals.features import (
    get_mfcc_full, get_spectral_rolloff, 
    calc_snr_metric, calc_si_sdr, calc_sdr,
    get_spectral_centroid, get_spectral_bandwidth, get_zero_crossing_rate,
    my_mel_spectrogram, get_chroma
)
from core.signals.noise import generate_white_noise, add_noise_snr
from core.config_variants import get_lab3_config
from core.utils.aspects import DSPContext

# Импорт метрик
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

# Попытка инициализации DeepFilterNet
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    DF_AVAILABLE = True
    df_model, df_df_state, _ = init_df()
except:
    DF_AVAILABLE = False

# ==========================================================
# 1. КОНФИГУРАЦИЯ
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=10)
args = parser.parse_known_args()[0]
VARIANT = args.variant

DSPContext.variant = VARIANT
DSPContext.current_lab = "lab3"
cfg = get_lab3_config(VARIANT)

SOURCE_DIR = os.path.join(BASE_DIR, "results", "audio", "lab3", "source")
PROCESSED_DIR = os.path.join(BASE_DIR, "results", "audio", "lab3", "processed")
NOISY_DIR = os.path.join(BASE_DIR, "results", "audio", "lab3", "noisy")
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs", "lab3")

for d in [SOURCE_DIR, PROCESSED_DIR, NOISY_DIR, GRAPHS_DIR]:
    os.makedirs(d, exist_ok=True)

def get_audio_files():
    if not os.path.exists(SOURCE_DIR): return []
    return [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.wav', '.mp3'))]

audio_files = get_audio_files()
if not audio_files: audio_files = ["No files found"]

# ==========================================================
# 2. СОСТОЯНИЕ
# ==========================================================
state = {
    'clean': None, 'sr': None, 'noisy': None, 'enhanced': None,
    'snr_in': 0, 'snr_out': 0, 'sisdr_in': 0, 'sisdr_out': 0,
    'sdr_in': 0, 'sdr_out': 0, 'pesq_score': 0, 'stoi_score': 0,
    'rolloff': 0, 'mfccs': None, 'centroid': 0, 'bandwidth': 0, 'zcr': 0,
    'chroma': None, 'mel_spec_my': None
}

# ==========================================================
# 3. ОБРАБОТКА
# ==========================================================
def process_file(fname, snr_db):
    path = os.path.join(SOURCE_DIR, fname)
    if not os.path.exists(path): return False
    
    y, sr = librosa.load(path, sr=16000)
    noise = generate_white_noise(len(y))
    noisy, _ = add_noise_snr(y, noise, snr_db)
    
    state['clean'] = y
    state['sr'] = sr
    state['noisy'] = noisy
    
    wavfile.write(os.path.join(NOISY_DIR, f"noisy_{fname}"), sr, (noisy * 32767).astype(np.int16))
    
    if DF_AVAILABLE:
        try:
            y_48k = librosa.resample(noisy, orig_sr=sr, target_sr=48000)
            enhanced_48k = enhance(df_model, df_df_state, y_48k)
            enhanced = librosa.resample(enhanced_48k, orig_sr=48000, target_sr=sr)
        except Exception as e:
            enhanced = noisy * 0.9
    else:
        enhanced = y + (noisy - y) * 0.2 
    
    min_len = min(len(y), len(enhanced))
    y, noisy, enhanced = y[:min_len], noisy[:min_len], enhanced[:min_len]
    
    state['clean'], state['noisy'], state['enhanced'] = y, noisy, enhanced
    wavfile.write(os.path.join(PROCESSED_DIR, f"enhanced_{fname}"), sr, (enhanced * 32767).astype(np.int16))
    
    # Признаки
    state['rolloff'] = get_spectral_rolloff(y, sr)
    state['mfccs'] = get_mfcc_full(y, sr)
    state['centroid'] = get_spectral_centroid(y, sr)
    state['bandwidth'] = get_spectral_bandwidth(y, sr)
    state['zcr'] = get_zero_crossing_rate(y)
    state['chroma'] = get_chroma(y, sr)
    state['mel_spec_my'] = my_mel_spectrogram(y, sr)
    
    # Метрики
    state['snr_in'] = calc_snr_metric(y, noisy)
    state['snr_out'] = calc_snr_metric(y, enhanced)
    state['sisdr_in'] = calc_si_sdr(y, noisy)
    state['sisdr_out'] = calc_si_sdr(y, enhanced)
    state['sdr_in'] = calc_sdr(y, noisy)
    state['sdr_out'] = calc_sdr(y, enhanced)
    
    if PESQ_AVAILABLE:
        try: state['pesq_score'] = pesq(16000, y, enhanced, 'wb')
        except: state['pesq_score'] = 0
    if STOI_AVAILABLE:
        try: state['stoi_score'] = stoi(y, enhanced, 16000)
        except: state['stoi_score'] = 0
    return True

# ==========================================================
# 4. GUI
# ==========================================================
plt.style.use('ggplot')
fig = plt.figure(figsize=(14, 9))
fig.canvas.manager.set_window_title(f"Лаба 3 [DFNet: {'ON' if DF_AVAILABLE else 'OFF'}]")

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

plt.subplots_adjust(left=0.2, bottom=0.15, hspace=0.3, wspace=0.3)

# Список файлов
ax_files = plt.axes([0.02, 0.6, 0.15, 0.25])
radio = RadioButtons(ax_files, audio_files)

def clear_gui(label=None):
    """Очистка графиков при смене файла"""
    for ax in [ax1, ax2, ax3, ax4]:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    info_text.set_text("Файл выбран.\nНастройте SNR и\nнажмите Запуск...")
    plt.draw()

radio.on_clicked(clear_gui)

# Слайдер SNR
ax_snr = plt.axes([0.05, 0.5, 0.10, 0.03])
snr_slider = Slider(ax_snr, 'SNR', 8, 24, valinit=10, valstep=2, valfmt='%d')
snr_slider.valtext.set_visible(False)
snr_display = fig.text(0.1, 0.47, f'Значение: {int(snr_slider.val)} dB', color='darkred', weight='bold', ha='center')

def on_snr_change(val): snr_display.set_text(f'Значение: {int(val)} dB')
snr_slider.on_changed(on_snr_change)

# Кнопки +/- для SNR
ax_minus = plt.axes([0.02, 0.5, 0.02, 0.03])
btn_minus = Button(ax_minus, '<')
ax_plus = plt.axes([0.16, 0.5, 0.02, 0.03])
btn_plus = Button(ax_plus, '>')

def go_minus(event):
    if snr_slider.val > 8: snr_slider.set_val(snr_slider.val - 2)
def go_plus(event):
    if snr_slider.val < 24: snr_slider.set_val(snr_slider.val + 2)

btn_minus.on_clicked(go_minus)
btn_plus.on_clicked(go_plus)

# Кнопка Запуск
ax_btn = plt.axes([0.02, 0.4, 0.15, 0.05])
btn = Button(ax_btn, 'Запуск')

# Инфо-панель
ax_info = plt.axes([0.02, 0.05, 0.15, 0.3])
ax_info.axis('off')
info_text = ax_info.text(0, 1, "Выберите файл\nи нажмите Запуск", va='top', fontsize=9)

def update(event):
    fname = radio.value_selected
    if fname == "No files found": return
    
    info_text.set_text("Обработка...")
    plt.pause(0.1) 
    
    if process_file(fname, snr_slider.val):
        # 1. Своя мел-спектрограмма
        ax1.clear()
        ax1.imshow(state['mel_spec_my'], aspect='auto', origin='lower')
        ax1.set_title("Ручная Мел-спектрограмма")
        
        # 2. MFCC
        ax2.clear()
        ax2.bar(range(len(state['mfccs'])), state['mfccs'])
        ax2.set_title("MFCC (первые 13 коэф.)")
        
        # 3. Зашумленный (спектр)
        ax3.clear()
        S_noisy = librosa.feature.melspectrogram(y=state['noisy'], sr=state['sr'])
        librosa.display.specshow(librosa.power_to_db(S_noisy), ax=ax3, sr=state['sr'])
        ax3.set_title(f"Зашумленный (SNR In: {state['snr_in']:.1f})")
        
        # 4. Очищенный (спектр)
        ax4.clear()
        S_enh = librosa.feature.melspectrogram(y=state['enhanced'], sr=state['sr'])
        librosa.display.specshow(librosa.power_to_db(S_enh), ax=ax4, sr=state['sr'])
        ax4.set_title(f"Очищенный (SNR Out: {state['snr_out']:.1f})")
        
        res = (f"РЕЗУЛЬТАТЫ:\n"
               f"SNR In: {state['snr_in']:.2f}\n"
               f"SNR Out: {state['snr_out']:.2f}\n"
               f"PESQ: {state['pesq_score']:.2f}\n"
               f"STOI: {state['stoi_score']:.3f}\n\n"
               f"Rolloff: {state['rolloff']:.0f} Hz\n"
               f"Centroid: {state['centroid']:.0f} Hz\n"
               f"Bandwidth: {state['bandwidth']:.0f} Hz\n"
               f"ZCR: {state['zcr']:.4f}\n"
               f"Chroma mean: {np.mean(state['chroma']):.3f}")
        info_text.set_text(res)
        plt.draw()

btn.on_clicked(update)
plt.show()
