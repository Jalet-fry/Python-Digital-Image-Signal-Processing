import sys
import os
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from scipy.io import wavfile

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from logic.lab2_logic import FilterProcessor
from ui.lab2_view import Lab2View
from core.config_variants import get_lab1_config, get_lab2_config
from core.utils.aspects import DSPContext

def main():
    VARIANT = 10
    DSPContext.variant = VARIANT
    DSPContext.current_lab = "lab2"
    
    cfg1 = get_lab1_config(VARIANT)
    cfg2 = get_lab2_config(VARIANT)

    # 1. Инициализация Бэкенда
    processor = FilterProcessor(cfg1, cfg2)
    print(">>> [LOGIC] Расчет всех фильтров и SNR с компенсацией задержек...")
    processor.prepare()

    # 2. Инициализация UI
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title(f'Лабораторная работа №2 - Вариант {VARIANT} (Фильтрация)')
    view = Lab2View(fig, cfg2)

    # 3. Обработчики событий
    def on_change(label):
        view.set_status(f"Selected: {label}")
        view.update(label, processor.results)
        
    def on_play(event):
        view.set_status("Playing selected...", color="orange")
        label = view.radio.value_selected
        res = processor.results
        
        # Логика выбора сигнала
        if "Чистый" in label: sig = res['x_clean']
        elif "Шум" in label: sig = res['noise']
        elif "Зашумленный" in label: sig = res['x_noisy']
        elif "MA" in label: sig = res['y_ma']
        elif "КИХ" in label: sig = res['y_fir']
        elif "БИХ" in label: sig = res['y_iir']
        else: sig = res['x_noisy']
        
        sd.stop()
        sd.play(sig / (np.max(np.abs(sig)) + 1e-9), res['sr'])
        view.set_status(f"Playing: {label}", color="green")

    def on_save_wav(event):
        view.set_status("Saving WAV...", color="orange")
        audio_dir = os.path.join(BASE_DIR, "results", "audio", "lab2")
        os.makedirs(audio_dir, exist_ok=True)
        
        label = view.radio.value_selected
        res = processor.results
        sig = res.get('x_noisy') # default
        if "Чистый" in label: sig = res['x_clean']
        elif "MA" in label: sig = res['y_ma']
        elif "КИХ" in label: sig = res['y_fir']
        elif "БИХ" in label: sig = res['y_iir']
        
        fname = f"var{VARIANT}_{label[3:10].strip().replace(' ', '_')}.wav"
        wavfile.write(os.path.join(audio_dir, fname), res['sr'], np.int16(sig/np.max(np.abs(sig)) * 32767))
        view.set_status(f"WAV saved: {fname}", color="darkgreen")

    def on_save_res(event):
        view.set_status("Saving graph...", color="orange")
        plots_dir = os.path.join(BASE_DIR, "results", "graphs", "lab2")
        os.makedirs(plots_dir, exist_ok=True)
        fname = f"var{VARIANT}_{view.radio.value_selected[:2].replace('.', '')}.png"
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        view.set_status(f"Graph saved: {fname}", color="darkgreen")

    # Привязка
    view.radio.on_clicked(on_change)
    view.btn_play.on_clicked(on_play)
    if hasattr(view, 'btn_save_wav'): view.btn_save_wav.on_clicked(on_save_wav)
    if hasattr(view, 'btn_save_res'): view.btn_save_res.on_clicked(on_save_res)
    
    # Запуск
    on_change(view.labels[0])
    plt.show(block=True)

if __name__ == "__main__":
    main()
