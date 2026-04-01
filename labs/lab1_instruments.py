import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    import sounddevice as sd
except ImportError:
    sd = None

from logic.lab1_logic import InstrumentProcessor
from ui.lab1_view import Lab1View
from core.config_variants import get_lab1_config
from core.utils.aspects import DSPContext

def main():
    VARIANT = 10 
    DSPContext.variant = VARIANT
    DSPContext.current_lab = "lab1"
    cfg = get_lab1_config(VARIANT)

    # 1. Инициализация Бэкенда
    processor = InstrumentProcessor(cfg)
    
    print(">>> [LOGIC] Подготовка всех 24 графиков и расчет ошибок...")
    processor.prepare()
    print(">>> [LOGIC] Расчеты завершены.")

    # 2. Инициализация UI
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title(f'Лабораторная работа №1 - Вариант {VARIANT} ({cfg.x.name})')
    view = Lab1View(fig, cfg)

    # 3. Обработчики событий
    def on_change(label):
        view.update_plots(label, processor.results)

    def on_play(event):
        if sd is None:
            view.set_status("Error: sounddevice not installed", color="red")
            return
        
        view.set_status("Playing audio...", color="orange")
        sd.stop()
        label = view.radio.value_selected
        
        # Выбор сигнала: X или Y (или свертка если выбрана)
        if 'Y' in label or 'y(t)' in label.lower():
            sig = processor.results['audio_y']
        else:
            sig = processor.results['audio_x']
            
        sd.play(sig / (np.max(np.abs(sig)) + 1e-9), processor.results['sr_audio'])
        view.set_status(f"Playing: {label}", color="green")

    def on_save_wav(event):
        view.set_status("Saving WAV files...", color="orange")
        path = processor.save_wav(BASE_DIR)
        view.set_status(f"WAV saved to: {path}", color="green")

    def on_save_res(event):
        view.set_status("Saving current graph...", color="orange")
        plots_dir = os.path.join(BASE_DIR, "results", "graphs")
        os.makedirs(plots_dir, exist_ok=True)
        fname = f"var{VARIANT}_{view.radio.value_selected[:2].replace('.', '')}.png"
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        view.set_status(f"Graph saved: {fname}", color="green")

    # Привязка событий
    view.radio.on_clicked(on_change)
    view.btn_play.on_clicked(on_play)
    view.btn_save_wav.on_clicked(on_save_wav)
    view.btn_save_res.on_clicked(on_save_res)
    
    # Стартовая отрисовка
    on_change(view.menu_labels[0])
    
    plt.show(block=True)

if __name__ == "__main__":
    main()
