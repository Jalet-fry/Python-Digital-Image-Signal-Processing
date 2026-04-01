import sys
import os
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

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
    print(">>> [LOGIC] Расчет фильтров (MA, FIR, IIR)...")
    processor.prepare()

    # 2. Инициализация UI
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    view = Lab2View(fig, cfg2)

    # 3. Обработчики событий
    def on_change(label):
        # Берем уже готовые результаты
        view.update(label, processor.results)
        
    def on_play(event):
        label = view.radio.value_selected
        res = processor.results
        sig = res.get('x_noisy')
        if "Чистый" in label: sig = res['x_clean']
        elif "MA" in label: sig = res['y_ma']
        elif "КИХ" in label: sig = res['y_fir']
        elif "БИХ" in label: sig = res['y_iir']
        
        sd.stop()
        sd.play(sig / (np.max(np.abs(sig)) + 1e-9), res['sr'])

    view.radio.on_clicked(on_change)
    view.btn_play.on_clicked(on_play)
    
    # Запуск
    on_change(view.labels[0])
    plt.show(block=True)

if __name__ == "__main__":
    main()
