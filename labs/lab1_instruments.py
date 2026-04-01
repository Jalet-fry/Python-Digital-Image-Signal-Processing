import sys
import os
import matplotlib.pyplot as plt

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

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
    
    # 2. ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ (Оптимизация)
    # Показываем статус в консоли, пока считаем тяжелое ДПФ
    print(">>> [LOGIC] Подготовка данных (расчет ДПФ/БПФ/Сверток)...")
    processor.prepare()
    print(">>> [LOGIC] Данные готовы.")

    # 3. Инициализация UI
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    view = Lab1View(fig, cfg)

    def on_change(label):
        # UI просто берет готовые результаты из памяти
        view.update_plots(label, processor.results)

    view.radio.on_clicked(on_change)
    on_change(view.menu_labels[0])
    
    plt.show(block=True)

if __name__ == "__main__":
    main()
