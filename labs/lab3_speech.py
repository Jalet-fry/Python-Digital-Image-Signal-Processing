import sys
import os
import matplotlib.pyplot as plt
import sounddevice as sd

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from logic.lab3_logic import SpeechProcessor
from ui.lab3_view import Lab3View
from core.utils.themes import UIColors
from core.utils.aspects import DSPContext

# Инициализация DeepFilterNet (если есть)
DF_AVAILABLE = False
df_model, df_state = None, None
try:
    import torch
    from df.enhance import init_df
    df_model, df_state, _ = init_df()
    DF_AVAILABLE = True
except: pass

def main():
    VARIANT = 10
    DSPContext.variant = VARIANT
    DSPContext.current_lab = "lab3"

    processor = SpeechProcessor(df_model, df_state)
    
    # Поиск файлов
    SOURCE_DIR = os.path.join(BASE_DIR, "source_audio_lab3")
    audio_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.wav')]) if os.path.exists(SOURCE_DIR) else []
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor=UIColors.BG_DARK)
    view = Lab3View(fig, audio_files)

    def on_run(event):
        fname = view.radio.value_selected
        path = os.path.join(SOURCE_DIR, fname)
        view.set_status(f"Обработка {fname}...")
        
        res = processor.process_file(path, view.snr_slider.val, use_df=DF_AVAILABLE)
        view.update(res, view.snr_slider.val)
        view.set_status("Готово")
        
        # Сохраняем результат в бэкенд для проигрывания
        processor.current_res = res

    def on_play_orig(event):
        if hasattr(processor, 'current_res'):
            sd.stop()
            sd.play(processor.current_res['clean'], processor.current_res['sr'])

    def on_play_proc(event):
        if hasattr(processor, 'current_res'):
            sd.stop()
            sd.play(processor.current_res['enhanced'], processor.current_res['sr'])

    view.btn_run.on_clicked(on_run)
    view.btn_orig.on_clicked(on_play_orig)
    view.btn_proc.on_clicked(on_play_proc)

    plt.show()

if __name__ == "__main__":
    main()
