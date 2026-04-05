import sys
import os
import traceback
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from logic.lab3_logic import SpeechProcessor
from ui.lab3_view import Lab3View
from core.utils.themes import UIColors
from core.utils.aspects import DSPContext

# Инициализация DeepFilterNet
DF_AVAILABLE = False
df_model, df_state = None, None
try:
    import torch
    from df.enhance import init_df
    df_model, df_state, _ = init_df()
    DF_AVAILABLE = True
except Exception:
    pass

# Состояние для блокировки повторных нажатий
is_processing = False

def main():
    try:
        VARIANT = 10
        DSPContext.variant = VARIANT
        DSPContext.current_lab = "lab3"

        processor = SpeechProcessor(df_model, df_state)
        
        # Поиск файлов
        SOURCE_DIR = os.path.join(BASE_DIR, "source_audio_lab3")
        if not os.path.exists(SOURCE_DIR):
            os.makedirs(SOURCE_DIR, exist_ok=True)
            
        audio_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.wav', '.mp3'))])
        if not audio_files:
            audio_files = ["No files found"]
        
        fig = plt.figure(num='Лабораторная работа №3 - Анализ речи', figsize=(15, 9))
        if fig.canvas.manager:
            fig.canvas.manager.set_window_title(f'Лабораторная работа №3 (DeepFilterNet: {"ON" if DF_AVAILABLE else "OFF"})')
        
        view = Lab3View(fig, audio_files)

        def on_run(event):
            global is_processing
            if is_processing: return
            
            # УПРОЩЕНО: Берем имя файла напрямую из выбранного значения виджета
            fname = view.radio.value_selected
            
            if not fname or fname == "No files found":
                view.set_status("Files not found!", color="red")
                return

            is_processing = True
            view.btn_run.label.set_text("BUSY...")
            view.btn_run.color = '#718096'
            view.clear_gui()
            
            path = os.path.join(SOURCE_DIR, fname)
            view.set_status(f"Processing {fname}...", color="orange")
            
            try:
                res = processor.process_file(path, view.snr_slider.val, use_df=DF_AVAILABLE)
                if res:
                    view.update(res, int(view.snr_slider.val))
                    view.set_status("Complete", color="#00FF00")
                else:
                    view.set_status("Failed!", color="red")
            except Exception as e:
                view.set_status(f"Error: {str(e)}", color="red")
                print(traceback.format_exc())
            finally:
                is_processing = False
                view.btn_run.label.set_text("RUN PROCESSING")
                view.btn_run.color = UIColors.BTN_RUN
                view.fig.canvas.draw_idle()

        def on_play_clean(event):
            if processor.current_res:
                sd.stop()
                sd.play(processor.current_res['clean'], processor.current_res['sr'])

        def on_play_noisy(event):
            if processor.current_res:
                sd.stop()
                sd.play(processor.current_res['noisy'], processor.current_res['sr'])

        def on_play_enh(event):
            if processor.current_res:
                sd.stop()
                sd.play(processor.current_res['enhanced'], processor.current_res['sr'])

        def on_stop(event):
            sd.stop()

        view.btn_run.on_clicked(on_run)
        view.btn_orig.on_clicked(on_play_clean)
        view.btn_noisy.on_clicked(on_play_noisy)
        view.btn_proc.on_clicked(on_play_enh)
        view.btn_stop.on_clicked(on_stop)

        plt.show()

    except Exception:
        print("\n" + "="*50)
        print("КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ LAB3:")
        print("="*50)
        traceback.print_exc()
        print("="*50)
        input("\nНажмите ENTER, чтобы закрыть это окно...")

if __name__ == "__main__":
    main()
