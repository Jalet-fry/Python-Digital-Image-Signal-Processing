import os
import sys

# Добавляем корень проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.lab4_logic import Lab4AIProcessor

# ПУТИ
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MY_VOICE = os.path.join(BASE_DIR, "source_audio_lab3", "voice_1_mono_miner.wav")
TARGET_VOICE = os.path.join(BASE_DIR, "source_audio_lab3", "voice_2_stereo_avsetaki.wav")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    processor = Lab4AIProcessor()
    
    output_path = os.path.join(OUTPUT_DIR, "vc_output.wav")
    
    print("--- [Задание 2] Neural Voice Conversion (Zero-Shot) ---")
    
    # В текущей реализации logic это Zero-Shot VC (Text-based Cloning)
    processor.voice_conversion(MY_VOICE, TARGET_VOICE, output_path)
    print(f"Готово! Результат: {output_path}")

if __name__ == "__main__":
    main()
