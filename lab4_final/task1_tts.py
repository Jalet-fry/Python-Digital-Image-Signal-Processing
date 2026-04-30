import os
import sys

# Добавляем корень проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.lab4_logic import Lab4AIProcessor

# ПУТИ
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    processor = Lab4AIProcessor()
    
    text = (
        "This is a demonstration of neural speech synthesis using SpeechBrain. "
        "We are using Tacotron 2 as the acoustic model and HiFi-GAN as the vocoder. "
        "This represents the third generation of Text-to-Speech systems."
    )
    
    output_path = os.path.join(OUTPUT_DIR, "tts_output.wav")
    
    print("--- [Задание 1] Neural TTS (SpeechBrain) ---")
    processor.tts_with_reference(text, output_path=output_path)
    print(f"Готово! Результат: {output_path}")

if __name__ == "__main__":
    main()
