import os
import sys
import time
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Добавляем корень проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.lab4_logic import Lab4AIProcessor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_experiment")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    processor = Lab4AIProcessor()
    
    source_wav = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source_audio_lab3", "voice_1_mono_miner.wav")
    target_wav = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source_audio_lab3", "voice_2_stereo_avsetaki.wav")
    
    # Загружаем оригинал
    audio, sr = librosa.load(source_wav, sr=16000)
    
    lengths = [1.0, 2.0, 3.0, 5.0, 10.0] # секунды
    results = []
    
    print("--- [Эксперимент] Минимальная длина для VC ---")
    
    for length in lengths:
        tmp_wav = os.path.join(OUTPUT_DIR, f"temp_{length}s.wav")
        out_wav = os.path.join(OUTPUT_DIR, f"vc_{length}s.wav")
        
        # Обрезаем аудио
        chunk = audio[:int(length * sr)]
        sf.write(tmp_wav, chunk, sr)
        
        start_time = time.time()
        try:
            processor.voice_conversion(tmp_wav, target_wav, out_wav)
            elapsed = time.time() - start_time
            results.append((length, elapsed, True))
            print(f"Длина {length}с: OK ({elapsed:.2f} сек)")
        except Exception as e:
            print(f"Длина {length}с: ОШИБКА ({e})")
            results.append((length, 0, False))
            
    # Визуализация
    plt.figure(figsize=(10, 5))
    valid_lengths = [r[0] for r in results if r[2]]
    valid_times = [r[1] for r in results if r[2]]
    
    plt.plot(valid_lengths, valid_times, marker='o', linestyle='-', color='b')
    plt.title("Зависимость времени обработки от длины входного файла (VC)")
    plt.xlabel("Длина (сек)")
    plt.ylabel("Время обработки (сек)")
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_plot.png"))
    print(f"График сохранен в {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'length_plot.png')}")

if __name__ == "__main__":
    main()
