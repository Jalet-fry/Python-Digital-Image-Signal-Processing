import os
import sys

# Добавляем корень проекта в путь
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.utils.converter import convert_to_wav_mono

SOURCE_DIR = os.path.join(BASE_DIR, "source_audio_lab3")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Папка {SOURCE_DIR} не найдена!")
        return

    # Поддерживаемые форматы для конвертации
    supported_ext = ('.mp3', '.m4a', '.wav', '.flac', '.mp4')
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(supported_ext)]
    
    print(f"--- АУДИО КОНВЕРТЕР (WAV 16kHz Mono) ---")
    print(f"Найдено подходящих файлов: {len(files)}")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for f in files:
        full_path = os.path.join(SOURCE_DIR, f)
        
        # 1. Пропускаем уже сконвертированные файлы
        if f.endswith("_converted.wav"):
            skipped_count += 1
            continue
            
        # 2. Проверяем, существует ли уже сконвертированная версия для этого файла
        target_name = os.path.splitext(f)[0] + "_converted.wav"
        if os.path.exists(os.path.join(SOURCE_DIR, target_name)):
            print(f"[-] Пропуск (уже есть): {f}")
            skipped_count += 1
            continue

        print(f"[*] Конвертация: {f} ...", end=" ", flush=True)
        new_file = convert_to_wav_mono(full_path)
        
        if new_file:
            print(f"УСПЕХ -> {os.path.basename(new_file)}")
            success_count += 1
        else:
            print("ОШИБКА!")
            fail_count += 1

    print(f"\n--- ИТОГО ---")
    print(f"Успешно: {success_count}")
    print(f"Пропущено: {skipped_count}")
    print(f"Ошибок: {fail_count}")

if __name__ == "__main__":
    main()
