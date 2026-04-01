import os
from pydub import AudioSegment
try:
    from static_ffmpeg import add_paths
    # Добавляет пути к ffmpeg и ffprobe в PATH автоматически
    add_paths()
    HAS_STATIC_FFMPEG = True
except ImportError:
    HAS_STATIC_FFMPEG = False

def convert_to_wav_mono(file_path, target_sr=16000):
    """
    Конвертирует аудиофайл (mp3, m4a, wav) в формат WAV Mono с нужным Sample Rate.
    """
    try:
        # Пытаемся загрузить файл
        ext = os.path.splitext(file_path)[1].lower()
        
        # pydub сам поймет формат, если ffmpeg/ffprobe в PATH
        audio = AudioSegment.from_file(file_path)
            
        # Превращаем в моно
        audio = audio.set_channels(1)
        # Устанавливаем Sample Rate
        audio = audio.set_frame_rate(target_sr)
        
        # Генерируем новый путь
        base_name = os.path.splitext(file_path)[0]
        new_path = base_name + "_converted.wav"
        
        audio.export(new_path, format="wav")
        return new_path
    except Exception as e:
        print(f"\nОшибка конвертации {file_path}: {e}")
        if not HAS_STATIC_FFMPEG:
            print("СОВЕТ: Попробуйте выполнить 'pip install static-ffmpeg'")
        return None
