import os
import time

import psutil
import torch
import whisper
import soundfile as sf
import numpy as np
from GPUtil import GPUtil
from scipy.ndimage import gaussian_filter1d
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

OUTPUT_DIR = Path("output")
RESULT_DIR = OUTPUT_DIR / "resource_evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.get_memory_usage()
        self.start_time = None

    def get_memory_usage(self):
        return self.process.memory_info().rss / 1024 / 1024

    def get_gpu_memory(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
            return 0
        except:
            return 0

    def start_measure(self):
        self.start_time = time.time()
        self.start_memory = self.get_memory_usage()

    def end_measure(self):
        elapsed = time.time() - self.start_time
        memory_used = self.get_memory_usage() - self.start_memory
        return {
            'time_seconds': elapsed,
            'ram_mb': memory_used,
            'gpu_mb': self.get_gpu_memory()
        }

def test_small_models():
    results = {}

    for model_name in ['tiny', 'base', 'small']:
        print(f"\nТест Whisper {model_name}...")
        monitor = ResourceMonitor()
        monitor.start_measure()

        model = whisper.load_model(model_name, device='cpu')
        result = model.transcribe("source.wav")

        stats = monitor.end_measure()
        stats['model_size'] = model_name
        results[f'whisper_{model_name}'] = stats

        del model
        torch.cuda.empty_cache()

    return results

def test_cpu_vs_gpu():
    if not torch.cuda.is_available():
        print("GPU недоступен, пропускаем тест")
        return {}

    results = {}

    for device in ['cpu', 'cuda']:
        print(f"\nТест на {device.upper()}...")
        monitor = ResourceMonitor()
        monitor.start_measure()

        whisper_model = whisper.load_model("base", device=device)
        silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v3_1_ru'
        )

        if device == 'cuda':
            silero_model = silero_model.cuda()

        result = whisper_model.transcribe("input/source.wav")
        text = result["text"].strip()

        base_audio = silero_model.apply_tts(
            text=text,
            speaker='xenia',
            sample_rate=24000
        )

        stats = monitor.end_measure()
        stats['device'] = device
        results[f'device_{device}'] = stats

        del whisper_model, silero_model
        torch.cuda.empty_cache()

    return results

def test_batch_processing():
    results = {}

    test_files = [
        ("source.wav", "original"),
    ]

    for file_path, label in test_files:
        if not os.path.exists(file_path):
            continue

        print(f"\nТест файла: {label}")
        monitor = ResourceMonitor()
        monitor.start_measure()

        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(file_path)
        text = result["text"].strip()

        silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v3_1_ru'
        )

        base_audio = silero_model.apply_tts(text=text, speaker='xenia', sample_rate=24000)

        target_audio, sr = sf.read("target.wav")
        if target_audio.ndim > 1:
            target_audio = target_audio.mean(axis=1)

        def apply_voice_filter(source_audio, target_audio, sr=24000):
            n_fft = 2048
            hop_length = n_fft // 4

            def extract_formant_envelope(audio, sr=24000, n_fft=2048, hop_length=512):
                spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=n_fft))
                envelope = gaussian_filter1d(spec, sigma=8)
                return envelope / (envelope.max() + 1e-8)

            target_envelope = extract_formant_envelope(
                target_audio[:min(len(target_audio), sr * 3)],
                sr, n_fft, hop_length
            )

            output = np.zeros(len(source_audio))
            window = np.hanning(n_fft)

            for i in range(0, len(source_audio) - n_fft, hop_length):
                chunk = source_audio[i:i + n_fft] * window
                chunk_spec = np.fft.rfft(chunk, n=n_fft)
                chunk_mag = np.abs(chunk_spec)
                chunk_phase = np.angle(chunk_spec)
                filtered_mag = chunk_mag * target_envelope * 2
                filtered_chunk = np.fft.irfft(
                    filtered_mag * np.exp(1j * chunk_phase),
                    n=n_fft
                )
                output[i:i + n_fft] += filtered_chunk * window

            output = output / (np.max(np.abs(output)) + 1e-8)
            return output

        output_audio = apply_voice_filter(base_audio, target_audio)

        audio_path = RESULT_DIR / f"output_{label}.wav"
        sf.write(str(audio_path), output_audio, 24000)

        stats = monitor.end_measure()
        stats['file'] = label
        stats['audio_duration'] = len(base_audio) / 24000
        stats['output_file'] = str(audio_path)
        results[label] = stats

        del whisper_model, silero_model
        torch.cuda.empty_cache()

    return results

def generate_report(cpu_gpu_results, batch_results):
    report_lines = []

    report_lines.append("\n1. ПРОИЗВОДИТЕЛЬНОСТЬ ПРИ ОГРАНИЧЕНИИ ПАМЯТИ:")

    if cpu_gpu_results:
        report_lines.append("\n   CPU vs GPU:")
        for key, stats in cpu_gpu_results.items():
            report_lines.append(f"   • {stats['device'].upper()}: {stats['time_seconds']:.2f}с, "
                  f"RAM: {stats['ram_mb']:.1f}MB, GPU: {stats['gpu_mb']:.1f}MB")

    if batch_results:
        report_lines.append("\n2. ОБРАБОТКА ФАЙЛОВ:")
        for key, stats in batch_results.items():
            report_lines.append(f"   • {key}: {stats['time_seconds']:.2f}с, "
                  f"RAM: {stats['ram_mb']:.1f}MB, "
                  f"Длительность аудио: {stats['audio_duration']:.1f}с")

        report_lines.append("\n3. РЕКОМЕНДАЦИИ:")
        report_lines.append("   • Для CPU: используйте Whisper 'tiny' вместо 'base'")
        report_lines.append("   • Для GPU: VRAM >= 2GB для комфортной работы")
        report_lines.append("   • Кэширование моделей: первый запуск самый долгий")

    report = "\n".join(report_lines)
    print(report)

    report_path = RESULT_DIR / "resource_report.txt"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)

    return report

print("Тест 1: Сравнение размеров моделей Whisper")
small_results = test_small_models()

print("\nТест 2: Сравнение CPU vs GPU")
cpu_gpu_results = test_cpu_vs_gpu()

print("\nТест 3: Измерение времени обработки")
batch_results = test_batch_processing()

generate_report(cpu_gpu_results, batch_results)

full_results = {
    'whisper_models': small_results,
    'cpu_gpu': cpu_gpu_results,
    'batch': batch_results
}

json_path = RESULT_DIR / "resource_evaluation.json"
with open(str(json_path), "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2, ensure_ascii=False)

print(f"\nВсе результаты сохранены в: {RESULT_DIR}")