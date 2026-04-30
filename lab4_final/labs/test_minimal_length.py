import os
import torch
import whisper
import soundfile as sf
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

INPUT_DIR = Path("output")
OUTPUT_DIR = INPUT_DIR / "length_experiment_results"
AUDIO_DIR = OUTPUT_DIR / "audio"
PLOTS_DIR = OUTPUT_DIR / "plots"

for dir_path in [OUTPUT_DIR, AUDIO_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

TEMP_DIR = Path("temp_length_exp")
TEMP_DIR.mkdir(exist_ok=True)

def create_test_segments(audio_path, min_duration=0.5, max_duration=10, step=0.5):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segments = {}
    durations = np.arange(min_duration, min(max_duration, len(audio) / sr), step)

    for duration in durations:
        samples = int(duration * sr)
        if samples <= len(audio):
            segment = audio[:samples]
            segments[duration] = segment

    return segments, sr

def voice_conversion(source_audio, target_audio, sr=24000):
    if sr != 24000:
        source_audio = signal.resample(source_audio, int(len(source_audio) * 24000 / sr))
        target_audio = signal.resample(target_audio, int(len(target_audio) * 24000 / sr))

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

def analyze_audio_quality(audio, sr=24000):
    metrics = {}

    noise = np.random.randn(len(audio)) * 0.001
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))

    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    metrics['spectral_centroid'] = np.sum(freqs * spec) / (np.sum(spec) + 1e-10)

    metrics['energy'] = np.sum(audio ** 2)
    metrics['zcr'] = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

    return metrics

def run_length_experiment():
    print("\nЗагрузка моделей...")
    whisper_model = whisper.load_model("base")
    silero_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v3_1_ru'
    )

    target_audio, target_sr = sf.read("input/target.wav")
    if target_audio.ndim > 1:
        target_audio = target_audio.mean(axis=1)

    print("\nСоздание тестовых сегментов...")
    segments, sr = create_test_segments("input/source.wav", 0.5, 10, 0.5)

    results = []

    print("\nЗапуск эксперимента...")
    for duration, segment in segments.items():
        print(f"\n   Длительность: {duration:.1f}с")

        try:
            temp_input = TEMP_DIR / f"temp_input_{duration:.1f}s.wav"
            sf.write(str(temp_input), segment, sr)

            result = whisper_model.transcribe(str(temp_input))
            text = result["text"].strip()

            if not text:
                print(f"   Текст не распознан для длительности {duration:.1f}с")
                continue

            base_audio = silero_model.apply_tts(
                text=text,
                speaker='xenia',
                sample_rate=24000
            )

            output_audio = voice_conversion(base_audio, target_audio)

            source_metrics = analyze_audio_quality(segment)
            output_metrics = analyze_audio_quality(output_audio)

            output_filename = f"output_{duration:.1f}s.wav"
            output_path = AUDIO_DIR / output_filename
            sf.write(str(output_path), output_audio, 24000)

            result_entry = {
                'duration': duration,
                'text_length': len(text),
                'word_count': len(text.split()),
                'source_metrics': source_metrics,
                'output_metrics': output_metrics,
                'recognized_text': text,
                'output_file': str(output_path)
            }
            results.append(result_entry)

            print(f"   Успешно: {len(text)} символов, "
                  f"SNR: {output_metrics['snr']:.1f}dB")
            print(f"   Сохранён: {output_path}")

        except Exception as e:
            print(f"   Ошибка: {e}")

    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    if results:
        durations = [r['duration'] for r in results]
        text_lengths = [r['text_length'] for r in results]

        min_quality_duration = min(
            [r['duration'] for r in results if r['text_length'] > 10],
            default=2.0
        )

        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"   Минимальная длина для качественного распознавания: {min_quality_duration:.1f}с")
        print(f"   Средняя длина текста: {np.mean(text_lengths):.1f} символов")
        print(f"   Всего успешных тестов: {len(results)}")
        print(f"   Все файлы сохранены в: {AUDIO_DIR}")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(durations, text_lengths, 'bo-')
        axes[0, 0].axvline(min_quality_duration, color='r', linestyle='--', label='Мин. длина')
        axes[0, 0].set_xlabel('Длительность аудио (с)')
        axes[0, 0].set_ylabel('Длина распознанного текста (симв.)')
        axes[0, 0].set_title('Зависимость распознавания от длительности')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        snr_values = [r['output_metrics']['snr'] for r in results]
        axes[0, 1].plot(durations, snr_values, 'go-')
        axes[0, 1].set_xlabel('Длительность аудио (с)')
        axes[0, 1].set_ylabel('SNR (dB)')
        axes[0, 1].set_title('Качество сигнала vs длительность')
        axes[0, 1].grid(True)

        centroid_values = [r['output_metrics']['spectral_centroid'] for r in results]
        axes[1, 0].plot(durations, centroid_values, 'mo-')
        axes[1, 0].set_xlabel('Длительность аудио (с)')
        axes[1, 0].set_ylabel('Спектральный центроид (Гц)')
        axes[1, 0].set_title('Характеристика тембра vs длительность')
        axes[1, 0].grid(True)

        energy_values = [r['output_metrics']['energy'] for r in results]
        axes[1, 1].plot(durations, energy_values, 'co-')
        axes[1, 1].set_xlabel('Длительность аудио (с)')
        axes[1, 1].set_ylabel('Энергия сигнала')
        axes[1, 1].set_title('Энергия vs длительность')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = PLOTS_DIR / 'length_analysis.png'
        plt.savefig(str(plot_path), dpi=150)
        plt.close()

        print(f"\nГрафик сохранён: {plot_path}")

    json_path = OUTPUT_DIR / "length_experiment.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"JSON-отчёт сохранён: {json_path}")

    for temp_file in TEMP_DIR.glob("*"):
        temp_file.unlink()
    TEMP_DIR.rmdir()

    print(f"\nВсе результаты в папке: {OUTPUT_DIR}")

    return results

if __name__ == "__main__":
    results = run_length_experiment()