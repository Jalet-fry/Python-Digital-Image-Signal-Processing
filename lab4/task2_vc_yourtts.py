import os
import torch
import whisper
import soundfile as sf
import numpy as np
import warnings

warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SOURCE_WAV = "source.wav"
TARGET_WAV = "target.wav"
OUTPUT_WAV = "output.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n[1] Распознавание текста...")
whisper_model = whisper.load_model("base", device=DEVICE)
result = whisper_model.transcribe(SOURCE_WAV)
text = result["text"].strip()
print(f"    Текст: \"{text}\"")

print("\n[2] Синтез речи (Silero TTS)...")
silero_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v3_1_ru'
)
silero_model.to(DEVICE)

base_audio = silero_model.apply_tts(
    text=text,
    speaker='xenia',
    sample_rate=24000
)
print(f"    Сгенерировано {len(base_audio)} сэмплов")

print("\n[3] Анализ целевого голоса...")
target_audio, sr = sf.read(TARGET_WAV)
if target_audio.ndim > 1:
    target_audio = target_audio.mean(axis=1)

if sr != 24000:
    from scipy.signal import resample

    target_audio = resample(target_audio, int(len(target_audio) * 24000 / sr))

def extract_formant_envelope(audio, sr=24000, n_fft=2048, hop_length=512):
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=n_fft))
    from scipy.ndimage import gaussian_filter1d
    envelope = gaussian_filter1d(spec, sigma=8)
    return envelope / (envelope.max() + 1e-8)


def apply_voice_filter(source_audio, target_audio, sr=24000):
    n_fft = 2048
    hop_length = n_fft // 4

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

sf.write(OUTPUT_WAV, output_audio, 24000)
