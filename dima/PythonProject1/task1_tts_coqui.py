import os
from TTS.api import TTS

INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".txt"):
        txt_path = os.path.join(INPUT_DIR, filename)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        wav_filename = f"{os.path.splitext(filename)[0]}.wav"
        wav_path = os.path.join(OUTPUT_DIR, wav_filename)
        tts.tts_to_file(text=text, file_path=wav_path)
        print(f"Создан: {wav_path}")
