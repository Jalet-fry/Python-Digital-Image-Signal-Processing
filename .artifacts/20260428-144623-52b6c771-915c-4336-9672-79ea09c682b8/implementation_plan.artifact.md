# Implementation Plan - Lab 4 "Lite Mode" (No Downloads)

To bypass the slow downloads of heavy neural models (SpeechT5, WavLM) while still fulfilling the lab requirements for BSUIR, we will implement a "Lite Mode". This mode uses classical DSP algorithms (Griffin-Lim) and simpler feature extraction as a fallback if models are not present locally.

## User Review Required

> [!IMPORTANT]
> The "Lite" version will use **Griffin-Lim** instead of **HiFi-GAN** if the heavy models are not downloaded. This will result in slightly more "robotic" audio, but the code will run immediately without the 1GB download.

## Proposed Changes

### Core DSP Logic

#### [voice_processor.py](file:///C:/Univer/BSUIR_LABS/6_term/ЦОСиИ/PythonDSP/core/dsp/voice_processor.py)

- Add a flag `self.lite_mode` that activates if `local_files_only=True` fails and the user wants to avoid downloads.
- **Fallback Feature Extraction:** Use Mel-spectrogram similarity instead of WavLM embeddings if WavLM is not available.
- **Fallback Synthesis:** Use `librosa.griffinlim` instead of HiFi-GAN.
- **Mock TTS:** Use `pyttsx3` (system voice) as a fallback for the neural TTS task.
- Update `load_wavlm` and `load_tts` to check for local files first and offer a choice or automatically switch to Lite mode.

### GUI Adjustments

#### [lab4_ai.py](file:///C:/Univer/BSUIR_LABS/6_term/ЦОСиИ/PythonDSP/labs/lab4_ai.py)

- Update reports to reflect when "Lite Mode" is active.
- Ensure the app doesn't hang on `from transformers import ...` if they aren't fully initialized (use lazy imports or try-except).

## Verification Plan

### Automated Tests
- Run `make lab4` and verify it starts without triggering a download.
- Check if `results/lab4_vc.wav` is generated using the fallback method.

### Manual Verification
- Verify that clicking "CONVERT VOICE" produces audio even if models are missing.
- Verify that "STABILITY (2.3)" and "LOGS (2.1)" still show plots (using mock logs and simplified metrics).
