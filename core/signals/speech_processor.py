import numpy as np
import librosa
import os
import torch
from scipy.io import wavfile
from core.signals.features import (
    my_mel_spectrogram, get_mfcc_full, get_spectral_rolloff,
    get_spectral_centroid, get_spectral_bandwidth, get_zero_crossing_rate,
    calc_snr_metric, calc_si_sdr, calc_pesq_manual, get_chroma
)
from core.signals.noise import generate_white_noise, add_noise_snr

class SpeechProcessor:
    def __init__(self, df_model=None, df_state=None, device='cpu'):
        self.df_model = df_model
        self.df_state = df_state
        self.device = device
        self.results = {}

    def process(self, file_path, snr_db, use_df=True):
        # 1. Загрузка
        y, sr = librosa.load(file_path, sr=16000)
        
        # 2. Зашумление (Пункт 4 задания)
        noise = generate_white_noise(len(y))
        noisy, _ = add_noise_snr(y, noise, snr_db)
        
        # 3. Очистка (Пункт 6 задания - DeepFilterNet)
        enhanced = None
        if use_df and self.df_model:
            try:
                from df.enhance import enhance
                noisy_48 = librosa.resample(noisy, orig_sr=sr, target_sr=48000)
                noisy_tensor = torch.from_numpy(noisy_48).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    enhanced_tensor = enhance(self.df_model, self.df_state, noisy_tensor)
                enhanced_48 = enhanced_tensor.cpu().numpy().squeeze()
                enhanced = librosa.resample(enhanced_48, orig_sr=48000, target_sr=sr)
            except Exception as e:
                print(f"DeepFilter Error: {e}")

        if enhanced is None:
            # Фолбэк на простой спектральный вычитатель
            stft = librosa.stft(noisy)
            mag, phase = librosa.magphase(stft)
            noise_mag = np.mean(mag[:, :10], axis=1, keepdims=True)
            mag_clean = np.maximum(mag - 1.5 * noise_mag, 0)
            enhanced = librosa.istft(mag_clean * phase)

        # Выравнивание длины
        min_len = min(len(y), len(noisy), len(enhanced))
        y, noisy, enhanced = y[:min_len], noisy[:min_len], enhanced[:min_len]
        
        # 4. Расчет признаков (Пункт 3: сравнение ручных и либ)
        features = {
            'rolloff_my': get_spectral_rolloff(y, sr),
            'centroid_my': get_spectral_centroid(y, sr),
            'zcr_my': get_zero_crossing_rate(y),
            'mfcc_my': get_mfcc_full(y, sr),
            'chroma_lib': get_chroma(y, sr),
            'spec_my': my_mel_spectrogram(y, sr),
            'spec_lib': librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr)),
            'bandwidth_my': get_spectral_bandwidth(y, sr)
        }
        
        # 5. Метрики (Пункт 5)
        metrics = {
            'snr_in': calc_snr_metric(y, noisy),
            'snr_out': calc_snr_metric(y, enhanced),
            'si_sdr': calc_si_sdr(y, enhanced),
            'pesq_my': calc_pesq_manual(y, enhanced, sr)
        }
        
        self.results = {
            'clean': y, 'noisy': noisy, 'enhanced': enhanced,
            'sr': sr, 'features': features, 'metrics': metrics
        }
        return self.results

    def get_comparison_table_row(self, filename):
        m = self.results['metrics']
        return [filename, f"{m['snr_in']:.1f}", f"{m['snr_out']:.1f}", 
                f"{m['si_sdr']:.1f}", f"{m['pesq_my']:.2f}"]
