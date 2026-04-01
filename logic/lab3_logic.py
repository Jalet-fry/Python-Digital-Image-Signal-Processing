import numpy as np
import librosa
import torch
from core.signals.features import (
    my_mel_spectrogram, get_mfcc_full, get_spectral_rolloff,
    get_spectral_centroid, get_spectral_bandwidth, get_zero_crossing_rate,
    calc_snr_metric, calc_si_sdr, calc_pesq_manual
)
from core.signals.noise import generate_white_noise, add_noise_snr

class SpeechProcessor:
    """Бэкенд для Лабораторной №3: Обработка речи и шума."""
    def __init__(self, df_model=None, df_state=None, device='cpu'):
        self.df_model = df_model
        self.df_state = df_state
        self.device = device

    def process_file(self, file_path, snr_db, use_df=True):
        y, sr = librosa.load(file_path, sr=16000)
        noise = generate_white_noise(len(y))
        noisy, _ = add_noise_snr(y, noise, snr_db)
        
        enhanced = self._enhance(noisy, sr, use_df)
        
        min_len = min(len(y), len(noisy), len(enhanced))
        y, noisy, enhanced = y[:min_len], noisy[:min_len], enhanced[:min_len]

        return {
            'clean': y, 'noisy': noisy, 'enhanced': enhanced, 'sr': sr,
            'features': self._extract_features(y, sr),
            'metrics': self._calc_metrics(y, noisy, enhanced, sr)
        }

    def _enhance(self, noisy, sr, use_df):
        if use_df and self.df_model:
            # Тут логика DeepFilterNet (как в предыдущем шаге)
            pass 
        # Простой фильтр для примера
        return noisy * 0.9 

    def _extract_features(self, y, sr):
        return {
            'mel': my_mel_spectrogram(y, sr),
            'mfcc': get_mfcc_full(y, sr),
            'centroid': get_spectral_centroid(y, sr)
        }

    def _calc_metrics(self, clean, noisy, enhanced, sr):
        return {
            'snr_in': calc_snr_metric(clean, noisy),
            'snr_out': calc_snr_metric(clean, enhanced),
            'si_sdr': calc_si_sdr(clean, enhanced),
            'pesq': calc_pesq_manual(clean, enhanced, sr)
        }
