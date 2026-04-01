import numpy as np
import librosa
import torch
import os
from core.dsp.features import (
    my_mel_spectrogram, get_mfcc_full, get_frame_features,
    calc_snr_metric, calc_si_sdr, calc_pesq_manual
)
from core.dsp.generator import add_noise_snr, generate_white_noise

class SpeechProcessor:
    def __init__(self, df_model=None, df_state=None, device='cpu'):
        self.df_model = df_model
        self.df_state = df_state
        self.device = device
        self.current_res = None

    def process_file(self, file_path, snr_db, use_df=True):
        # 1. Загрузка
        y, sr = librosa.load(file_path, sr=16000)
        
        # 2. Зашумление
        noise = generate_white_noise(len(y))
        noisy, _ = add_noise_snr(y, noise, snr_db)
        
        # 3. Очистка
        enhanced = self._enhance(noisy, sr, use_df)
        
        # Выравнивание длин
        min_len = min(len(y), len(noisy), len(enhanced))
        y, noisy, enhanced = y[:min_len], noisy[:min_len], enhanced[:min_len]

        # 4. Расчет признаков для UI
        # Сравнение спектрограмм (Пункт 2 методички)
        mel_my = my_mel_spectrogram(y, sr)
        mel_lib = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128))
        
        # MFCC и временные признаки
        mfcc = get_mfcc_full(enhanced, sr)
        centroids, zcr = get_frame_features(enhanced, sr)

        self.current_res = {
            'clean': y, 'noisy': noisy, 'enhanced': enhanced, 'sr': sr,
            'features': {
                'mel_my': mel_my,
                'mel_lib': mel_lib,
                'mfcc': mfcc,
                'centroid': centroids,
                'zcr': zcr
            },
            'metrics': {
                'snr_in': calc_snr_metric(y, noisy),
                'snr_out': calc_snr_metric(y, enhanced),
                'si_sdr': calc_si_sdr(y, enhanced),
                'pesq': calc_pesq_manual(y, enhanced, sr)
            }
        }
        return self.current_res

    def _enhance(self, noisy, sr, use_df):
        if use_df and self.df_model:
            try:
                from df.enhance import enhance
                # DeepFilterNet работает на 48кГц
                noisy_48 = librosa.resample(noisy, orig_sr=sr, target_sr=48000)
                noisy_tensor = torch.from_numpy(noisy_48).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    enhanced_tensor = enhance(self.df_model, self.df_state, noisy_tensor)
                enhanced_48 = enhanced_tensor.cpu().numpy().squeeze()
                return librosa.resample(enhanced_48, orig_sr=48000, target_sr=sr)
            except Exception as e:
                print(f"DeepFilter Error: {e}")
        
        # Fallback: Спектральное вычитание (если DeepFilter недоступен)
        stft = librosa.stft(noisy)
        mag, phase = librosa.magphase(stft)
        noise_mag = np.mean(mag[:, :10], axis=1, keepdims=True) # Оценка шума по паузе
        mag_clean = np.maximum(mag - 1.5 * noise_mag, 0)
        return librosa.istft(mag_clean * phase)
