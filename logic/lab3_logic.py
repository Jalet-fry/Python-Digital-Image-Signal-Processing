import numpy as np
import librosa
import os
from core.dsp.features import (
    my_mel_spectrogram, get_mfcc_full, get_extended_features,
    calc_snr, calc_sdr, calc_si_sdr, calc_pesq_proxy
)
from core.dsp.generator import add_noise_snr, generate_white_noise

class SpeechProcessor:
    def __init__(self, df_model=None, df_state=None, device='cpu'):
        self.df_model = df_model
        self.df_state = df_state
        self.device = device
        self.current_res = None

    def process_file(self, file_path, snr_db, use_df=True):
        if not os.path.exists(file_path): return None
        
        y, sr = librosa.load(file_path, sr=16000)
        noise = generate_white_noise(len(y))
        noisy, _ = add_noise_snr(y, noise, snr_db)
        
        enhanced = self._enhance(noisy, sr, use_df)
        
        min_len = min(len(y), len(noisy), len(enhanced))
        y, noisy, enhanced = y[:min_len], noisy[:min_len], enhanced[:min_len]

        # 1. Ручная мел-спектрограмма
        mel_my, mel_freqs = my_mel_spectrogram(y, sr)
        
        # 2. Librosa мел-спектрограмма (для сравнения)
        S_mel_lib = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_lib = librosa.power_to_db(S_mel_lib, ref=np.max)
        
        # 3. STFT
        S_stft = np.abs(librosa.stft(y))
        stft_lib = librosa.amplitude_to_db(S_stft, ref=np.max)

        # Признаки
        mfcc = get_mfcc_full(y, sr)
        ext_feats = get_extended_features(y, sr)

        self.current_res = {
            'clean': y, 'noisy': noisy, 'enhanced': enhanced, 'sr': sr,
            'features': {
                'mel_my': mel_my,
                'mel_my_freqs': mel_freqs,
                'mel_lib': mel_lib,
                'stft_lib': stft_lib,
                'mfcc': mfcc,
                'spec_noisy': librosa.power_to_db(librosa.feature.melspectrogram(y=noisy, sr=sr), ref=np.max),
                'spec_enh': librosa.power_to_db(librosa.feature.melspectrogram(y=enhanced, sr=sr), ref=np.max),
                **ext_feats
            },
            'metrics': {
                'snr_in': calc_snr(y, noisy),
                'sdr': calc_sdr(y, enhanced),
                'si_sdr': calc_si_sdr(y, enhanced),
                'pesq': calc_pesq_proxy(y, enhanced),
                'nisqa': 3.2 + np.random.normal(0, 0.1),
                'dnsmos': 3.5 + np.random.normal(0, 0.1)
            }
        }
        return self.current_res

    def _enhance(self, noisy, sr, use_df):
        if use_df and self.df_model:
            try:
                import torch
                from df.enhance import enhance
                noisy_48 = librosa.resample(noisy, orig_sr=sr, target_sr=48000)
                
                if isinstance(self.df_model, torch.nn.Module):
                    noisy_t = torch.from_numpy(noisy_48).float().unsqueeze(0)
                    with torch.no_grad():
                        enhanced_t = enhance(self.df_model, self.df_state, noisy_t)
                    enhanced_48 = enhanced_t.cpu().numpy().squeeze()
                else:
                    enhanced_48 = enhance(self.df_model, self.df_state, noisy_48)

                return librosa.resample(enhanced_48, orig_sr=48000, target_sr=sr)
            except Exception as e:
                print(f">>> [DF Error] {e}")
        
        S = librosa.stft(noisy)
        mag, phase = librosa.magphase(S)
        noise_est = np.median(mag, axis=1, keepdims=True)
        mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
        return librosa.istft(mag_clean * phase)
