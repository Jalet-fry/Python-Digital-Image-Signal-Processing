import numpy as np
from scipy.fftpack import dct
import librosa
from core.utils.aspects import log_dsp_action

@log_dsp_action
def my_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Ручная реализация мел-спектрограммы (Пункт 2 методички)."""
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)

    window = np.hanning(n_fft)
    n_frames = 1 + (len(y) - n_fft) // hop_length
    
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bin_pts[m], bin_pts[m+1]):
            filters[m, k] = (k - bin_pts[m]) / (bin_pts[m+1] - bin_pts[m])
        for k in range(bin_pts[m+1], bin_pts[m+2]):
            filters[m, k] = (bin_pts[m+2] - k) / (bin_pts[m+2] - bin_pts[m+1])

    mel_spec = []
    for i in range(n_frames):
        frame = y[i*hop_length : i*hop_length + n_fft]
        if len(frame) < n_fft: break
        mag_spec = np.abs(np.fft.rfft(frame * window))
        power_spec = (mag_spec**2) / n_fft
        mel_frame = np.dot(filters, power_spec)
        mel_spec.append(mel_frame)
    
    return np.array(mel_spec).T

@log_dsp_action
def get_mfcc_full(y, sr, n_mfcc=13):
    mel_spec = my_mel_spectrogram(y, sr)
    log_mel = np.log10(mel_spec + 1e-10)
    mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    return np.mean(mfcc, axis=1) # Вектор средних для баров

@log_dsp_action
def get_spectral_features(y, sr):
    """Все признаки из оригинального lab3_speech.py."""
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 1. Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    # 2. Bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    # 3. Rolloff
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    # 4. ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    # 5. Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    
    return {
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rolloff': rolloff,
        'zcr': zcr,
        'chroma': chroma
    }

@log_dsp_action
def calc_snr_metric(clean, processed):
    min_len = min(len(clean), len(processed))
    s, s_hat = clean[:min_len], processed[:min_len]
    return 10 * np.log10(np.sum(s**2) / (np.sum((s - s_hat)**2) + 1e-12))

@log_dsp_action
def calc_si_sdr(reference, estimated):
    min_len = min(len(reference), len(estimated))
    s, s_hat = reference[:min_len], estimated[:min_len]
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + 1e-12)
    target = alpha * s
    noise = s_hat - target
    return 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-12))

@log_dsp_action
def calc_pesq_manual(clean, processed, sr=16000):
    # Упрощенная имитация
    min_len = min(len(clean), len(processed))
    s, h = clean[:min_len], processed[:min_len]
    S = np.abs(np.fft.rfft(s, n=512))
    H = np.abs(np.fft.rfft(h, n=512))
    L_s, L_h = (S + 1e-6)**0.23, (H + 1e-6)**0.23
    disturbance = np.mean(np.abs(L_s - L_h))
    return np.clip(4.5 - (disturbance * 5), 1.0, 4.5)
