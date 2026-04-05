import numpy as np
from scipy.fftpack import dct
import librosa
from core.dsp.fourier import fft
from core.utils.aspects import log_dsp_action

@log_dsp_action
def get_fft_mag_safe(x, n_target=2048):
    """
    Безопасное вычисление магнитудного спектра (из DSP_Old_Version).
    Берет центральное окно сигнала, чтобы рекурсивный FFT не переполнил стек.
    """
    if len(x) > n_target:
        mid = len(x) // 2
        x_segment = x[mid : mid + n_target]
    else:
        x_segment = x
        
    N = len(x_segment)
    if N == 0: return np.array([])
    
    # Zero Padding до степени 2
    N_padded = 1 << (N - 1).bit_length()
    x_padded = np.pad(x_segment, (0, N_padded - N))
    
    res = fft(x_padded)
    return np.abs(res[:len(res)//2 + 1])

@log_dsp_action
def my_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Ручная реализация мел-спектрограммы с фильтрами."""
    window = np.hanning(n_fft)
    n_frames = 1 + (len(y) - n_fft) // hop_length
    
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)
    
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr/2), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        left, center, right = bin_pts[m], bin_pts[m+1], bin_pts[m+2]
        for k in range(left, center):
            if k < filters.shape[1]: filters[m, k] = (k - left) / (center - left + 1e-12)
        for k in range(center, right):
            if k < filters.shape[1]: filters[m, k] = (right - k) / (right - center + 1e-12)

    mel_energies = []
    for i in range(n_frames):
        frame = y[i*hop_length : i*hop_length + n_fft]
        if len(frame) < n_fft: break
        mag_spec = np.abs(np.fft.rfft(frame * window, n=n_fft))
        power_spec = mag_spec**2 / n_fft
        mel_energies.append(np.dot(filters, power_spec))
    
    return np.log10(np.array(mel_energies).T + 1e-10)

@log_dsp_action
def get_mfcc_full(y, sr, n_mfcc=13):
    mel_spec = my_mel_spectrogram(y, sr)
    log_mel_energy = np.mean(mel_spec, axis=1)
    return dct(log_mel_energy, type=2, axis=-1, norm='ortho')[:n_mfcc]

@log_dsp_action
def get_extended_features(y, sr):
    """Все характеристики из старой версии + либроса."""
    mag_spec = get_fft_mag_safe(y)
    n_fft_actual = (len(mag_spec) - 1) * 2
    freqs = np.fft.rfftfreq(n_fft_actual, 1/sr)
    
    # 1. Centroid
    centroid = np.sum(freqs * mag_spec**2) / (np.sum(mag_spec**2) + 1e-12)
    # 2. Rolloff
    power_spec = mag_spec**2
    total_energy = np.sum(power_spec)
    threshold = 0.85 * total_energy
    cumulative_energy = np.cumsum(power_spec)
    rolloff = freqs[np.where(cumulative_energy >= threshold)[0][0]] if total_energy > 0 else 0
    # 3. ZCR
    zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))
    # 4. Chroma (librosa)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048))
    # 5. Bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * mag_spec**2) / (np.sum(mag_spec**2) + 1e-12))

    return {
        'centroid': centroid, 'rolloff': rolloff, 'zcr': zcr, 
        'chroma': chroma, 'bandwidth': bandwidth
    }

@log_dsp_action
def calc_snr_metric(clean, processed):
    min_len = min(len(clean), len(processed))
    s, sh = clean[:min_len], processed[:min_len]
    return 10 * np.log10(np.sum(s**2) / (np.sum((s - sh)**2) + 1e-12))

@log_dsp_action
def calc_si_sdr(reference, estimated):
    r, e = reference.flatten(), estimated.flatten()
    min_len = min(len(r), len(e))
    r, e = r[:min_len], e[:min_len]
    alpha = np.dot(e, r) / (np.sum(r**2) + 1e-12)
    target = alpha * r
    res = e - target
    return 10 * np.log10(np.sum(target**2) / (np.sum(res**2) + 1e-12))

@log_dsp_action
def calc_pesq_manual(clean, processed, sr=16000):
    min_len = min(len(clean), len(processed))
    s, h = clean[:min_len], processed[:min_len]
    S = np.abs(np.fft.rfft(s, n=512))
    H = np.abs(np.fft.rfft(h, n=512))
    L_s, L_h = (S + 1e-6)**0.23, (H + 1e-6)**0.23
    disturbance = np.mean(np.abs(L_s - L_h))
    return np.clip(4.5 - (disturbance * 5), 1.0, 4.5)
