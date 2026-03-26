import numpy as np
from scipy.fftpack import dct
from core.signals.fourier import fft
from core.utils.aspects import log_dsp_action
import sys

# Увеличим лимит рекурсии на всякий случай
sys.setrecursionlimit(5000)

# Глобальный флаг для Лабы 3
USE_CUSTOM_FFT = True 

def get_fft_mag(x):
    """
    Вычисляет магнитудный спектр. 
    ОГРАНИЧЕНИЕ: Рекурсивный FFT в Python очень медленный на больших массивах.
    Для стабильности берем фрагмент сигнала (окно) в 2048 отсчетов.
    """
    if USE_CUSTOM_FFT:
        # Если сигнал слишком длинный, берем только центральную часть 2048 отсчетов
        # Это предотвращает зависание и соответствует принципу оконного анализа
        N_target = 2048
        if len(x) > N_target:
            mid = len(x) // 2
            x_segment = x[mid : mid + N_target]
        else:
            x_segment = x
            
        N = len(x_segment)
        if N == 0: return np.array([])
        
        # Zero Padding до степени двойки
        N_padded = 1 << (N - 1).bit_length()
        if N_padded > N:
            x_padded = np.pad(x_segment, (0, N_padded - N))
        else:
            x_padded = x_segment
            
        res = fft(x_padded)
        return np.abs(res[:len(res)//2 + 1])
    else:
        # Библиотечный FFT работает быстро с любой длиной
        return np.abs(np.fft.rfft(x))

@log_dsp_action
def get_chroma(x, sr):
    import librosa
    # Для Chroma используем библиотеку, как разрешено в задании
    chroma = librosa.feature.chroma_stft(y=x, sr=sr, n_fft=2048)
    return np.mean(chroma, axis=1)

@log_dsp_action
def get_spectral_rolloff(x, sr, roll_percent=0.85):
    mag_spec = get_fft_mag(x)
    if len(mag_spec) == 0: return 0.0
    
    n_fft_actual = (len(mag_spec) - 1) * 2
    freqs = np.fft.rfftfreq(n_fft_actual, 1/sr)
    
    power_spec = mag_spec**2
    total_energy = np.sum(power_spec)
    if total_energy == 0: return 0.0
    
    threshold = roll_percent * total_energy
    cumulative_energy = np.cumsum(power_spec)
    
    try:
        rolloff_idx = np.where(cumulative_energy >= threshold)[0][0]
        return freqs[rolloff_idx]
    except IndexError:
        return 0.0

@log_dsp_action
def my_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Ручная реализация мел-спектрограммы. 
    Здесь фрейминг (оконное преобразование) реализован в цикле.
    """
    # Ограничим длину сигнала для спектрограммы, если он слишком длинный (более 10 сек)
    # чтобы не ждать вечность при отрисовке ручного алгоритма
    max_samples = 10 * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    n_frames = 1 + (len(y) - n_fft) // hop_length
    window = np.hanning(n_fft)
    
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
    # Для ускорения отрисовки используем rfft из numpy внутри цикла
    for i in range(n_frames):
        frame = y[i*hop_length : i*hop_length + n_fft]
        if len(frame) < n_fft: break
        mag_spec = np.abs(np.fft.rfft(frame * window, n=n_fft))
        power_spec = mag_spec**2 / n_fft
        mel_energies.append(np.dot(filters, power_spec))
    
    return np.log10(np.array(mel_energies).T + 1e-10)

@log_dsp_action
def get_mfcc(x, sr, n_mfcc=13, n_mels=40):
    mel_spec = my_mel_spectrogram(x, sr, n_mels=n_mels)
    log_mel_energy = np.mean(mel_spec, axis=1)
    return dct(log_mel_energy, type=2, axis=-1, norm='ortho')[:n_mfcc]

get_mfcc_full = get_mfcc

@log_dsp_action
def get_spectral_centroid(x, sr):
    mag_spec = get_fft_mag(x)
    if len(mag_spec) == 0: return 0.0
    n_fft_actual = (len(mag_spec) - 1) * 2
    freqs = np.fft.rfftfreq(n_fft_actual, 1/sr)
    return np.sum(freqs * mag_spec**2) / (np.sum(mag_spec**2) + 1e-12)

@log_dsp_action
def get_spectral_bandwidth(x, sr):
    centroid = get_spectral_centroid(x, sr)
    mag_spec = get_fft_mag(x)
    if len(mag_spec) == 0: return 0.0
    n_fft_actual = (len(mag_spec) - 1) * 2
    freqs = np.fft.rfftfreq(n_fft_actual, 1/sr)
    return np.sqrt(np.sum(((freqs - centroid)**2) * mag_spec**2) / (np.sum(mag_spec**2) + 1e-12))

@log_dsp_action
def get_zero_crossing_rate(x):
    return np.sum(np.abs(np.diff(np.sign(x)))) / (2 * len(x))

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

def calc_sdr(c, p): return calc_snr_metric(c, p)
