import numpy as np
from scipy.fftpack import dct
from core.signals.fourier import fft
from core.utils.aspects import log_dsp_action


@log_dsp_action
def my_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Реализация мел-спектрограммы по формулам (43-46) из методички.
    """
    # Функции перевода (формулы 43, 44)
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)

    # 1. Оконное преобразование и спектрограмма (Формула 42)
    n_frames = 1 + (len(y) - n_fft) // hop_length
    window = np.hanning(n_fft)
    
    # 2. Построение банка мел-фильтров (Формула 45)
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    
    # Перевод частот в индексы бинов ДПФ
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        for k in range(bin_pts[m], bin_pts[m+1]):
            filters[m, k] = (k - bin_pts[m]) / (bin_pts[m+1] - bin_pts[m])
        for k in range(bin_pts[m+1], bin_pts[m+2]):
            filters[m, k] = (bin_pts[m+2] - k) / (bin_pts[m+2] - bin_pts[m+1])

    # 3. Применение фильтров к кадрам
    mel_spec = []
    for i in range(n_frames):
        frame = y[i*hop_length : i*hop_length + n_fft]
        if len(frame) < n_fft: break
        
        # ДПФ и спектр мощности (Периодограмма)
        mag_spec = np.abs(np.fft.rfft(frame * window))
        power_spec = (mag_spec**2) / n_fft
        
        # Энергия в мел-полосах
        mel_frame = np.dot(filters, power_spec)
        mel_spec.append(mel_frame)
    
    # Логарифмирование (Формула 46)
    return np.log10(np.array(mel_spec).T + 1e-10)

# ==========================================================
# 2. ПРИЗНАКИ (Самостоятельно для вар. 10)
# ==========================================================

@log_dsp_action
def get_mfcc_full(y, sr, n_mfcc=13):
    """
    Вычисление MFCC: Мел-спектр -> Логарифм -> DCT.
    """
    mel_spec = my_mel_spectrogram(y, sr)
    # Среднее по времени для получения вектора признаков
    log_mel_energy = np.mean(mel_spec, axis=1)
    # Дискретное косинусное преобразование (Шаг 5 алгоритма MFCC)
    return dct(log_mel_energy, type=2, axis=-1, norm='ortho')[:n_mfcc]

@log_dsp_action
def get_spectral_rolloff(y, sr, roll_percent=0.85):
    """
    Спектральный спад: частота, ниже которой лежит roll_percent энергии.
    Алгоритм из методички (стр. 56).
    """
    mag_spec = np.abs(np.fft.rfft(y))
    power_spec = mag_spec**2
    total_energy = np.sum(power_spec)
    threshold = roll_percent * total_energy
    
    cumulative_energy = np.cumsum(power_spec)
    try:
        rolloff_idx = np.where(cumulative_energy >= threshold)[0][0]
    except IndexError:
        rolloff_idx = len(power_spec) - 1
    
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    return freqs[min(rolloff_idx, len(freqs)-1)]

# ==========================================================
# 3. МЕТРИКИ КАЧЕСТВА (Самостоятельно для вар. 10)
# ==========================================================

@log_dsp_action
def calc_snr_metric(clean, processed):
    """
    SNR по формуле (48): 10 * log10( ||s||^2 / ||s - s_hat||^2 )
    """
    min_len = min(len(clean), len(processed))
    s = clean[:min_len]
    s_hat = processed[:min_len]
    
    noise = s - s_hat
    p_signal = np.sum(s**2)
    p_noise = np.sum(noise**2)
    
    return 10 * np.log10(p_signal / (p_noise + 1e-12))

@log_dsp_action
def calc_sdr(reference, estimated):
    """
    Вычисляет стандартный SDR (Signal-to-Distortion Ratio).
    """
    reference = reference.flatten()
    estimated = estimated.flatten()
    min_len = min(len(reference), len(estimated))
    s = reference[:min_len]
    s_hat = estimated[:min_len]
    
    noise = s - s_hat
    return 10 * np.log10(np.sum(s**2) / (np.sum(noise**2) + 1e-12))

@log_dsp_action
def calc_si_sdr(reference, estimated):
    """
    Вычисляет SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).
    """
    reference = reference.flatten()
    estimated = estimated.flatten()
    min_len = min(len(reference), len(estimated))
    s = reference[:min_len]
    s_hat = estimated[:min_len]
    
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + 1e-12)
    target = alpha * s
    noise = s_hat - target
    return 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-12))

@log_dsp_action
def calc_pesq_manual(clean, processed, sr=16000):
    """
    Упрощенная перцептивная метрика (аналог PESQ/BSD).
    """
    min_len = min(len(clean), len(processed))
    s = clean[:min_len]
    h = processed[:min_len]
    
    # Акустическое преобразование (упрощенно через спектр)
    S = np.abs(np.fft.rfft(s, n=512))
    H = np.abs(np.fft.rfft(h, n=512))
    
    # Закон Цвикера для громкости
    L_s = (S + 1e-6)**0.23
    L_h = (H + 1e-6)**0.23
    
    disturbance = np.mean(np.abs(L_s - L_h))
    score = 4.5 - (disturbance * 10) 
    return np.clip(score, 1.0, 4.5)

# Дополнительные признаки

def get_spectral_centroid(x, sr):
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    return np.sum(freqs * mag) / (np.sum(mag) + 1e-12)

def get_spectral_bandwidth(x, sr):
    centroid = get_spectral_centroid(x, sr)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    return np.sqrt(np.sum(((freqs - centroid)**2) * mag) / (np.sum(mag) + 1e-12))

def get_zero_crossing_rate(x):
    return np.sum(np.abs(np.diff(np.sign(x)))) / (2 * len(x))

def get_chroma(x, sr):
    import librosa
    return np.mean(librosa.feature.chroma_stft(y=x, sr=sr), axis=1)

def calc_lsd(clean, processed):
    S1 = np.log10(np.abs(np.fft.rfft(clean))**2 + 1e-12)
    S2 = np.log10(np.abs(np.fft.rfft(processed))**2 + 1e-12)
    return np.sqrt(np.mean((S1 - S2)**2))
