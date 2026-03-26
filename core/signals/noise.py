import numpy as np

def generate_white_noise(length, level=1.0):
    """Генерация белого шума."""
    return level * np.random.normal(0, 1, length)

def add_noise_snr(signal, noise, snr_db):
    """
    Наложение шума на сигнал с заданным соотношением сигнал/шум (SNR).
    SNR = 10 * log10(P_signal / P_noise)
    """
    # 1. Считаем мощность сигнала
    p_signal = np.mean(signal**2)
    
    # 2. Считаем мощность шума
    p_noise_current = np.mean(noise**2)
    
    # 3. Вычисляем требуемую мощность шума для целевого SNR
    # P_noise_target = P_signal / (10^(SNR/10))
    p_noise_target = p_signal / (10**(snr_db / 10))
    
    # 4. Масштабируем шум
    scaling_factor = np.sqrt(p_noise_target / (p_noise_current + 1e-12))
    scaled_noise = noise * scaling_factor
    
    return signal + scaled_noise, scaled_noise
