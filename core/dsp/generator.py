import numpy as np
from core.utils.aspects import log_dsp_action

@log_dsp_action
def generate_instrument_signal(amplitudes, f0, harmonics, phases, duration=0.02, sr=10000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for a, h in zip(amplitudes, harmonics):
        signal += a * np.sin(2 * np.pi * (f0 * h) * t + phases)
    return t, signal

def generate_white_noise(length, level=1.0):
    return level * np.random.normal(0, 1, length)

def add_noise_snr(signal, noise, snr_db):
    p_signal = np.mean(signal**2)
    p_noise_current = np.mean(noise**2)
    p_noise_target = p_signal / (10**(snr_db / 10))
    scaling_factor = np.sqrt(p_noise_target / (p_noise_current + 1e-12))
    scaled_noise = noise * scaling_factor
    return signal + scaled_noise, scaled_noise
