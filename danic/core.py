"""
=============================================================================
core.py — Ядро: генерация сигналов, добавление помех, спектральный анализ
=============================================================================
Слой «Core / Math»: чистые функции без побочных эффектов (Pure Functions).
Не импортирует ничего из visualization, io_utils или filters.
=============================================================================
"""

from __future__ import annotations

import numpy as np

from config import AppConfig, DistortionConfig, SignalConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Генерация сигналов
# ═══════════════════════════════════════════════════════════════════════════

def generate_signal(cfg: SignalConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Модель: x(t) = Σ Aₖ · sin(2π · k·f0 · t),  k ∈ harmonics
    Затем сигнал нормализуется по максимуму, чтобы пиковое значение = 1.

    Параметры:
        cfg : SignalConfig — конфигурация сигнала

    Возвращает:
        t : временная ось, с
        x : нормализованный сигнал амплитуды
    """
    N = int(np.round(cfg.fs * cfg.duration))
    t = np.linspace(0, cfg.duration, N, endpoint=False)
    x = np.zeros(N)

    for Ak, k in zip(cfg.amplitudes, cfg.harmonics):
        x += Ak * np.sin(2 * np.pi * k * cfg.f0 * t)

    x /= np.max(np.abs(x))        
    return t, x


def add_distortions(x: np.ndarray, cfg_signal: SignalConfig,
                    cfg_dist: DistortionConfig) -> np.ndarray:
    """
    Добавляет к сигналу три вида искажений:
      1. Белый гауссов шум       — имитирует тепловой/электрический шум АЦП.
      2. Тональная помеха — узкополосная синусоида в полосе подавления КИХ.
      3. ВЧ-синусоида   — проверяет работу НЧ-фильтров.

    Параметры:
        x          : чистый сигнал
        cfg_signal : SignalConfig — для получения fs
        cfg_dist   : DistortionConfig — параметры помех

    Возвращает x_noisy.
    """
    rng = np.random.default_rng(cfg_dist.seed)
    N = len(x)
    t = np.arange(N) / cfg_signal.fs

    white_noise   = rng.normal(0, cfg_dist.noise_std, N)
    tonal_interf  = cfg_dist.tonal_amp * np.sin(2 * np.pi * cfg_dist.tonal_freq * t)
    hf_interf     = cfg_dist.hf_amp    * np.sin(2 * np.pi * cfg_dist.hf_freq    * t)

    return x + white_noise + tonal_interf + hf_interf


# ═══════════════════════════════════════════════════════════════════════════
#  Спектральный анализ
# ═══════════════════════════════════════════════════════════════════════════

def amplitude_spectrum(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает односторонний амплитудный спектр (до fs/2).
    """
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    A = 2 * np.abs(X) / N
    A[0] /= 2   # DC-компонента не удваивается
    return freqs, A

def phase_spectrum(signal: np.ndarray, fs:int):
    N = len(signal)
    fft_data = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    phases = np.unwrap(np.angle(fft_data))
    return freqs, phases


def to_db(H_mag: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """
    Переводит линейную АЧХ |H| в логарифмический масштаб (дБ):
        H_dB = 20·log10(|H|)
    Значения ниже порога floor_db ограничиваются снизу.

    Параметры:
        H_mag    : линейные значения АЧХ (или амплитуды)
        floor_db : нижний порог ограничения в дБ

    Возвращает массив значений в дБ.
    """
    with np.errstate(divide='ignore'):
        H_db = 20 * np.log10(np.maximum(np.abs(H_mag), 10 ** (floor_db / 20)))
    return H_db


# ═══════════════════════════════════════════════════════════════════════════
#  Консольный вывод диагностики (не визуализация, не файловый I/O)
# ═══════════════════════════════════════════════════════════════════════════

def print_signal_info(cfg: AppConfig) -> None:
    """Выводит сводку параметров сигнала и помех в консоль."""
    sc = cfg.signal
    dc = cfg.distortion
    print(f"  Длина сигнала : {sc.n_samples} отсчётов ({sc.duration} с, fs={sc.fs} Гц)")
    print(f"  Гармоники     : {[k * sc.f0 for k in sc.harmonics]} Гц")
    print(f"  Помехи        : белый шум σ={dc.noise_std}, "
          f"тональная {dc.tonal_freq} Гц, ВЧ {dc.hf_freq} Гц")
