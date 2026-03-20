from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignalConfig:

    fs: int = 44100               # Частота дискретизации, Гц
    duration: float = 2.0         # Длительность сигнала, с
    f0: int = 330                # Основная частота Арфы, Гц
    harmonics: tuple[int, ...] = (1, 2, 4, 8)
    amplitudes: tuple[float, ...] = (1.0, 0.3, 0.1, 0.05)

    @property
    def n_samples(self) -> int:
        return int(self.fs * self.duration)


@dataclass(frozen=True)
class DistortionConfig:

    noise_std: float = 0.01       # СКО белого гауссова шума
    tonal_freq: float = 1500.0    # Частота тональной помехи, Гц (попадает в полосу КИХ)
    tonal_amp: float = 0.8        # Амплитуда тональной помехи
    hf_freq: float = 50.0       # Частота ВЧ-шума, Гц
    hf_amp: float = 0.8          # Амплитуда ВЧ-шума
    seed: int = 42                # Зерно генератора для воспроизводимости


@dataclass(frozen=True)
class FilterConfig:

    # ── Однородный фильтр (Рекурсивный)
    m_ma: int = 14                # Порядок М

    # ── КИХ-фильтр (НЧ, окно Блэкмана)
    m_fir: int = 101              # Порядок M
    f_cutoff_fir: float = 500.0   # Частота среза, Гц

    # ── БИХ-фильтр (однополюсный ВЧ) ──────────────────
    f_cutoff_iir: float = 600.0         # Частота среза, Гц


@dataclass(frozen=True)
class IOConfig:
    # Директория сохранения результатов относительно расположения модуля
    out_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "results", "lab2_filters"
        )
    )

    def resolved_out_dir(self) -> str:
        return os.path.normpath(self.out_dir)


@dataclass(frozen=True)
class AppConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    distortion: DistortionConfig = field(default_factory=DistortionConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    io: IOConfig = field(default_factory=IOConfig)
