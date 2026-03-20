

from __future__ import annotations

import os

import numpy as np
from matplotlib.figure import Figure
from scipy.io import wavfile

from config import IOConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Инициализация директории
# ═══════════════════════════════════════════════════════════════════════════

def ensure_output_dir(cfg: IOConfig) -> str:
    """
    Создаёт директорию результатов, если она не существует.

    Возвращает нормализованный абсолютный путь к директории.
    """
    out_dir = cfg.resolved_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════
#  Сохранение WAV
# ═══════════════════════════════════════════════════════════════════════════

def save_wav(filename: str, data: np.ndarray, fs: int, cfg: IOConfig) -> str:
    """
    Сохраняет массив float в WAV-файл (16-бит, нормализованный).

    Нормализация по максимальной амплитуде предотвращает клиппинг.

    Параметры:
        filename : имя файла (без пути, например '01_clean.wav')
        data     : аудиоданные в формате float
        fs       : частота дискретизации, Гц
        cfg      : IOConfig

    Возвращает полный путь к сохранённому файлу.
    """
    out_dir = ensure_output_dir(cfg)
    peak = np.max(np.abs(data))
    data_norm = data / peak if peak > 0 else data
    data_int16 = np.int16(data_norm * 32767)

    path = os.path.join(out_dir, filename)
    wavfile.write(path, fs, data_int16)
    print(f"Сохранён WAV: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Сохранение графиков
# ═══════════════════════════════════════════════════════════════════════════

def save_figure(fig: Figure, filename: str, cfg: IOConfig,
                dpi: int = 150) -> str:
    """
    Сохраняет matplotlib Figure в PNG-файл.

    Параметры:
        fig      : объект Figure для сохранения
        filename : имя файла (например 'plot1_time_domain.png')
        cfg      : IOConfig
        dpi      : разрешение, точек на дюйм

    Возвращает полный путь к сохранённому файлу.
    """
    out_dir = ensure_output_dir(cfg)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Сохранён: {path}")
    return path
