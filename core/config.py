from dataclasses import dataclass, field
from typing import List, Union, Optional
import numpy as np

@dataclass
class InstrumentParams:
    name: str
    amplitudes: List[float]
    f0: float
    harmonics: List[int]
    phi: float

@dataclass
class Lab1Config:
    x: InstrumentParams
    y: InstrumentParams
    N: int = 1024
    sr: int = 8000
    sr_audio: int = 44100
    duration_audio: float = 3.0

@dataclass
class FIRParams:
    type: str
    window: str
    f_range: Union[float, List[float]]
    M: int

@dataclass
class IIRParams:
    type: str
    f0: Optional[float] = None
    bw: Optional[float] = None
    fc: Optional[float] = None

@dataclass
class Lab2Config:
    M_ma: int
    ma_recursive: bool
    fir: FIRParams
    iir: IIRParams

@dataclass
class Lab3Config:
    representation: str  # e.g., "Mel-spectrogram"
    features: List[str]  # e.g., ["MFCC", "Rolloff"]
    metrics: List[str]   # e.g., ["SNR", "PESQ"]
    model: str           # e.g., "DeepFilterNet2"
    snr_range: List[int] # [8, 24]
    snr_step: int        # 2
