import numpy as np
from core.config import Lab1Config, InstrumentParams, Lab2Config, FIRParams, IIRParams, Lab3Config

# ДАННЫЕ ИЗ ТАБЛИЦЫ МЕТОДИЧКИ (Ровно как в DSP_Old_Version)
LAB1_DATA = {
    10: {
        'x': InstrumentParams(
            name='Виолончель', 
            amplitudes=[1.0, 0.6, 0.4, 0.2], 
            f0=110, 
            harmonics=[1, 2, 3, 4], 
            phi=0
        ),
        'y': InstrumentParams(
            name='Контрабас', 
            amplitudes=[1.0, 0.7, 0.5], 
            f0=55, 
            harmonics=[1, 2, 3], 
            phi=0
        )
    }
}

def get_lab1_config(variant: int = 10) -> Lab1Config:
    data = LAB1_DATA.get(variant, LAB1_DATA[10])
    return Lab1Config(
        variant=variant,
        x=data['x'], 
        y=data['y'],
        N=1024,
        sr=10000,
        sr_audio=44100,
        duration_audio=3.0 # Возвращаем оригинальную длительность
    )

def get_lab2_config(variant: int = 10) -> Lab2Config:
    return Lab2Config(
        M_ma=79,
        ma_recursive=True,
        fir=FIRParams(type='bandpass', window='blackman', f_range=[80, 300], M=151),
        iir=IIRParams(type='bandpass', f0=200, bw=60)
    )

def get_lab3_config(variant: int = 10) -> Lab3Config:
    return Lab3Config(
        representation="Mel-spectrogram",
        features=["MFCC", "Spectral Rolloff"],
        metrics=["SNR", "PESQ"],
        model="DeepFilterNet3",
        snr_range=[8, 24],
        snr_step=2
    )
