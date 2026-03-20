import numpy as np
from core.config import Lab1Config, InstrumentParams, Lab2Config, FIRParams, IIRParams, Lab3Config

# Данные для Лабораторной работы №1
LAB1_DATA = {
    10: {
        'x': InstrumentParams(name='Виолончель', amplitudes=[1.0, 0.6, 0.4, 0.2], f0=110, harmonics=[1, 2, 3, 4], phi=0),
        'y': InstrumentParams(name='Контрабас', amplitudes=[1.0, 0.7, 0.5], f0=55, harmonics=[1, 2, 3], phi=0)
    },
    # Можно добавить остальные варианты в таком же стиле...
}

# Данные для Лабораторной работы №2
LAB2_DATA = {
    10: Lab2Config(
        M_ma=79,
        ma_recursive=True,
        fir=FIRParams(type='bandpass', window='blackman', f_range=[80, 300], M=151),
        iir=IIRParams(type='bandpass', f0=200, bw=60)
    ),
}

# Данные для Лабораторной работы №3
LAB3_DATA = {
    10: Lab3Config(
        representation="Mel-spectrogram",
        features=["MFCC", "Spectral Rolloff"],
        metrics=["SNR", "PESQ"],
        model="DeepFilterNet2",
        snr_range=[8, 24],
        snr_step=2
    )
}

def get_lab1_config(variant: int = 10) -> Lab1Config:
    data = LAB1_DATA.get(variant, LAB1_DATA[10])
    return Lab1Config(x=data['x'], y=data['y'])

def get_lab2_config(variant: int = 10) -> Lab2Config:
    return LAB2_DATA.get(variant, LAB2_DATA[10])

def get_lab3_config(variant: int = 10) -> Lab3Config:
    return LAB3_DATA.get(variant, LAB3_DATA[10])
