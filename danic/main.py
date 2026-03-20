from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import numpy as np
import filters
from config import AppConfig
from core import (
    add_distortions,
    amplitude_spectrum,
    generate_signal,
    print_signal_info,
    to_db,
)
from io_utils import ensure_output_dir, save_figure, save_wav
from plots import (
    plot_filter_freqresp,
    plot_filtered_time,
    plot_phase_spectra,
    plot_spectra,
    plot_time_domain,
)


# ═══════════════════════════════════════════════════════════════════════════
# Консольный анализ АЧХ
# ═══════════════════════════════════════════════════════════════════════════

def _print_freqresp_report(w_ma: np.ndarray, H_ma: np.ndarray,
                            w_fir: np.ndarray, H_fir: np.ndarray,
                            w_iir: np.ndarray, H_iir: np.ndarray,
                            cfg: AppConfig) -> None:
    """Выводит ключевые значения АЧХ на частотах помехи и среза."""
    fc = cfg.filters
    dc = cfg.distortion

    print("\n── Анализ АЧХ (ключевые значения): ──")

    # Ослабление на частоте тональной помехи
    idx_tonal_ma  = np.argmin(np.abs(w_ma  - dc.tonal_freq))
    idx_tonal_fir = np.argmin(np.abs(w_fir - dc.tonal_freq))
    idx_tonal_iir = np.argmin(np.abs(w_iir - dc.tonal_freq))
    print(f"  MA  @ {dc.tonal_freq} Гц : {to_db(np.abs(H_ma [idx_tonal_ma ])):.1f} дБ")
    print(f"  КИХ @ {dc.tonal_freq} Гц : {to_db(np.abs(H_fir[idx_tonal_fir])):.1f} дБ  ← подавление помехи")
    print(f"  БИХ @ {dc.tonal_freq} Гц : {to_db(np.abs(H_iir[idx_tonal_iir])):.1f} дБ")

    # Ослабление на частоте среза БИХ
    idx_fc_ma  = np.argmin(np.abs(w_ma  - fc.f_cutoff_iir))
    idx_fc_fir = np.argmin(np.abs(w_fir - fc.f_cutoff_iir))
    idx_fc_iir = np.argmin(np.abs(w_iir - fc.f_cutoff_iir))
    print(f"\n  MA  @ fc={fc.f_cutoff_iir} Гц : {to_db(np.abs(H_ma [idx_fc_ma ])):.1f} дБ")
    print(f"  КИХ @ fc={fc.f_cutoff_iir} Гц : {to_db(np.abs(H_fir[idx_fc_fir])):.1f} дБ")
    print(f"  БИХ @ fc={fc.f_cutoff_iir} Гц : {to_db(np.abs(H_iir[idx_fc_iir])):.1f} дБ  ← должно быть ≈ −3 дБ")


# ═══════════════════════════════════════════════════════════════════════════
#  Главный пайплайн
# ═══════════════════════════════════════════════════════════════════════════

def run(cfg: AppConfig) -> None:
    """
    Запускает полный пайплайн лабораторной работы №2.

    Параметры:
        cfg : AppConfig — единственный источник всех параметров
    """
    print("=" * 60)
    print("  ЛАБ. РАБОТА №2 — ПРОЕКТИРОВАНИЕ ЦИФРОВЫХ ФИЛЬТРОВ")
    print("=" * 60)

    # ── Подготовка директории вывода ───────────────────────────────
    out_dir = ensure_output_dir(cfg.io)

    # ── Генерация сигналов ─────────────────────────────────────────
    print("\n[1/4] Генерация сигналов...")
    t, x_clean = generate_signal(cfg.signal)
    x_noisy    = add_distortions(x_clean, cfg.signal, cfg.distortion)
    print_signal_info(cfg)

    # ── Проектирование фильтров ────────────────────────────────────
    print("\n[2/4] Проектирование фильтров...")

    filters.print_ma_info(cfg.filters)

    # КИХ (Вариант 12: НЧ)
    h_fir = filters.design_fir_blackman_lf(cfg.filters, cfg.signal)
    print(f"\n  2.2 КИХ НЧ: M={cfg.filters.m_fir}, fc={cfg.filters.f_cutoff_fir} Гц, Blackman")
    filters.print_fir_coefficients(h_fir, n=10)

    # БИХ (Вариант 12: ВЧ)
    B_iir, A_iir, alpha = filters.design_iir_one_poly_hf(cfg.filters, cfg.signal)
    filters.print_iir_params(B_iir, A_iir, alpha, cfg.filters)

    # ── Применение фильтров ────────────────────────────────────────
    print("\n[3/4] Применение фильтров...")
    y_ma  = filters.apply_ma(x_noisy, cfg.filters)
    y_fir = filters.apply_fir(x_noisy, cfg.filters, cfg.signal)
    y_iir = filters.apply_iir(x_noisy, cfg.filters, cfg.signal)
    print("  Готово: MA (shifted), FIR (shifted), IIR")

    # ── АЧХ фильтров ──────────────────────────────────────────────
    w_ma,  H_ma  = filters.freqresp_moving_average(cfg.filters, cfg.signal)
    w_fir, H_fir = filters.freqresp_fir(h_fir, cfg.signal)
    w_iir, H_iir = filters.freqresp_iir(B_iir, A_iir, cfg.signal)

    _print_freqresp_report(w_ma, H_ma, w_fir, H_fir, w_iir, H_iir, cfg)

    # ── Визуализация и сохранение графиков ─────────────────────────
    print("\n[4/4] Построение графиков...")

    fig1 = plot_time_domain(t, x_clean, x_noisy, cfg)
    save_figure(fig1, "plot1_time_domain.png", cfg.io)
    plt.close(fig1)

    fig2 = plot_filter_freqresp(w_ma, H_ma, w_fir, H_fir, w_iir, H_iir, cfg)
    save_figure(fig2, "plot2_freq_response.png", cfg.io)
    plt.close(fig2)

    fig3 = plot_spectra(x_noisy, y_ma, y_fir, y_iir, cfg)
    save_figure(fig3, "plot3_spectra.png", cfg.io)
    plt.close(fig3)

    fig4 = plot_filtered_time(t, x_clean, x_noisy, y_ma, y_fir, y_iir, cfg)
    save_figure(fig4, "plot4_time_all.png", cfg.io)
    plt.close(fig4)

    fig5 = plot_phase_spectra(x_clean, x_noisy, y_fir, y_iir, cfg)
    save_figure(fig5, "plot5_phase_spectra.png", cfg.io)
    plt.close(fig5)

    # ── Экспорт WAV ────────────────────────────────────────────────
    print("\n[WAV] Сохранение аудиофайлов...")
    fs = cfg.signal.fs
    save_wav("01_clean_signal.wav",  x_clean, fs, cfg.io)
    save_wav("02_noisy_signal.wav",    x_noisy, fs, cfg.io)
    save_wav("03_filtered_MA.wav",     y_ma,    fs, cfg.io)
    save_wav("04_filtered_FIR.wav", y_fir,   fs, cfg.io)
    save_wav("05_filtered_IIR.wav", y_iir,   fs, cfg.io)

    noise = x_noisy - x_clean
    save_wav("06_noise.wav", noise, fs, cfg.io)

    print("\n" + "=" * 60)
    print("  РАБОТА ЗАВЕРШЕНА. Результаты сохранены в:")
    print(f"  {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run(AppConfig())
