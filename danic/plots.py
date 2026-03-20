from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from config import AppConfig
from core import amplitude_spectrum, to_db


# ═══════════════════════════════════════════════════════════════════════════
#  Вспомогательная функция
# ═══════════════════════════════════════════════════════════════════════════

def _annotate_freq_axes(ax: plt.Axes,
                        f_max: float,
                        w: np.ndarray,
                        H: np.ndarray,
                        label: str,
                        color: str,
                        title: str,
                        vlines: list[tuple[float, str, str]] | None = None) -> None:
    """Рисует АЧХ в дБ на одном Axes."""
    mask = w <= f_max
    ax.plot(w[mask], to_db(H[mask]), color=color, linewidth=1.5, label=label)
    if vlines:
        for vf, vc, vl in vlines:
            ax.axvline(vf, color=vc, linestyle='--', linewidth=1, label=vl)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("|H(f)|, дБ")
    ax.set_ylim(-80, 5)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
#  График 1 — Временная область
# ═══════════════════════════════════════════════════════════════════════════

def plot_time_domain(t: np.ndarray,
                      x_clean: np.ndarray,
                      x_noisy: np.ndarray,
                      cfg: AppConfig,
                      window_ms: float = 30.0) -> Figure:
    sc = cfg.signal
    dc = cfg.distortion
    n_show = int(window_ms * 1e-3 * sc.fs)
    t_ms   = t[:n_show] * 1000

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle("График 1 — Временная область: чистый vs зашумлённый сигнал",
                  fontsize=13, fontweight='bold')

    axes[0].plot(t_ms, x_clean[:n_show], color='steelblue', linewidth=1)
    axes[0].set_ylabel("Амплитуда")
    axes[0].set_title(f"Чистый сигнал (f₀ = {sc.f0} Гц)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t_ms, x_noisy[:n_show], color='tomato', linewidth=0.8)
    axes[1].set_ylabel("Амплитуда")
    axes[1].set_xlabel("Время, мс")
    axes[1].set_title(f"Зашумлённый сигнал (шум σ={dc.noise_std}, помеха {dc.tonal_freq} Гц)")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  График 2 — АЧХ всех трёх фильтров (Исправлено под Вариант 12)
# ═══════════════════════════════════════════════════════════════════════════

def plot_filter_freqresp(w_ma: np.ndarray, H_ma: np.ndarray,
                          w_fir: np.ndarray, H_fir: np.ndarray,
                          w_iir: np.ndarray, H_iir: np.ndarray,
                          cfg: AppConfig,
                          f_max: float = 1200.0) -> Figure:
    fc = cfg.filters
    dc = cfg.distortion

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("График 2 — АЧХ проектируемых фильтров (дБ)",
                  fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {'MA': 'royalblue', 'FIR': 'darkorange', 'IIR': 'forestgreen'}

    # 1. Скользящее среднее
    ax1 = fig.add_subplot(gs[0, 0])
    _annotate_freq_axes(
        ax1, f_max, w_ma, H_ma,
        label=f"СС (M={fc.m_ma})", color=colors['MA'],
        title=f"Скользящее среднее (M={fc.m_ma})"
    )

    # 2. КИХ НЧ (Вариант 12)
    ax2 = fig.add_subplot(gs[0, 1])
    _annotate_freq_axes(
        ax2, f_max, w_fir, H_fir,
        label="КИХ НЧ (Blackman)", color=colors['FIR'],
        title=f"КИХ НЧ: fc={fc.f_cutoff_fir} Гц, M={fc.m_fir}",
        vlines=[(fc.f_cutoff_fir, 'red', f"fc={fc.f_cutoff_fir} Гц")]
    )

    # 3. БИХ ВЧ (Вариант 12)
    ax3 = fig.add_subplot(gs[1, 0])
    _annotate_freq_axes(
        ax3, f_max, w_iir, H_iir,
        label="БИХ ВЧ", color=colors['IIR'],
        title=f"БИХ ВЧ: fc={fc.f_cutoff_iir} Гц",
        vlines=[(fc.f_cutoff_iir, 'red', f"fc={fc.f_cutoff_iir} Гц")]
    )

    # 4. Совмещённый
    ax4 = fig.add_subplot(gs[1, 1])
    for w, H, label, color in [
        (w_ma,  H_ma,  f"СС (M={fc.m_ma})",        colors['MA']),
        (w_fir, H_fir, f"КИХ НЧ ({fc.f_cutoff_fir})", colors['FIR']),
        (w_iir, H_iir, f"БИХ ВЧ ({fc.f_cutoff_iir})", colors['IIR']),
    ]:
        mask = w <= f_max
        ax4.plot(w[mask], to_db(H[mask]), color=color, linewidth=1.5, label=label)

    ax4.axvline(fc.f_cutoff_fir, color='orange', linestyle=':', linewidth=1)
    ax4.axvline(fc.f_cutoff_iir, color='green', linestyle=':', linewidth=1)
    ax4.set_title("Совмещённая АЧХ")
    ax4.set_xlabel("Частота, Гц")
    ax4.set_ylabel("|H(f)|, дБ")
    ax4.set_ylim(-80, 5)
    ax4.grid(True, alpha=0.4)
    ax4.legend(fontsize=8)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  График 3 и 4 — Спектры (Исправлено)
# ═══════════════════════════════════════════════════════════════════════════

def plot_spectra(x_noisy: np.ndarray,
                  y_ma: np.ndarray,
                  y_fir: np.ndarray,
                  y_iir: np.ndarray,
                  cfg: AppConfig,
                  f_max: float = 1500.0) -> Figure:
    sc = cfg.signal
    fc = cfg.filters
    dc = cfg.distortion

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("График 3 — Амплитудные спектры сигналов", fontsize=13, fontweight='bold')

    datasets = [
        (x_noisy, "Зашумлённый сигнал", 'tomato'),
        (y_ma,    f"После СС (M={fc.m_ma})", 'royalblue'),
        (y_fir,   f"После КИХ НЧ (fc={fc.f_cutoff_fir})", 'darkorange'),
        (y_iir,   f"После БИХ ВЧ (fc={fc.f_cutoff_iir})", 'forestgreen'),
    ]

    for ax, (sig, title, color) in zip(axes, datasets):
        freqs, A = amplitude_spectrum(sig, sc.fs)
        mask = freqs <= f_max
        ax.plot(freqs[mask], A[mask], color=color, linewidth=0.9)
        ax.axvline(dc.tonal_freq, color='purple', linestyle='--', alpha=0.6, label="Помеха")
        ax.set_ylabel("Ампл.")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Частота, Гц")
    plt.tight_layout()
    return fig

def plot_phase_spectra(x_clean:np.ndarray,
                      x_noisy: np.ndarray,
                      y_fir: np.ndarray,
                      y_iir: np.ndarray,
                      cfg: AppConfig,
                      f_max: float = 1500.0) -> Figure:
    from core import phase_spectrum
    sc = cfg.signal

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("График 4 — Фазовые спектры сигналов (рад)", fontsize=13, fontweight='bold')

    datasets = [
        (x_clean, "Чистый сигнал", 'red'),
        (x_noisy, "Зашумлённый сигнал", 'tomato'),
        (y_fir, "После КИХ НЧ (Линейная фаза)", 'darkorange'),
        (y_iir, "После БИХ ВЧ (Нелинейная фаза)", 'forestgreen'),
    ]

    for ax, (sig, title, color) in zip(axes, datasets):
        freqs, phases = phase_spectrum(sig, sc.fs)
        mask = freqs <= f_max
        ax.plot(freqs[mask], phases[mask], color=color, linewidth=0.8)
        ax.set_ylabel("Фаза, рад")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        # Устанавливаем лимиты для фазы от -pi до pi
#        ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)

    axes[-1].set_xlabel("Частота, Гц")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════
#  График 5 — Все сигналы во времени (Исправлено)
# ═══════════════════════════════════════════════════════════════════════════

def plot_filtered_time(t: np.ndarray,
                        x_clean: np.ndarray,
                        x_noisy: np.ndarray,
                        y_ma: np.ndarray,
                        y_fir: np.ndarray,
                        y_iir: np.ndarray,
                        cfg: AppConfig,
                        window_ms: float = 30.0) -> Figure:
    sc = cfg.signal
    fc = cfg.filters

    n_show = int(window_ms * 1e-3 * sc.fs)
    t_ms   = t[:n_show] * 1000

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    fig.suptitle("График 5 — Результаты фильтрации во временной области",
                  fontsize=13, fontweight='bold')

    rows = [
        (x_clean, "Чистый сигнал", 'steelblue'),
        (x_noisy, "Зашумлённый", 'tomato'),
        (y_ma,    f"СС (M={fc.m_ma})", 'royalblue'),
        (y_fir,   f"КИХ НЧ (fc={fc.f_cutoff_fir})", 'darkorange'),
        (y_iir,   f"БИХ ВЧ (fc={fc.f_cutoff_iir})", 'forestgreen'),
    ]

    for ax, (sig, lbl, col) in zip(axes, rows):
        ax.plot(t_ms, sig[:n_show], color=col, linewidth=0.9)
        ax.set_ylabel("Ампл.")
        ax.set_title(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Время, мс")
    plt.tight_layout()
    return fig