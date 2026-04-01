import matplotlib.pyplot as plt

class UIColors:
    # Палитра - Сделаем фон чуть мягче (не чисто черный)
    BG_DARK = '#14161C'       # Очень темно-серый, комфортный для глаз
    BG_PANEL = '#1E212B'      # Фон для виджетов
    AXIS_BG = '#1A1C24'       # Фон графиков
    
    TEXT_MAIN = '#E0E6ED'     # Почти белый
    TEXT_DIM = '#94A3B8'      # Спокойный серый
    TEXT_ACCENT = '#00D2FF'   # Яркий голубой акцент
    
    GRID = '#2D333F'          # Цвет сетки
    AXIS_SPINE = '#3F4759'    # Цвет рамок
    
    BTN_RUN = '#10B981'       # Emerald
    BTN_PLAY = '#3B82F6'      # Blue
    RADIO_ACTIVE = '#10B981'
    RADIO_BG = '#1E212B'      # Добавлено для совместимости

    LAB1 = {'x': '#818CF8', 'y': '#34D399', 'rec': '#F87171', 'err': '#FB923C'}
    LAB2 = {'clean': '#34D399', 'noisy': '#FB923C', 'ma': '#60A5FA', 'fir': '#A78BFA', 'iir': '#F472B6'}
    LAB3 = {'mel': 'magma', 'mfcc': '#818CF8', 'metrics': '#10B981'}

    # Алиасы для обратной совместимости с Лаб 2 и Лаб 3
    SIG_X = LAB1['x']
    SIG_Y = LAB1['y']
    SIG_REC = LAB1['rec']
    SIG_ERR = LAB1['err']
    
    SIG_CLEAN = LAB2['clean']
    SIG_NOISY = LAB2['noisy']
    SIG_MA = LAB2['ma']
    SIG_FIR = LAB2['fir']
    SIG_IIR = LAB2['iir']

    @staticmethod
    def apply_style(plt_obj=None):
        """Применяет глобальные настройки стиля Matplotlib."""
        p = plt_obj if plt_obj is not None else plt
        p.rcParams.update({
            'axes.facecolor': UIColors.AXIS_BG,
            'axes.edgecolor': UIColors.AXIS_SPINE,
            'axes.labelcolor': UIColors.TEXT_MAIN,
            'axes.titlecolor': UIColors.TEXT_MAIN,
            'xtick.color': UIColors.TEXT_DIM,
            'ytick.color': UIColors.TEXT_DIM,
            'grid.color': UIColors.GRID,
            'figure.facecolor': UIColors.BG_DARK,
            'text.color': UIColors.TEXT_MAIN,
            'font.family': 'sans-serif'
        })

    # Алиас для совместимости
    @staticmethod
    def apply_dark_theme(plt_obj=None):
        UIColors.apply_style(plt_obj)

    @staticmethod
    def setup_axis(ax, title, x_label="", y_label=""):
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel(x_label, fontsize=8, color=UIColors.TEXT_DIM)
        ax.set_ylabel(y_label, fontsize=8, color=UIColors.TEXT_DIM)
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color(UIColors.AXIS_SPINE)
