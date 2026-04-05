import matplotlib.pyplot as plt

class UIColors:
    # Ультимативная контрастная темная тема
    BG_DARK = '#0A0A0A'
    BG_PANEL = '#141414'
    AXIS_BG = '#000000'
    
    TEXT_MAIN = '#FFFFFF'
    TEXT_DIM = '#B0B0B0'
    TEXT_ACCENT = '#00FFFF'
    
    GRID = '#222222'
    AXIS_SPINE = '#FFFFFF'
    
    BTN_RUN = '#00FF00'
    BTN_PLAY = '#007FFF'
    RADIO_ACTIVE = '#00FFFF'

    LAB1 = {'x': '#00FFFF', 'y': '#FF00FF', 'rec': '#00FF00', 'err': '#FF0000'}
    LAB2 = {'clean': '#00FFFF', 'noisy': '#FF8000', 'ma': '#00FF00', 'fir': '#FFFF00', 'iir': '#FF00FF'}
    LAB3 = {'mel': 'magma', 'mfcc': '#00FFFF', 'metrics': '#00FF00'}

    # Алиасы для совместимости с View (AttributeError Fix)
    SIG_X = LAB1['x']; SIG_Y = LAB1['y']; SIG_REC = LAB1['rec']; SIG_ERR = LAB1['err']
    SIG_CLEAN = LAB2['clean']; SIG_NOISY = LAB2['noisy']; SIG_MA = LAB2['ma']; SIG_FIR = LAB2['fir']; SIG_IIR = LAB2['iir']

    @staticmethod
    def apply_style(plt_obj=None):
        p = plt_obj if plt_obj is not None else plt
        p.rcParams.update({
            'axes.facecolor': UIColors.AXIS_BG,
            'axes.edgecolor': UIColors.AXIS_SPINE,
            'axes.labelcolor': UIColors.TEXT_MAIN,
            'axes.titlecolor': UIColors.TEXT_ACCENT,
            'xtick.color': UIColors.TEXT_MAIN,
            'ytick.color': UIColors.TEXT_MAIN,
            'grid.color': UIColors.GRID,
            'figure.facecolor': UIColors.BG_DARK,
            'text.color': UIColors.TEXT_MAIN,
            'font.family': 'sans-serif',
            'font.weight': 'bold',
            'savefig.facecolor': UIColors.BG_DARK
        })
        if hasattr(p, 'gcf'):
            p.gcf().patch.set_facecolor(UIColors.BG_DARK)

    @staticmethod
    def setup_axis(ax, title, x_label="", y_label=""):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15, color=UIColors.TEXT_ACCENT)
        ax.set_xlabel(x_label, fontsize=9, color=UIColors.TEXT_MAIN, fontweight='bold', labelpad=8)
        ax.set_ylabel(y_label, fontsize=9, color=UIColors.TEXT_MAIN, fontweight='bold', labelpad=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color(UIColors.AXIS_SPINE)
        ax.tick_params(axis='both', colors=UIColors.TEXT_MAIN, labelsize=8)
