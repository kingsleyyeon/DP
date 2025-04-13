
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_plot_style(use_tex=False):
    """
    Set global matplotlib style.
    """
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6
    })

def create_figure(figsize=(7, 5.5)):
    """
    Create a matplotlib figure with the standard size.
    """
    plt.figure(figsize=figsize)

def save_figure(filename, dpi=300):
    """
    Save the current figure with standard layout and resolution.
    """
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
