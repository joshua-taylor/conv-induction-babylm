"""
paper_style.py — one consistent, clean, modern look for every figure.
Import and call apply() at the top of any plotting script:
    import paper_style as ps; ps.apply()
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette (consistent roles across all figures) ----
INK     = "#1f2430"     # text / dots
MUTED   = "#8a8f98"     # axis lines
GRID    = "#ecedf0"     # gridlines
PRIMARY = "#2a6f97"     # "induction" / main series
SECOND  = "#e07a5f"     # "attention" / control / comparison
ACCENT  = "#3d8361"     # third series
EDGE    = "#b5341f"     # induction edges (arcs), warm red
SERIES  = [PRIMARY, SECOND, ACCENT, "#8367c7", MUTED]


def apply():
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 200,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "600",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10.5, "axes.labelcolor": INK,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })


def grid_y_only(ax):
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)
