"""Generate training dynamics visualizations for encoder MLM pretraining.

Generates realistic training curves showing:
1. MLM loss decay over steps
2. Learning rate schedule
3. Combined loss + LR visualization

Run:
    python generate_encoder_training_dynamics.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


COLORS = {
    "ink": "#1F2A37",
    "subtle": "#6B7280",
    "grid": "#D9E1EA",
    "panel": "#F7FAFD",
    "blue": "#2F6DB2",
    "blue_light": "#78A9E0",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "teal": "#76B7B2",
    "red": "#E15759",
    "gray": "#8A94A6",
    "gold": "#C58F1F",
}


def set_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 1.0,
            "axes.facecolor": COLORS["panel"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        }
    )


def make_axes(size=(6.4, 4.0)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["panel"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def generate_synthetic_training_data(num_steps=20000, initial_lr=1e-4, warmup_steps=2000):
    """Generate realistic MLM pretraining curves.
    
    Simulates:
    - Warmup phase: 0-2k steps with linear LR increase
    - Main phase: 2k-20k steps with cosine LR decay
    - Loss: exponential decay with noise
    """
    steps = np.arange(0, num_steps + 1, 100)  # Sample every 100 steps
    
    # Learning rate schedule: linear warmup + cosine decay
    lrs = []
    for step in steps:
        if step < warmup_steps:
            lr = initial_lr * (step / warmup_steps)


#             "savefig.bbox": "tight",
#             "savefig.dpi": 150,
#         }
#     )
# 
# 
# def make_axes(size=(6.4, 4.0)):
#     fig, ax = plt.subplots(figsize=size, constrained_layout=True)
#     fig.patch.set_facecolor("white")
#     ax.set_facecolor(COLORS["panel"])
#     for spine in ("top", "right"):
#         ax.spines[spine].set_visible(False)
#     return fig, ax
# 
# 
# def generate_synthetic_training_data(num_steps=20000, initial_lr=1e-4, warmup_steps=2000):
#     """Generate realistic MLM pretraining curves.
#     
#     Simulates:
#     - Warmup phase: 0-2k steps with linear LR increase
#     - Main phase: 2k-20k steps with cosine LR decay
#     - Loss: exponential decay with noise
#     """
#     steps = np.arange(0, num_steps + 1, 100)  # Sample every 100 steps
#     
#     # Learning rate schedule: linear warmup + cosine decay
#     lrs = []
#     for step in steps:
#         if step < warmup_steps:
#             lr = initial_lr * (step / warmup_steps)
