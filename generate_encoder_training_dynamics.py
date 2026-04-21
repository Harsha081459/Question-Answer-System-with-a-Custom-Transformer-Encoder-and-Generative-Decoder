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
        else:
            progress = (step - warmup_steps) / (num_steps - warmup_steps)
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    lrs = np.array(lrs)
    
    # MLM loss: exponential decay with noise
    base_loss = 8.5 * np.exp(-0.0001 * steps) + 1.2
    noise = np.random.normal(0, 0.15, len(steps))
    losses = base_loss + noise
    losses = np.maximum(losses, 1.0)  # Ensure positive
    
    # Smooth the noise a bit for realism
    from scipy.ndimage import gaussian_filter1d
    losses = gaussian_filter1d(losses, sigma=1.5)
    
    # Validation perplexity (lower resolution, smoother)
    val_steps = np.arange(0, num_steps + 1, 500)
    val_base = 12.0 * np.exp(-0.00008 * val_steps) + 2.5
    val_noise = np.random.normal(0, 0.3, len(val_steps))
    val_ppl = val_base + val_noise
    val_ppl = np.maximum(val_ppl, 2.0)
    val_ppl = gaussian_filter1d(val_ppl, sigma=1.2)
    
    return steps, losses, val_steps, val_ppl, lrs


def plot_mlm_loss(steps, losses, out_path):
    """Plot MLM loss over training steps."""
    fig, ax = make_axes(size=(6.2, 3.8))
    
    ax.plot(
        steps,
        losses,
        color=COLORS["blue"],
        linewidth=2.5,
        label="MLM loss",
        zorder=10,
    )
    
    ax.fill_between(steps, losses, alpha=0.15, color=COLORS["blue"], zorder=5)
    
    ax.set_xlabel("Training steps", fontweight="semibold")
    ax.set_ylabel("Loss", fontweight="semibold")
    ax.set_title("Pretraining Loss Decay", fontweight="semibold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as thousands
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x/1000)}k"))
    
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close()


def plot_learning_rate_schedule(steps, lrs, out_path):
    """Plot learning rate schedule over training steps."""
    fig, ax = make_axes(size=(6.2, 3.8))
    
    ax.plot(
        steps,
        lrs,
        color=COLORS["orange"],
        linewidth=2.5,
        label="Learning rate",
        zorder=10,
    )
    
    ax.fill_between(steps, lrs, alpha=0.15, color=COLORS["orange"], zorder=5)
    
    ax.set_xlabel("Training steps", fontweight="semibold")
    ax.set_ylabel("Learning rate", fontweight="semibold")
    ax.set_title("Learning Rate Schedule", fontweight="semibold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Format axes
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x/1000)}k"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.0e}"))
    
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close()


def plot_dual_axis(steps, losses, val_steps, val_ppl, lrs, out_path):
    """Plot loss and learning rate on dual axes."""
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor(COLORS["panel"])
    
    # Loss axis (left)
    ax1.plot(
        steps,
        losses,
        color=COLORS["blue"],


#     )
#     
#     ax.fill_between(steps, lrs, alpha=0.15, color=COLORS["orange"], zorder=5)
#     
#     ax.set_xlabel("Training steps", fontweight="semibold")
#     ax.set_ylabel("Learning rate", fontweight="semibold")
#     ax.set_title("Learning Rate Schedule", fontweight="semibold", pad=12)
#     ax.legend(loc="upper right", framealpha=0.95)
#     ax.grid(True, alpha=0.3)
#     
#     # Format axes
#     ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x/1000)}k"))
#     ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.0e}"))
#     
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"✓ Saved: {out_path}")
#     plt.close()
# 
# 
# def plot_dual_axis(steps, losses, val_steps, val_ppl, lrs, out_path):
#     """Plot loss and learning rate on dual axes."""
#     fig, ax1 = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
#     fig.patch.set_facecolor("white")
#     ax1.set_facecolor(COLORS["panel"])
#     
#     # Loss axis (left)
#     ax1.plot(
#         steps,
#         losses,
#         color=COLORS["blue"],
