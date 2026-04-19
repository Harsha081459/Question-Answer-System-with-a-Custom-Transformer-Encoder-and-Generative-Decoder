"""Regenerate the report figures with a cleaner, more consistent visual style.

Run from the project root after the evaluation JSON files and training histories
have been produced:

    source .venv/bin/activate
    python generate_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 1.1,
            "axes.facecolor": COLORS["panel"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.9,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def make_axes(size=(6.4, 4.0)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["panel"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def annotate_vertical_bars(ax, bars, fmt="{:.1f}", dy=3):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=COLORS["ink"],
            fontweight="semibold",
        )


def annotate_horizontal_bars(ax, bars, fmt="{:.2f}", dx=4):
    for bar in bars:
        value = bar.get_width()
        ax.annotate(
            fmt.format(value),
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
            fontweight="semibold",
        )


def pretty_extractive_name(name: str) -> str:
    mapping = {
        "hf_deepset_roberta-base-squad2": "RoBERTa-base",
        "hf_bert-large-uncased-whole-word-masking-finetuned-squad": "BERT-large",
        "custom_scratch_encoder_squadv2": "Scratch encoder",
        "hf_distilbert-base-uncased-distilled-squad": "DistilBERT",
    }
    return mapping.get(name, name.replace("hf_", "").replace("_", " "))


def pretty_generative_name(name: str) -> str:
    mapping = {
        "our_hybrid_decoder": "Hybrid",
        "t5-small": "T5-small",
        "t5-base": "T5-base",
        "google_flan-t5-small": "Flan-T5-small",
    }
    return mapping.get(name, name.replace("_", " "))


def training_history_plot(history_path: Path, title: str, out_path: Path, color: str) -> None:
    history = load_json(history_path)
    epochs = [int(item["epoch"]) for item in history]
    losses = [float(item["train_loss"]) for item in history]

    fig, ax = make_axes(size=(6.35, 3.95))

    if len(epochs) > 1:
        ax.plot(
            epochs,
            losses,
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=9,
            markerfacecolor="white",


#     }
#     return mapping.get(name, name.replace("hf_", "").replace("_", " "))
# 
# 
# def pretty_generative_name(name: str) -> str:
#     mapping = {
#         "our_hybrid_decoder": "Hybrid",
#         "t5-small": "T5-small",
#         "t5-base": "T5-base",
#         "google_flan-t5-small": "Flan-T5-small",
#     }
#     return mapping.get(name, name.replace("_", " "))
# 
# 
# def training_history_plot(history_path: Path, title: str, out_path: Path, color: str) -> None:
#     history = load_json(history_path)
#     epochs = [int(item["epoch"]) for item in history]
#     losses = [float(item["train_loss"]) for item in history]
# 
#     fig, ax = make_axes(size=(6.35, 3.95))
# 
#     if len(epochs) > 1:
#         ax.plot(
#             epochs,
#             losses,
#             color=color,
#             linewidth=3.0,
#             marker="o",
#             markersize=9,
#             markerfacecolor="white",
