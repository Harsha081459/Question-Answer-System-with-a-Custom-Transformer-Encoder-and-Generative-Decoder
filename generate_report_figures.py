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
            markeredgecolor=color,
            markeredgewidth=2.0,
        )
    else:
        ax.scatter(
            epochs,
            losses,
            s=130,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )

    for epoch, loss in zip(epochs, losses):
        ax.annotate(
            f"{loss:.3f}",
            xy=(epoch, loss),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["ink"],
            fontweight="semibold",
        )

    start = losses[0]
    final = losses[-1]
    delta = final - start
    summary_lines = [
        f"Epochs logged: {len(epochs)}",
        f"Start loss: {start:.3f}",
        f"Final loss: {final:.3f}",
    ]
    if len(epochs) > 1:
        summary_lines.append(f"Delta: {delta:+.3f}")
    else:
        summary_lines.append("Single logged epoch")

    ax.text(
        0.02,
        0.06,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#EEF4FB",
            "edgecolor": "#C9D5E4",
            "linewidth": 1.0,
        },
    )

    ax.set_title(title, pad=12, fontweight="semibold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    epoch_min = min(epochs)
    epoch_max = max(epochs)
    if epoch_min == epoch_max:
        ax.set_xlim(epoch_min - 0.5, epoch_max + 0.5)
    else:
        ax.set_xlim(epoch_min - 0.15, epoch_max + 0.15)

    if len(losses) == 1:
        pad = max(0.03, abs(final) * 0.08)
    else:
        span = max(losses) - min(losses)
        pad = max(0.03, span * 0.30)
    ax.set_ylim(min(losses) - pad, max(losses) + pad)

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def extractive_baseline_plot(data_path: Path, out_path: Path) -> None:
    data = load_json(data_path)
    rows = data["ranking_by_best_f1"]

    labels = [pretty_extractive_name(row["name"]) for row in rows]
    em = []
    f1 = []
    for row in rows:
        model = next(item for item in data["models"] if item["name"] == row["name"])
        summary = model["summary"]
        em.append(float(summary["best_exact"]))
        f1.append(float(summary["best_f1"]))

    x = list(range(len(labels)))
    width = 0.34

    fig, ax = make_axes(size=(7.4, 4.15))
    em_bars = ax.bar([i - width / 2 for i in x], em, width=width, color=COLORS["blue"], label="EM")
    f1_bars = ax.bar([i + width / 2 for i in x], f1, width=width, color=COLORS["orange"], label="F1")

    annotate_vertical_bars(ax, em_bars, fmt="{:.1f}")
    annotate_vertical_bars(ax, f1_bars, fmt="{:.1f}")

    ax.set_title("Extractive QA baselines on SQuAD v2", pad=12, fontweight="semibold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.set_ylim(0, max(max(em), max(f1)) * 1.16)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(loc="upper right", ncol=2)
    ax.text(
        0.02,
        0.95,
        "Higher is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=COLORS["subtle"],
    )

    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def generative_comparison_plot(data_path: Path, out_path: Path) -> None:
    data = load_json(data_path)
    rows = data["ranking_by_f1"]

    labels = []
    scores = []
    colors = []
    for row in rows:
        model = next(item for item in data["models"] if item["name"] == row["name"])
        labels.append(pretty_generative_name(row["name"]))
        scores.append(float(row["f1"]))
        if model["name"] == "our_hybrid_decoder":
            colors.append(COLORS["blue"])
        elif model["name"] == "t5-base":
            colors.append(COLORS["green"])
        elif model["name"] == "t5-small":
            colors.append(COLORS["gray"])
        else:
            colors.append(COLORS["teal"])

    fig, ax = make_axes(size=(7.1, 4.0))
    bars = ax.bar(labels, scores, color=colors, width=0.72)
    annotate_vertical_bars(ax, bars, fmt="{:.1f}")

    ax.set_title("Generative QA F1 on the shared 1k validation subset", pad=12, fontweight="semibold")


# 
#     fig.savefig(out_path, dpi=320)
#     plt.close(fig)
# 
# 
# def generative_comparison_plot(data_path: Path, out_path: Path) -> None:
#     data = load_json(data_path)
#     rows = data["ranking_by_f1"]
# 
#     labels = []
#     scores = []
#     colors = []
#     for row in rows:
#         model = next(item for item in data["models"] if item["name"] == row["name"])
#         labels.append(pretty_generative_name(row["name"]))
#         scores.append(float(row["f1"]))
#         if model["name"] == "our_hybrid_decoder":
#             colors.append(COLORS["blue"])
#         elif model["name"] == "t5-base":
#             colors.append(COLORS["green"])
#         elif model["name"] == "t5-small":
#             colors.append(COLORS["gray"])
#         else:
#             colors.append(COLORS["teal"])
# 
#     fig, ax = make_axes(size=(7.1, 4.0))
#     bars = ax.bar(labels, scores, color=colors, width=0.72)
#     annotate_vertical_bars(ax, bars, fmt="{:.1f}")
# 
#     ax.set_title("Generative QA F1 on the shared 1k validation subset", pad=12, fontweight="semibold")
