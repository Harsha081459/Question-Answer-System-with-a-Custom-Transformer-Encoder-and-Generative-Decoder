"""Generate all publication-quality report figures. Run: python generate_all_report_figures.py"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PAL = ["#2196F3","#FF5722","#4CAF50","#9C27B0"]
PHASE_COLORS = {"mlm":"#1565C0","ext":"#00897B","gen":"#E65100"}
INK, SUBTLE, GRID, PANEL = "#1F2A37","#6B7280","#D9E1EA","#F7FAFD"

def load_json(p): return json.loads(p.read_text("utf-8"))

def set_theme():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
        "axes.titlesize":16,"axes.labelsize":13,"axes.edgecolor":INK,
        "axes.linewidth":1.1,"axes.facecolor":PANEL,"axes.grid":True,
        "axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False,
        "grid.color":GRID,"grid.linewidth":0.8,"grid.alpha":0.4,
        "grid.linestyle":"--","legend.frameon":False,"legend.fontsize":11,
        "xtick.labelsize":11,"ytick.labelsize":11,
        "figure.facecolor":"white","savefig.facecolor":"white","savefig.bbox":"tight"})

def save(fig, name):
    for ext in ("png","pdf"):
        fig.savefig(FIGURES/f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")

def _box(ax,x,y,w,h,txt,fc,ec="#555",fs=9,tc=INK,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.015",
        lw=1.4,ec=ec,fc=fc))
    fw="bold" if bold else "medium"
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight=fw,linespacing=1.2)

def _arr(ax,s,e,c="#555",lw=1.8):
    ax.add_patch(FancyArrowPatch(s,e,arrowstyle="-|>",mutation_scale=14,lw=lw,color=c))

# ── Figure 1: System Architecture ──
def fig_architecture():
    fig,ax=plt.subplots(figsize=(14,5.5),dpi=300)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    # Phase labels
    for x,w,label,col in [(0.01,0.28,"Phase 1: MLM Pretraining",PHASE_COLORS["mlm"]),
                           (0.34,0.28,"Phase 2: Extractive QA",PHASE_COLORS["ext"]),
                           (0.67,0.32,"Phase 3: Generative QA",PHASE_COLORS["gen"])]:
        ax.add_patch(FancyBboxPatch((x,0.88),w,0.09,boxstyle="round,pad=0.01,rounding_size=0.02",lw=0,fc=col))
        ax.text(x+w/2,0.925,label,ha="center",va="center",color="white",fontsize=12,fontweight="bold")
    # Phase 1
    _box(ax,0.02,0.55,0.12,0.25,"Wikipedia\n+ C4\n(streaming)","#BBDEFB",ec=PHASE_COLORS["mlm"],fs=9)
    _arr(ax,(0.14,0.675),(0.16,0.675),PHASE_COLORS["mlm"])
    _box(ax,0.16,0.55,0.12,0.25,"15% Token\nMasking\n(80/10/10)","#90CAF9",ec=PHASE_COLORS["mlm"],fs=9)
    _arr(ax,(0.28,0.675),(0.295,0.675),PHASE_COLORS["mlm"])
    # Shared encoder (spans phases)
    _box(ax,0.295,0.45,0.19,0.40,"Custom BERT\nEncoder\n12 layers, 768-dim\n12 heads, FFN 3072","#E3F2FD",ec=PHASE_COLORS["mlm"],fs=9.5,bold=True)
    _arr(ax,(0.39,0.45),(0.39,0.35),PHASE_COLORS["mlm"])
    _box(ax,0.30,0.18,0.18,0.15,"MLM Head\n(tied weights)\nPredict masked","#E8EAF6",ec=PHASE_COLORS["mlm"],fs=8.5)


#     print(f"  -> {name}")
# 
# def _box(ax,x,y,w,h,txt,fc,ec="#555",fs=9,tc=INK,bold=False):
#     ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.015",
#         lw=1.4,ec=ec,fc=fc))
#     fw="bold" if bold else "medium"
#     ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight=fw,linespacing=1.2)
# 
# def _arr(ax,s,e,c="#555",lw=1.8):
#     ax.add_patch(FancyArrowPatch(s,e,arrowstyle="-|>",mutation_scale=14,lw=lw,color=c))
# 
# # ── Figure 1: System Architecture ──
# def fig_architecture():
#     fig,ax=plt.subplots(figsize=(14,5.5),dpi=300)
#     ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
#     # Phase labels
#     for x,w,label,col in [(0.01,0.28,"Phase 1: MLM Pretraining",PHASE_COLORS["mlm"]),
#                            (0.34,0.28,"Phase 2: Extractive QA",PHASE_COLORS["ext"]),
#                            (0.67,0.32,"Phase 3: Generative QA",PHASE_COLORS["gen"])]:
#         ax.add_patch(FancyBboxPatch((x,0.88),w,0.09,boxstyle="round,pad=0.01,rounding_size=0.02",lw=0,fc=col))
#         ax.text(x+w/2,0.925,label,ha="center",va="center",color="white",fontsize=12,fontweight="bold")
#     # Phase 1
#     _box(ax,0.02,0.55,0.12,0.25,"Wikipedia\n+ C4\n(streaming)","#BBDEFB",ec=PHASE_COLORS["mlm"],fs=9)
#     _arr(ax,(0.14,0.675),(0.16,0.675),PHASE_COLORS["mlm"])
#     _box(ax,0.16,0.55,0.12,0.25,"15% Token\nMasking\n(80/10/10)","#90CAF9",ec=PHASE_COLORS["mlm"],fs=9)
#     _arr(ax,(0.28,0.675),(0.295,0.675),PHASE_COLORS["mlm"])
#     # Shared encoder (spans phases)
#     _box(ax,0.295,0.45,0.19,0.40,"Custom BERT\nEncoder\n12 layers, 768-dim\n12 heads, FFN 3072","#E3F2FD",ec=PHASE_COLORS["mlm"],fs=9.5,bold=True)
#     _arr(ax,(0.39,0.45),(0.39,0.35),PHASE_COLORS["mlm"])
#     _box(ax,0.30,0.18,0.18,0.15,"MLM Head\n(tied weights)\nPredict masked","#E8EAF6",ec=PHASE_COLORS["mlm"],fs=8.5)
