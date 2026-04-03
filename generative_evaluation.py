import argparse
import json
from pathlib import Path

import evaluate
import torch

from generative_data import NO_ANSWER_TEXT, GenQADataConfig, build_dataloaders, normalize_text
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(p_toks))
    rec = overlap / max(1, len(g_toks))
    return 2 * prec * rec / max(1e-8, prec + rec)


def _decode_generated_ids(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
    text_ids = []
    for tid in out_ids:
        if tid in {bos, pad}:
            continue
        if tid == eos:
            break
        text_ids.append(tid)
    return tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def _build_target_ids(tokenizer, text: str, bos: int, eos: int, max_new_tokens: int, device: str) -> torch.Tensor:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_new_tokens - 1),
    )["input_ids"]
    seq = [bos] + ids + [eos]
    return torch.tensor([seq], dtype=torch.long, device=device)


def _select_gate_threshold(
    is_noans_flags,
    score_diffs,
    threshold_points: int,
):
    if not score_diffs:
        return 0.0, {"threshold_scan": [], "gate_balance": 0.0}

    lo = min(score_diffs)
    hi = max(score_diffs)
    if lo == hi:
        thresholds = [lo]
    else:
        steps = max(3, threshold_points)
        thresholds = [lo + (hi - lo) * (i / (steps - 1)) for i in range(steps)]
    thresholds.append(0.0)
    thresholds = sorted(set(round(t, 6) for t in thresholds))

    best = {
        "threshold": 0.0,
        "gate_balance": -1.0,
        "no_answer_accuracy": 0.0,
        "answerable_accuracy": 0.0,
    }
    for th in thresholds:
        noans_total = 0


#     seq = [bos] + ids + [eos]
#     return torch.tensor([seq], dtype=torch.long, device=device)
# 
# 
# def _select_gate_threshold(
#     is_noans_flags,
#     score_diffs,
#     threshold_points: int,
# ):
#     if not score_diffs:
#         return 0.0, {"threshold_scan": [], "gate_balance": 0.0}
# 
#     lo = min(score_diffs)
#     hi = max(score_diffs)
#     if lo == hi:
#         thresholds = [lo]
#     else:
#         steps = max(3, threshold_points)
#         thresholds = [lo + (hi - lo) * (i / (steps - 1)) for i in range(steps)]
#     thresholds.append(0.0)
#     thresholds = sorted(set(round(t, 6) for t in thresholds))
# 
#     best = {
#         "threshold": 0.0,
#         "gate_balance": -1.0,
#         "no_answer_accuracy": 0.0,
#         "answerable_accuracy": 0.0,
#     }
#     for th in thresholds:
#         noans_total = 0
