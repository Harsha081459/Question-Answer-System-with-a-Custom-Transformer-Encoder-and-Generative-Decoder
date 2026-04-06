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
        noans_correct = 0
        ans_total = 0
        ans_correct = 0
        for is_noans, diff in zip(is_noans_flags, score_diffs):
            pred_noans = diff > th
            if is_noans:
                noans_total += 1
                if pred_noans:
                    noans_correct += 1
            else:
                ans_total += 1
                if not pred_noans:
                    ans_correct += 1
        noans_acc = 100.0 * noans_correct / max(1, noans_total)
        ans_acc = 100.0 * ans_correct / max(1, ans_total)
        gate_balance = 0.5 * (noans_acc + ans_acc)
        if gate_balance > best["gate_balance"]:
            best = {
                "threshold": th,
                "gate_balance": gate_balance,
                "no_answer_accuracy": noans_acc,
                "answerable_accuracy": ans_acc,
            }
    best["threshold_scan"] = [thresholds[0], thresholds[-1], len(thresholds)]
    return float(best["threshold"]), best


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    val_loader,
    device,
    beam_size=4,
    max_new_tokens=32,
    length_penalty=1.0,
    no_answer_text=NO_ANSWER_TEXT,
    enable_no_answer_gate=False,
    no_answer_threshold=0.0,
    tune_no_answer_threshold=False,
    threshold_points=101,
    max_eval_examples=0,
):
    model.eval()
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.pad_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.pad_token_id
    pad = tokenizer.pad_token_id

    gate_active = enable_no_answer_gate or tune_no_answer_threshold
    noans_target_ids = None
    if gate_active:
        noans_target_ids = _build_target_ids(
            tokenizer=tokenizer,
            text=no_answer_text,
            bos=bos,
            eos=eos,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    raw_preds = []
    golds = []
    is_noans_flags = []
    score_diffs = []

    processed = 0
    stop_early = max_eval_examples is not None and int(max_eval_examples) > 0
    for batch in val_loader:
        bsz = batch["encoder_input_ids"].size(0)
        for i in range(bsz):
            if stop_early and processed >= int(max_eval_examples):
                break
            enc_ids = batch["encoder_input_ids"][i : i + 1].to(device)
            enc_ttype = batch["encoder_token_type_ids"][i : i + 1].to(device)
            enc_mask = batch["encoder_attention_mask"][i : i + 1].to(device)
            if gate_active:
                out, pred_logprob, _ = model.generate(
                    encoder_input_ids=enc_ids,
                    encoder_token_type_ids=enc_ttype,
                    encoder_attention_mask=enc_mask,
                    bos_token_id=bos,


#         noans_target_ids = _build_target_ids(
#             tokenizer=tokenizer,
#             text=no_answer_text,
#             bos=bos,
#             eos=eos,
#             max_new_tokens=max_new_tokens,
#             device=device,
#         )
# 
#     raw_preds = []
#     golds = []
#     is_noans_flags = []
#     score_diffs = []
# 
#     processed = 0
#     stop_early = max_eval_examples is not None and int(max_eval_examples) > 0
#     for batch in val_loader:
#         bsz = batch["encoder_input_ids"].size(0)
#         for i in range(bsz):
#             if stop_early and processed >= int(max_eval_examples):
#                 break
#             enc_ids = batch["encoder_input_ids"][i : i + 1].to(device)
#             enc_ttype = batch["encoder_token_type_ids"][i : i + 1].to(device)
#             enc_mask = batch["encoder_attention_mask"][i : i + 1].to(device)
#             if gate_active:
#                 out, pred_logprob, _ = model.generate(
#                     encoder_input_ids=enc_ids,
#                     encoder_token_type_ids=enc_ttype,
#                     encoder_attention_mask=enc_mask,
#                     bos_token_id=bos,
