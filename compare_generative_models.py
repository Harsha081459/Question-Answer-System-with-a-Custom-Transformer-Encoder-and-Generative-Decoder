import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generative_data import NO_ANSWER_TEXT, normalize_text
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


DEFAULT_HF_MODELS = ["t5-small", "t5-base", "google/flan-t5-small"]


@dataclass
class ModelRunSpec:
    name: str
    kind: str  # "custom" or "hf"
    ref: str
    tokenizer_ref: str
    decoder_variant: str = "hybrid"


def parse_args():
    p = argparse.ArgumentParser(description="Compare generative QA models on SQuAD v2 validation.")
    p.add_argument("--custom_checkpoint", default="checkpoints_generative_qa_hybrid_span_restart1_20260426_010636/latest.pt")
    p.add_argument("--custom_tokenizer", default="checkpoints_pretrain_base_seq256/step_20000")
    p.add_argument("--custom_name", default="our_hybrid_decoder")
    p.add_argument("--hf_models", default=",".join(DEFAULT_HF_MODELS))
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--beam_size", type=int, default=1)
    p.add_argument("--length_penalty", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_eval_examples", type=int, default=4000)
    p.add_argument("--instruction_prefix", default="")
    p.add_argument("--no_answer_text", default=NO_ANSWER_TEXT)
    p.add_argument("--output_json", default="comparison_generative_seq2seq.json")
    p.add_argument("--output_csv", default="comparison_generative_seq2seq.csv")
    return p.parse_args()


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


def build_prompt(question: str, context: str, instruction_prefix: str = "") -> str:
    if instruction_prefix:
        return f"{instruction_prefix.strip()} question: {question} context: {context}"
    return f"question: {question} context: {context}"


def load_eval_dataset(max_eval_examples: int):
    ds = load_dataset("squad_v2", split="validation")
    if max_eval_examples > 0:
        ds = ds.select(range(min(max_eval_examples, len(ds))))
    return ds


def load_custom_model(checkpoint_path: str, tokenizer_path: str, device: str, decoder_variant: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enc_cfg = ModelConfig(**payload["encoder_config"])
    dec_cfg = DecoderConfig(**payload["decoder_config"])
    model = GenerativeQAModelHybrid(enc_cfg, dec_cfg) if decoder_variant == "hybrid" else StandardGenerativeQAModel(enc_cfg, dec_cfg)
    model.load_state_dict(payload["model"], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token
    model.to(device).eval()
    return model, tokenizer


def load_hf_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device).eval()
    return model, tokenizer


def decode_custom(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
    text_ids = []
    for tid in out_ids:
        if tid in {bos, pad}:
            continue
        if tid == eos:
            break
        text_ids.append(tid)
    return tokenizer.decode(text_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def run_model(spec: ModelRunSpec, dataset, args, device: str):
    if spec.kind == "custom":
        model, tokenizer = load_custom_model(spec.ref, spec.tokenizer_ref, device, spec.decoder_variant)
        is_custom = True
    else:
        model, tokenizer = load_hf_model(spec.ref, device)
        is_custom = False

    preds = []
    golds = []
    is_noans_flags = []
    total_len = 0

    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
        questions = [ex["question"] for ex in batch]
        contexts = [ex["context"] for ex in batch]
        targets = []
        inputs = [build_prompt(q, c, args.instruction_prefix) for q, c in zip(questions, contexts)]
        for ex in batch:
            ans = ex["answers"]["text"]
            gold = args.no_answer_text if len(ans) == 0 else ans[0].strip()
            targets.append(gold)
            is_noans_flags.append(len(ans) == 0)

        if is_custom:
            batch_preds = []
            for inp in inputs:
                enc = tokenizer(
                    [inp],
                    truncation=True,
                    max_length=args.max_input_len,
                    padding=True,
                    return_tensors="pt",
                )


#     golds = []
#     is_noans_flags = []
#     total_len = 0
# 
#     bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
#     eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
#     pad = tokenizer.pad_token_id
# 
#     for start in range(0, len(dataset), args.batch_size):
#         batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
#         questions = [ex["question"] for ex in batch]
#         contexts = [ex["context"] for ex in batch]
#         targets = []
#         inputs = [build_prompt(q, c, args.instruction_prefix) for q, c in zip(questions, contexts)]
#         for ex in batch:
#             ans = ex["answers"]["text"]
#             gold = args.no_answer_text if len(ans) == 0 else ans[0].strip()
#             targets.append(gold)
#             is_noans_flags.append(len(ans) == 0)
# 
#         if is_custom:
#             batch_preds = []
#             for inp in inputs:
#                 enc = tokenizer(
#                     [inp],
#                     truncation=True,
#                     max_length=args.max_input_len,
#                     padding=True,
#                     return_tensors="pt",
#                 )
