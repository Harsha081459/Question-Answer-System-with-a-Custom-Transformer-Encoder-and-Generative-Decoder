import argparse
import collections
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import inspect

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DataCollatorWithPadding

from extractive_finetuning import BertForQuestionAnswering
from mlm_pretraining import ModelConfig


DEFAULT_HF_MODELS = [
    "deepset/roberta-base-squad2",
    "distilbert-base-uncased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
]


@dataclass
class ModelRunSpec:
    name: str
    source: str  # "custom" or "hf"
    model_ref: str
    tokenizer_ref: str
    config_ref: Optional[str] = None


def parse_args():
    p = argparse.ArgumentParser(description="Compare extractive QA models on SQuAD v2 validation.")
    p.add_argument(
        "--custom_model_dir",
        default="checkpoints_qa_squad_v2_lr5e-5_len256_e3",
        help="Directory containing model.safetensors and tokenizer files for custom QA model",
    )
    p.add_argument(
        "--custom_config_dir",
        default="checkpoints_pretrain_base_seq256/step_20000",
        help="Directory containing model_config.json for custom QA model",
    )
    p.add_argument(
        "--hf_models",
        default=",".join(DEFAULT_HF_MODELS),
        help="Comma-separated Hugging Face QA model IDs",
    )
    p.add_argument("--skip_custom", action="store_true")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--max_eval_examples", type=int, default=0, help="0 means full validation split")
    p.add_argument("--output_json", default="comparison_squadv2_results.json")
    p.add_argument("--output_csv", default="comparison_squadv2_results.csv")
    p.add_argument("--review_txt", default="comparison_squadv2_review.txt")
    return p.parse_args()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_custom_model_and_tokenizer(model_dir: Path, config_dir: Path, device: str):
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.exists():
        cfg_path = config_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {model_dir} or {config_dir}")

    cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token

    model = BertForQuestionAnswering(cfg)
    state_path = model_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    state = load_file(str(state_path))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, tok


def load_hf_model_and_tokenizer(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tok


def build_eval_dataset(max_eval_examples: int):
    ds = load_dataset("squad_v2", split="validation")
    if max_eval_examples > 0:
        ds = ds.select(range(min(max_eval_examples, len(ds))))
    return ds


def tokenize_with_overflow(dataset, tokenizer, max_length: int, doc_stride: int):
    doc_stride = min(doc_stride, max(8, max_length // 4))

    def prep(examples):
        tok = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,


# 
# 
# def load_hf_model_and_tokenizer(model_id: str, device: str):
#     tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token or tok.sep_token
#     model = AutoModelForQuestionAnswering.from_pretrained(model_id)
#     model.to(device)
#     model.eval()
#     return model, tok
# 
# 
# def build_eval_dataset(max_eval_examples: int):
#     ds = load_dataset("squad_v2", split="validation")
#     if max_eval_examples > 0:
#         ds = ds.select(range(min(max_eval_examples, len(ds))))
#     return ds
# 
# 
# def tokenize_with_overflow(dataset, tokenizer, max_length: int, doc_stride: int):
#     doc_stride = min(doc_stride, max(8, max_length // 4))
# 
#     def prep(examples):
#         tok = tokenizer(
#             examples["question"],
#             examples["context"],
#             truncation="only_second",
#             max_length=max_length,
#             stride=doc_stride,
#             return_overflowing_tokens=True,
