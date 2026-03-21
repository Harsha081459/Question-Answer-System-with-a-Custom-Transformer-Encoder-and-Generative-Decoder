import argparse
import collections
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from extractive_finetuning import BertForQuestionAnswering, load_pretrained_encoder
from mlm_pretraining import ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--dataset", default="squad", choices=["squad", "squad_v2"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--doc_stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_json", default="qa_error_analysis.json")
    return p.parse_args()


def qtype(question: str) -> str:
    q = question.strip().lower()
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("who"):
        return "who"
    if q.startswith("why"):
        return "why"
    return "other"


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    cfg = ModelConfig(**json.loads((ckpt_dir / "model_config.json").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    model = BertForQuestionAnswering(cfg)
    load_pretrained_encoder(model, str(ckpt_dir))

    data = load_dataset(args.dataset)["validation"]
    doc_stride = min(args.doc_stride, max(8, args.max_length // 4))

    def prep(examples):
        tok = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=args.max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sm = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            sid = tok.sequence_ids(i)


#         return "why"
#     return "other"
# 
# 
# def main():
#     args = parse_args()
#     ckpt_dir = Path(args.checkpoint_dir)
#     cfg = ModelConfig(**json.loads((ckpt_dir / "model_config.json").read_text(encoding="utf-8")))
#     tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
#     model = BertForQuestionAnswering(cfg)
#     load_pretrained_encoder(model, str(ckpt_dir))
# 
#     data = load_dataset(args.dataset)["validation"]
#     doc_stride = min(args.doc_stride, max(8, args.max_length // 4))
# 
#     def prep(examples):
#         tok = tokenizer(
#             examples["question"],
#             examples["context"],
#             truncation="only_second",
#             max_length=args.max_length,
#             stride=doc_stride,
#             return_overflowing_tokens=True,
#             return_offsets_mapping=True,
#             padding=False,
#         )
#         sm = tok.pop("overflow_to_sample_mapping")
#         tok["example_id"] = []
#         for i in range(len(tok["input_ids"])):
#             sid = tok.sequence_ids(i)
