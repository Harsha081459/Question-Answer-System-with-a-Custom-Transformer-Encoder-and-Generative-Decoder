import argparse
import collections
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from extractive_finetuning import BertForQuestionAnswering
from mlm_pretraining import ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_checkpoint_dir", required=True, help="Dir with model_config.json")
    p.add_argument("--finetuned_model_dir", required=True, help="Dir with model.safetensors")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold_points", type=int, default=201)
    p.add_argument("--out_json", default="squad_v2_threshold_tuning.json")
    return p.parse_args()


def main():
    args = parse_args()
    pretrain_dir = Path(args.pretrain_checkpoint_dir)
    finetuned_dir = Path(args.finetuned_model_dir)

    cfg = ModelConfig(**json.loads((pretrain_dir / "model_config.json").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(str(finetuned_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    model = BertForQuestionAnswering(cfg)
    state = load_file(str(finetuned_dir / "model.safetensors"))
    model.load_state_dict(state, strict=True)



# from mlm_pretraining import ModelConfig
# 
# 
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--pretrain_checkpoint_dir", required=True, help="Dir with model_config.json")
#     p.add_argument("--finetuned_model_dir", required=True, help="Dir with model.safetensors")
#     p.add_argument("--max_length", type=int, default=256)
#     p.add_argument("--doc_stride", type=int, default=64)
#     p.add_argument("--batch_size", type=int, default=8)
#     p.add_argument("--num_workers", type=int, default=4)
#     p.add_argument("--threshold_points", type=int, default=201)
#     p.add_argument("--out_json", default="squad_v2_threshold_tuning.json")
#     return p.parse_args()
# 
# 
# def main():
#     args = parse_args()
#     pretrain_dir = Path(args.pretrain_checkpoint_dir)
#     finetuned_dir = Path(args.finetuned_model_dir)
# 
#     cfg = ModelConfig(**json.loads((pretrain_dir / "model_config.json").read_text(encoding="utf-8")))
#     tokenizer = AutoTokenizer.from_pretrained(str(finetuned_dir), use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.sep_token
# 
#     model = BertForQuestionAnswering(cfg)
#     state = load_file(str(finetuned_dir / "model.safetensors"))
#     model.load_state_dict(state, strict=True)
# 
