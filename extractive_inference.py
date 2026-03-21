import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from extractive_finetuning import BertForQuestionAnswering
from mlm_pretraining import ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="Run QA inference on custom question/context.")
    p.add_argument("--model_dir", required=True, help="Path to finetuned model folder")
    p.add_argument("--pretrain_config_dir", default="", help="Path with model_config.json if missing in model_dir")
    p.add_argument("--question", required=True, help="Question text")
    p.add_argument("--context", required=True, help="Context paragraph")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=64)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--no_answer_threshold", type=float, default=None, help="Use for SQuAD v2 style no-answer")
    return p.parse_args()


def load_model_and_tokenizer(model_dir: Path, pretrain_config_dir: Path | None):
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        if pretrain_config_dir is None:
            raise FileNotFoundError(
                "model_config.json missing in model_dir. Pass --pretrain_config_dir containing model_config.json."
            )
        config_path = pretrain_config_dir / "model_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = ModelConfig(**json.load(f))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    model = BertForQuestionAnswering(cfg)
    state_path = model_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    state = load_file(str(state_path))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, cfg


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    pre_cfg_dir = Path(args.pretrain_config_dir) if args.pretrain_config_dir else None

    model, tokenizer, cfg = load_model_and_tokenizer(model_dir, pre_cfg_dir)

    max_length = min(args.max_length, cfg.max_position_embeddings)
    doc_stride = min(args.doc_stride, max(8, max_length // 4))

    enc = tokenizer(
        [args.question],
        [args.context],
        truncation="only_second",
        max_length=max_length,


# 
#     tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.sep_token
# 
#     model = BertForQuestionAnswering(cfg)
#     state_path = model_dir / "model.safetensors"
#     if not state_path.exists():
#         raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
#     state = load_file(str(state_path))
#     model.load_state_dict(state, strict=True)
#     model.eval()
#     return model, tokenizer, cfg
# 
# 
# def main():
#     args = parse_args()
#     model_dir = Path(args.model_dir)
#     pre_cfg_dir = Path(args.pretrain_config_dir) if args.pretrain_config_dir else None
# 
#     model, tokenizer, cfg = load_model_and_tokenizer(model_dir, pre_cfg_dir)
# 
#     max_length = min(args.max_length, cfg.max_position_embeddings)
#     doc_stride = min(args.doc_stride, max(8, max_length // 4))
# 
#     enc = tokenizer(
#         [args.question],
#         [args.context],
#         truncation="only_second",
#         max_length=max_length,
