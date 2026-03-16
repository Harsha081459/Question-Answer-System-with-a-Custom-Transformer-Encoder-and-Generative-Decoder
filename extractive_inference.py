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


# 
# import torch
# from safetensors.torch import load_file
# from transformers import AutoTokenizer
# 
# from extractive_finetuning import BertForQuestionAnswering
# from mlm_pretraining import ModelConfig
# 
# 
# def parse_args():
#     p = argparse.ArgumentParser(description="Run QA inference on custom question/context.")
#     p.add_argument("--model_dir", required=True, help="Path to finetuned model folder")
#     p.add_argument("--pretrain_config_dir", default="", help="Path with model_config.json if missing in model_dir")
#     p.add_argument("--question", required=True, help="Question text")
#     p.add_argument("--context", required=True, help="Context paragraph")
#     p.add_argument("--max_length", type=int, default=256)
#     p.add_argument("--doc_stride", type=int, default=64)
#     p.add_argument("--n_best", type=int, default=20)
#     p.add_argument("--max_answer_length", type=int, default=30)
#     p.add_argument("--no_answer_threshold", type=float, default=None, help="Use for SQuAD v2 style no-answer")
#     return p.parse_args()
# 
# 
# def load_model_and_tokenizer(model_dir: Path, pretrain_config_dir: Path | None):
#     config_path = model_dir / "model_config.json"
#     if not config_path.exists():
#         if pretrain_config_dir is None:
#             raise FileNotFoundError(
#                 "model_config.json missing in model_dir. Pass --pretrain_config_dir containing model_config.json."
#             )
