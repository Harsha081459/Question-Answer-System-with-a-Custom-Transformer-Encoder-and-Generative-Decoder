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


# import json
# from pathlib import Path
# 
# import evaluate
# import numpy as np
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, Trainer, TrainingArguments
# 
# from extractive_finetuning import BertForQuestionAnswering, load_pretrained_encoder
# from mlm_pretraining import ModelConfig
# 
# 
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--checkpoint_dir", required=True)
#     p.add_argument("--dataset", default="squad", choices=["squad", "squad_v2"])
#     p.add_argument("--max_length", type=int, default=128)
#     p.add_argument("--doc_stride", type=int, default=32)
#     p.add_argument("--batch_size", type=int, default=8)
#     p.add_argument("--out_json", default="qa_error_analysis.json")
#     return p.parse_args()
# 
# 
# def qtype(question: str) -> str:
#     q = question.strip().lower()
#     if q.startswith("when"):
#         return "when"
#     if q.startswith("where"):
#         return "where"
