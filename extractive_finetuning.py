import argparse
import collections
import json
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mlm_pretraining import BertEncoder, ModelConfig


class BertForQuestionAnswering(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = BertEncoder(cfg)
        self.qa_outputs = nn.Linear(cfg.hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions=None,
        end_positions=None,
    ):
        hidden = self.encoder(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(hidden)
        start_logits, end_logits = logits[..., 0], logits[..., 1]

        loss = None
        if start_positions is not None and end_positions is not None:
            start_loss = nn.functional.cross_entropy(start_logits, start_positions)
            end_loss = nn.functional.cross_entropy(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        if loss is None:
            return {"start_logits": start_logits, "end_logits": end_logits}
        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="squad", choices=["squad", "squad_v2"])
    p.add_argument("--output_dir", type=str, default="checkpoints_qa_squad")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def load_pretrained_encoder(model: BertForQuestionAnswering, checkpoint_dir: str):
    ckpt_path = Path(checkpoint_dir) / "checkpoint.pt"
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload["model"]
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=True)
    return int(payload.get("step", 0))



# 
# 
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--checkpoint_dir", type=str, required=True)
#     p.add_argument("--dataset", type=str, default="squad", choices=["squad", "squad_v2"])
#     p.add_argument("--output_dir", type=str, default="checkpoints_qa_squad")
#     p.add_argument("--max_length", type=int, default=384)
#     p.add_argument("--doc_stride", type=int, default=128)
#     p.add_argument("--per_device_batch_size", type=int, default=8)
#     p.add_argument("--grad_accum", type=int, default=2)
#     p.add_argument("--learning_rate", type=float, default=3e-5)
#     p.add_argument("--num_train_epochs", type=float, default=2.0)
#     p.add_argument("--warmup_ratio", type=float, default=0.06)
#     p.add_argument("--weight_decay", type=float, default=0.01)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--num_workers", type=int, default=4)
#     p.add_argument("--fp16", action="store_true")
#     p.add_argument("--bf16", action="store_true")
#     return p.parse_args()
# 
# 
# def load_pretrained_encoder(model: BertForQuestionAnswering, checkpoint_dir: str):
#     ckpt_path = Path(checkpoint_dir) / "checkpoint.pt"
#     payload = torch.load(str(ckpt_path), map_location="cpu")
#     state = payload["model"]
#     enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
#     model.encoder.load_state_dict(enc_state, strict=True)
#     return int(payload.get("step", 0))
# 
