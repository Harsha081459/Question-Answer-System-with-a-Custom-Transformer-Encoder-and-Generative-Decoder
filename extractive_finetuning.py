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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    with open(ckpt_dir / "model_config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)
    if args.max_length > cfg.max_position_embeddings:
        raise ValueError(
            f"--max_length ({args.max_length}) exceeds pretrained max_position_embeddings "
            f"({cfg.max_position_embeddings}). Use --max_length {cfg.max_position_embeddings}."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    model = BertForQuestionAnswering(cfg)
    restored_step = load_pretrained_encoder(model, str(ckpt_dir))
    print(f"Loaded encoder weights from step={restored_step}")

    dataset = load_dataset(args.dataset)
    train_examples = dataset["train"]
    eval_examples = dataset["validation"]

    max_length = args.max_length
    # Keep stride safely below tokenizer's effective max len to avoid tokenizers panic.
    doc_stride = min(args.doc_stride, max(8, max_length // 4))

    def prepare_train_features(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:


#         start_positions = []
#         end_positions = []
#         for i, offsets in enumerate(offset_mapping):
#             input_ids = tokenized["input_ids"][i]
#             cls_index = input_ids.index(tokenizer.cls_token_id)
#             sequence_ids = tokenized.sequence_ids(i)
#             sample_idx = sample_mapping[i]
#             answers = examples["answers"][sample_idx]
# 
#             if len(answers["answer_start"]) == 0:
#                 start_positions.append(cls_index)
#                 end_positions.append(cls_index)
#                 continue
# 
#             start_char = answers["answer_start"][0]
#             end_char = start_char + len(answers["text"][0])
# 
#             token_start_index = 0
#             while sequence_ids[token_start_index] != 1:
#                 token_start_index += 1
# 
#             token_end_index = len(input_ids) - 1
#             while sequence_ids[token_end_index] != 1:
#                 token_end_index -= 1
# 
#             if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
#                 start_positions.append(cls_index)
#                 end_positions.append(cls_index)
#             else:
#                 while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
