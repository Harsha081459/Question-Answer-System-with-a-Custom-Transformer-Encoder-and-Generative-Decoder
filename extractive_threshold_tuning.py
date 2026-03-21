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

    data = load_dataset("squad_v2")["validation"]
    max_length = min(args.max_length, cfg.max_position_embeddings)
    doc_stride = min(args.doc_stride, max(8, max_length // 4))

    def prep(examples):
        tok = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            sids = tok.sequence_ids(i)
            ex_i = sample_map[i]
            tok["example_id"].append(examples["id"][ex_i])
            tok["offset_mapping"][i] = [o if sids[k] == 1 else None for k, o in enumerate(tok["offset_mapping"][i])]
        return tok

    eval_features = data.map(prep, batched=True, remove_columns=data.column_names)
    eval_features_model = eval_features.remove_columns(["example_id", "offset_mapping"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="tmp_eval_squadv2_threshold",
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=args.num_workers,
            report_to="none",
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )

    preds, _, _ = trainer.predict(eval_features_model)
    start_logits, end_logits = preds



#             padding=False,
#         )
#         sample_map = tok.pop("overflow_to_sample_mapping")
#         tok["example_id"] = []
#         for i in range(len(tok["input_ids"])):
#             sids = tok.sequence_ids(i)
#             ex_i = sample_map[i]
#             tok["example_id"].append(examples["id"][ex_i])
#             tok["offset_mapping"][i] = [o if sids[k] == 1 else None for k, o in enumerate(tok["offset_mapping"][i])]
#         return tok
# 
#     eval_features = data.map(prep, batched=True, remove_columns=data.column_names)
#     eval_features_model = eval_features.remove_columns(["example_id", "offset_mapping"])
# 
#     trainer = Trainer(
#         model=model,
#         args=TrainingArguments(
#             output_dir="tmp_eval_squadv2_threshold",
#             per_device_eval_batch_size=args.batch_size,
#             dataloader_num_workers=args.num_workers,
#             report_to="none",
#             remove_unused_columns=False,
#         ),
#         data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
#         tokenizer=tokenizer,
#     )
# 
#     preds, _, _ = trainer.predict(eval_features_model)
#     start_logits, end_logits = preds
# 
