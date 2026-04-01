import re
import string
from dataclasses import dataclass

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


NO_ANSWER_TEXT = "No answer in context."
NO_ANSWER_SENTENCE = "The context does not contain the answer."


def normalize_text(s: str) -> str:
    def remove_articles(text):
        return " ".join([w for w in text.split() if w not in {"a", "an", "the"}])

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


@dataclass
class GenQADataConfig:
    tokenizer_path: str
    max_input_len: int = 256
    max_target_len: int = 48
    include_squad_v2: bool = True
    answerable_repeat: int = 1
    no_answer_repeat: int = 1
    target_style: str = "span"  # "span" or "sentence"
    no_answer_target_text: str = NO_ANSWER_TEXT
    instruction_prefix: str = ""
    seed: int = 42


def build_tokenizer(tokenizer_path: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token
    return tok


def _has_answer(example) -> bool:
    return len(example["answers"]["text"]) > 0


def _is_no_answer(example) -> bool:
    return len(example["answers"]["text"]) == 0


def load_train_val(
    include_squad_v2: bool = True,
    answerable_repeat: int = 1,
    no_answer_repeat: int = 1,
):
    ds1 = load_dataset("squad")
    train = [ds1["train"]]
    val = [ds1["validation"]]
    if include_squad_v2:
        ds2 = load_dataset("squad_v2")
        train.append(ds2["train"])
        val.append(ds2["validation"])
    train_ds = concatenate_datasets(train)
    val_ds = concatenate_datasets(val)
    if include_squad_v2 and answerable_repeat > 1:


# def build_tokenizer(tokenizer_path: str):
#     tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.sep_token
#     return tok
# 
# 
# def _has_answer(example) -> bool:
#     return len(example["answers"]["text"]) > 0
# 
# 
# def _is_no_answer(example) -> bool:
#     return len(example["answers"]["text"]) == 0
# 
# 
# def load_train_val(
#     include_squad_v2: bool = True,
#     answerable_repeat: int = 1,
#     no_answer_repeat: int = 1,
# ):
#     ds1 = load_dataset("squad")
#     train = [ds1["train"]]
#     val = [ds1["validation"]]
#     if include_squad_v2:
#         ds2 = load_dataset("squad_v2")
#         train.append(ds2["train"])
#         val.append(ds2["validation"])
#     train_ds = concatenate_datasets(train)
#     val_ds = concatenate_datasets(val)
#     if include_squad_v2 and answerable_repeat > 1:
