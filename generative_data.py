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
        answerable_ds = train_ds.filter(_has_answer)
        train_ds = concatenate_datasets([train_ds] + [answerable_ds] * (answerable_repeat - 1))
    if include_squad_v2 and no_answer_repeat > 1:
        no_answer_ds = train_ds.filter(_is_no_answer)
        train_ds = concatenate_datasets([train_ds] + [no_answer_ds] * (no_answer_repeat - 1))
    return train_ds, val_ds


def _slice_sentence_around_index(context: str, char_index: int) -> str:
    if not context:
        return ""
    if char_index < 0:
        char_index = 0
    if char_index >= len(context):
        char_index = len(context) - 1

    left_candidates = [
        context.rfind(".", 0, char_index),
        context.rfind("?", 0, char_index),
        context.rfind("!", 0, char_index),
    ]
    left = max(left_candidates)
    start = 0 if left == -1 else left + 1

    right_positions = []
    for ch in (".", "?", "!"):
        pos = context.find(ch, char_index)
        if pos != -1:
            right_positions.append(pos + 1)
    end = min(right_positions) if right_positions else len(context)
    return context[start:end].strip()


def _find_answer_sentence(context: str, answers: dict) -> str:
    if not context:
        return ""
    answer_texts = answers.get("text", [])
    if not answer_texts:
        return ""

    answer_text = answer_texts[0].strip()
    answer_starts = answers.get("answer_start", [])
    answer_start = answer_starts[0] if answer_starts else -1
    if answer_start is None or answer_start < 0:
        answer_start = context.lower().find(answer_text.lower())

    if answer_start is not None and answer_start >= 0:
        sent = _slice_sentence_around_index(context, int(answer_start))
        if sent:
            return sent

    # Fallback: try direct sentence search via regex split.
    for sent in re.split(r"(?<=[.!?])\s+", context):
        s = sent.strip()
        if s and answer_text.lower() in s.lower():
            return s

    return answer_text


def add_targets(example, target_style: str = "span", no_answer_target_text: str = NO_ANSWER_TEXT):
    answers = example["answers"]
    if len(answers["text"]) == 0:
        target = no_answer_target_text.strip()
    else:
        if target_style == "sentence":
            target = _find_answer_sentence(example.get("context", ""), answers)
        else:
            target = answers["text"][0].strip()
    return {"target_text": target}




#     answer_start = answer_starts[0] if answer_starts else -1
#     if answer_start is None or answer_start < 0:
#         answer_start = context.lower().find(answer_text.lower())
# 
#     if answer_start is not None and answer_start >= 0:
#         sent = _slice_sentence_around_index(context, int(answer_start))
#         if sent:
#             return sent
# 
#     # Fallback: try direct sentence search via regex split.
#     for sent in re.split(r"(?<=[.!?])\s+", context):
#         s = sent.strip()
#         if s and answer_text.lower() in s.lower():
#             return s
# 
#     return answer_text
# 
# 
# def add_targets(example, target_style: str = "span", no_answer_target_text: str = NO_ANSWER_TEXT):
#     answers = example["answers"]
#     if len(answers["text"]) == 0:
#         target = no_answer_target_text.strip()
#     else:
#         if target_style == "sentence":
#             target = _find_answer_sentence(example.get("context", ""), answers)
#         else:
#             target = answers["text"][0].strip()
#     return {"target_text": target}
# 
# 
