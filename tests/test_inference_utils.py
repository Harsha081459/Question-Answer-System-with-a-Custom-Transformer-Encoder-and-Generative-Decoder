from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import BertTokenizerFast

from generative_inference import build_target_ids, decode_generated_ids

VOCAB_FILE = Path(__file__).resolve().parent / "fixtures" / "vocab.txt"


@pytest.fixture(scope="module")
def tokenizer():
    vocab = {
        line.strip(): idx
        for idx, line in enumerate(VOCAB_FILE.read_text(encoding="utf-8").splitlines())
        if line.strip()
    }
    backend = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.normalizer = BertNormalizer(lowercase=True)
    backend.pre_tokenizer = BertPreTokenizer()
    backend.decoder = WordPieceDecoder(prefix="##")
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return BertTokenizerFast(tokenizer_object=backend)


def test_build_target_ids_wraps_bos_eos(tokenizer):
    ids = build_target_ids(
        tokenizer=tokenizer,
        text="the answer",
        bos=tokenizer.cls_token_id,
        eos=tokenizer.sep_token_id,
        max_new_tokens=8,
        device="cpu",
    )
    assert ids.dtype == torch.long
    assert ids.shape[0] == 1
    assert ids[0, 0].item() == tokenizer.cls_token_id
    assert ids[0, -1].item() == tokenizer.sep_token_id
    assert ids.shape[1] == 2 + len(tokenizer("the answer", add_special_tokens=False)["input_ids"])


def test_build_target_ids_truncates(tokenizer):
    text = "the answer is the answer is the answer"
    raw_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    ids = build_target_ids(
        tokenizer=tokenizer,
        text=text,
        bos=tokenizer.cls_token_id,
        eos=tokenizer.sep_token_id,
        max_new_tokens=4,
        device="cpu",
    )
    assert ids.shape[1] == 2 + min(raw_len, 3)


def test_decode_generated_ids_strips_special_tokens(tokenizer):
    answer_ids = tokenizer("the answer", add_special_tokens=False)["input_ids"]
    out_ids = [tokenizer.cls_token_id] + answer_ids + [tokenizer.sep_token_id, tokenizer.pad_token_id]
    assert decode_generated_ids(
        tokenizer,
        out_ids,
        bos=tokenizer.cls_token_id,
        eos=tokenizer.sep_token_id,
        pad=tokenizer.pad_token_id,
    ) == "the answer"


def test_decode_generated_ids_stops_at_eos(tokenizer):
    answer_ids = tokenizer("the answer", add_special_tokens=False)["input_ids"]
    trailing_ids = tokenizer("python world", add_special_tokens=False)["input_ids"]
    out_ids = answer_ids + [tokenizer.sep_token_id] + trailing_ids
    assert decode_generated_ids(
        tokenizer,
        out_ids,
        bos=tokenizer.cls_token_id,
        eos=tokenizer.sep_token_id,
        pad=tokenizer.pad_token_id,
    ) == "the answer"
