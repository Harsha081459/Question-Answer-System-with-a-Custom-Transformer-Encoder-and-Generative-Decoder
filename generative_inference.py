import argparse
import json

import torch
from transformers import AutoTokenizer

from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True, help="Path to best.pt or latest.pt")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--instruction_prefix", default="")
    p.add_argument("--decoder_variant", choices=["standard", "hybrid"], default="standard")
    p.add_argument("--enable_no_answer_gate", action="store_true")
    p.add_argument("--no_answer_text", default="The context does not contain the answer.")
    p.add_argument("--no_answer_threshold", type=float, default=0.0)
    return p.parse_args()


def decode_generated_ids(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
    text_ids = []
    for t in out_ids:
        if t in {bos, pad}:
            continue
        if t == eos:
            break
        text_ids.append(t)
    return tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def build_target_ids(tokenizer, text: str, bos: int, eos: int, max_new_tokens: int, device: str) -> torch.Tensor:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_new_tokens - 1),
    )["input_ids"]
    seq = [bos] + ids + [eos]
    return torch.tensor([seq], dtype=torch.long, device=device)


def main():
    args = parse_args()
    payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    enc_cfg = ModelConfig(**payload["encoder_config"])
    dec_cfg = DecoderConfig(**payload["decoder_config"])
    if args.decoder_variant == "hybrid":
        model = GenerativeQAModelHybrid(enc_cfg, dec_cfg)
    else:
        model = StandardGenerativeQAModel(enc_cfg, dec_cfg)
    model.load_state_dict(payload["model"], strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token

    if args.instruction_prefix:
        inp = f"{args.instruction_prefix.strip()} question: {args.question} context: {args.context}"
    else:
        inp = f"question: {args.question} context: {args.context}"
    enc = tok(
        [inp],
        truncation=True,
        max_length=min(args.max_input_len, enc_cfg.max_position_embeddings),
        return_tensors="pt",


#     seq = [bos] + ids + [eos]
#     return torch.tensor([seq], dtype=torch.long, device=device)
# 
# 
# def main():
#     args = parse_args()
#     payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
#     enc_cfg = ModelConfig(**payload["encoder_config"])
#     dec_cfg = DecoderConfig(**payload["decoder_config"])
#     if args.decoder_variant == "hybrid":
#         model = GenerativeQAModelHybrid(enc_cfg, dec_cfg)
#     else:
#         model = StandardGenerativeQAModel(enc_cfg, dec_cfg)
#     model.load_state_dict(payload["model"], strict=True)
# 
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device).eval()
#     tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.sep_token
# 
#     if args.instruction_prefix:
#         inp = f"{args.instruction_prefix.strip()} question: {args.question} context: {args.context}"
#     else:
#         inp = f"question: {args.question} context: {args.context}"
#     enc = tok(
#         [inp],
#         truncation=True,
#         max_length=min(args.max_input_len, enc_cfg.max_position_embeddings),
#         return_tensors="pt",
