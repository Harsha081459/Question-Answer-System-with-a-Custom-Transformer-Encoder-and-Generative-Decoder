import argparse
import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from generative_data import GenQADataConfig, NO_ANSWER_TEXT, NO_ANSWER_SENTENCE, build_dataloaders
from generative_evaluation import evaluate_model
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def load_encoder_from_pretrain(model, pretrain_ckpt_path: str):
    payload = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model"]
    enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=True)
    return int(payload.get("step", 0))


def _remap_decoder_state_keys(state: dict, decoder_variant: str):
    if decoder_variant not in {"standard", "hybrid"}:
        raise ValueError("decoder_variant must be one of: standard, hybrid")

    remapped = {}
    num_changed = 0
    for k, v in state.items():
        nk = k
        if decoder_variant == "hybrid":
            nk = nk.replace(".multihead_attn.", ".cross_attn.")
            nk = nk.replace(".linear1.", ".ffn_fc1.")
            nk = nk.replace(".linear2.", ".ffn_fc2.")
        else:
            nk = nk.replace(".cross_attn.", ".multihead_attn.")
            nk = nk.replace(".ffn_fc1.", ".linear1.")
            nk = nk.replace(".ffn_fc2.", ".linear2.")
        if nk != k:
            num_changed += 1
        remapped[nk] = v
    return remapped, num_changed


def load_model_from_gen_checkpoint(model, checkpoint_path: str, decoder_variant: str = "standard"):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_raw = payload["model"] if "model" in payload else payload
    state, changed = _remap_decoder_state_keys(state_raw, decoder_variant=decoder_variant)
    if changed:
        print(f"[init] Remapped {changed} decoder tensor keys for decoder_variant={decoder_variant}")
    current_state = model.state_dict()

    # Allow target-length continuation by resizing decoder positional embeddings.
    pos_key = "decoder_embeddings.position_embeddings.weight"
    if pos_key in state and pos_key in current_state:
        ckpt_pos = state[pos_key]
        cur_pos = current_state[pos_key]
        if ckpt_pos.shape != cur_pos.shape:
            resized = cur_pos.clone()
            copy_len = min(ckpt_pos.size(0), cur_pos.size(0))
            resized[:copy_len] = ckpt_pos[:copy_len]
            state[pos_key] = resized
            print(
                f"[init] Resized decoder position embeddings "
                f"{tuple(ckpt_pos.shape)} -> {tuple(cur_pos.shape)} "
                f"(copied first {copy_len} positions)"
            )

    model.load_state_dict(state, strict=True)
    return int(payload.get("step", 0)), float(payload.get("best_metric", -1e9))


def save_gen_checkpoint(path: str, model, optimizer, scheduler, scaler, step, epoch, best_metric, enc_cfg, dec_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "best_metric": best_metric,
            "encoder_config": enc_cfg.__dict__,
            "decoder_config": dec_cfg.to_dict(),
        },
        path,
    )


#             resized = cur_pos.clone()
#             copy_len = min(ckpt_pos.size(0), cur_pos.size(0))
#             resized[:copy_len] = ckpt_pos[:copy_len]
#             state[pos_key] = resized
#             print(
#                 f"[init] Resized decoder position embeddings "
#                 f"{tuple(ckpt_pos.shape)} -> {tuple(cur_pos.shape)} "
#                 f"(copied first {copy_len} positions)"
#             )
# 
#     model.load_state_dict(state, strict=True)
#     return int(payload.get("step", 0)), float(payload.get("best_metric", -1e9))
# 
# 
# def save_gen_checkpoint(path: str, model, optimizer, scheduler, scaler, step, epoch, best_metric, enc_cfg, dec_cfg):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(
#         {
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict(),
#             "scaler": scaler.state_dict() if scaler is not None else None,
#             "step": step,
#             "epoch": epoch,
#             "best_metric": best_metric,
#             "encoder_config": enc_cfg.__dict__,
#             "decoder_config": dec_cfg.to_dict(),
#         },
#         path,
#     )
