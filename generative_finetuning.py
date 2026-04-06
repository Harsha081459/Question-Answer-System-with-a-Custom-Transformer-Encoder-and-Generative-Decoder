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


def load_gen_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0)), float(payload.get("best_metric", -1e9))


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    grad_accum_steps,
    max_grad_norm,
    label_smoothing,
    global_step,
    amp_dtype,
    amp_enabled,
):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(pbar, start=1):
        enc_ids = batch["encoder_input_ids"].to(device, non_blocking=True)
        enc_ttype = batch["encoder_token_type_ids"].to(device, non_blocking=True)
        enc_mask = batch["encoder_attention_mask"].to(device, non_blocking=True)
        dec_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        dec_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=dec_ids,
                decoder_attention_mask=dec_mask,
                labels=labels,
                label_smoothing=label_smoothing,
            )
            loss = out["loss"] / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.item() * grad_accum_steps

        if i % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        pbar.set_postfix(loss=f"{running_loss / max(1, i):.4f}", step=global_step)

    return global_step, running_loss / max(1, len(train_loader))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", default="checkpoints_pretrain_base_seq256/step_20000")
    p.add_argument("--pretrain_ckpt", default="checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt")
    p.add_argument("--init_from_checkpoint", default="")
    p.add_argument("--output_dir", default="checkpoints_generative_qa")
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=48)
    p.add_argument("--train_batch_size", type=int, default=6)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--resume_path", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_warmup_epochs", type=int, default=2)
    p.add_argument("--unfreeze_top_layers", type=int, default=4)
    p.add_argument("--no_squad_v2", action="store_true")


# 
#     return global_step, running_loss / max(1, len(train_loader))
# 
# 
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--tokenizer_path", default="checkpoints_pretrain_base_seq256/step_20000")
#     p.add_argument("--pretrain_ckpt", default="checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt")
#     p.add_argument("--init_from_checkpoint", default="")
#     p.add_argument("--output_dir", default="checkpoints_generative_qa")
#     p.add_argument("--max_input_len", type=int, default=256)
#     p.add_argument("--max_target_len", type=int, default=48)
#     p.add_argument("--train_batch_size", type=int, default=6)
#     p.add_argument("--eval_batch_size", type=int, default=8)
#     p.add_argument("--grad_accum", type=int, default=2)
#     p.add_argument("--num_workers", type=int, default=4)
#     p.add_argument("--epochs", type=int, default=6)
#     p.add_argument("--lr", type=float, default=3e-4)
#     p.add_argument("--encoder_lr", type=float, default=8e-5)
#     p.add_argument("--weight_decay", type=float, default=0.01)
#     p.add_argument("--warmup_ratio", type=float, default=0.06)
#     p.add_argument("--max_grad_norm", type=float, default=1.0)
#     p.add_argument("--label_smoothing", type=float, default=0.05)
#     p.add_argument("--eval_every_epochs", type=int, default=1)
#     p.add_argument("--save_every_epochs", type=int, default=1)
#     p.add_argument("--resume_path", default="")
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--freeze_warmup_epochs", type=int, default=2)
#     p.add_argument("--unfreeze_top_layers", type=int, default=4)
#     p.add_argument("--no_squad_v2", action="store_true")
