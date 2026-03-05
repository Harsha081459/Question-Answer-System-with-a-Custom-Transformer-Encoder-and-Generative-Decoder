import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, interleave_datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


@dataclass
class ModelConfig:
    vocab_size: int
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


SIZE_PRESETS = {
    "base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
}


class BertEmbeddings(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(pos_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.attn_dropout = nn.Dropout(cfg.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, attention_mask):
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
        scores = scores.masked_fill(mask == 0, -1e4)
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        out = self.out_proj(ctx)
        out = self.proj_dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BertEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_ln(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, cfg: ModelConfig, tied_embedding: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.decoder.bias = self.bias
        self.decoder.weight = tied_embedding.weight

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class BertForMLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = BertEncoder(cfg)
        self.mlm = MLMHead(cfg, self.encoder.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        x = self.encoder(input_ids, token_type_ids, attention_mask)
        logits = self.mlm(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss


class StreamingTextDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=128, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        # Prefer parquet-backed datasets to avoid old script-based dataset failures.
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        web = load_dataset("allenai/c4", "en", split="train", streaming=True)
        self.stream = interleave_datasets([wiki, web], probabilities=[0.7, 0.3], seed=seed)
        self.stream = self.stream.shuffle(seed=seed, buffer_size=20000)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        token_buffer = []
        _ = random.Random(self.seed + worker_id)
        # Compatibility: some datasets versions expose "shard", others "to_iterable_dataset".
        if num_workers > 1 and hasattr(self.stream, "shard"):
            stream = self.stream.shard(num_shards=num_workers, index=worker_id)
        else:
            stream = self.stream
        for ex in stream:
            if num_workers > 1:
                # Manual worker partition fallback when .shard() is unavailable.
                ex_idx = getattr(self, "_worker_example_idx", 0)
                self._worker_example_idx = ex_idx + 1
                if (ex_idx % num_workers) != worker_id:
                    continue
            text = ex.get("text", "")
            if not text or len(text) < 20:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if len(ids) == 0:
                continue
            token_buffer.extend(ids)
            while len(token_buffer) >= (self.max_len - 2):
                chunk = token_buffer[: self.max_len - 2]
                token_buffer = token_buffer[self.max_len - 2 :]
                input_ids = [self.cls_id] + chunk + [self.sep_id]
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }


def collate_mlm(batch, tokenizer, mlm_prob=0.15):
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, x in enumerate(batch):


#             text = ex.get("text", "")
#             if not text or len(text) < 20:
#                 continue
#             ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
#             if len(ids) == 0:
#                 continue
#             token_buffer.extend(ids)
#             while len(token_buffer) >= (self.max_len - 2):
#                 chunk = token_buffer[: self.max_len - 2]
#                 token_buffer = token_buffer[self.max_len - 2 :]
#                 input_ids = [self.cls_id] + chunk + [self.sep_id]
#                 attention_mask = [1] * len(input_ids)
#                 token_type_ids = [0] * len(input_ids)
#                 yield {
#                     "input_ids": input_ids,
#                     "attention_mask": attention_mask,
#                     "token_type_ids": token_type_ids,
#                 }
# 
# 
# def collate_mlm(batch, tokenizer, mlm_prob=0.15):
#     pad_id = tokenizer.pad_token_id
#     mask_id = tokenizer.mask_token_id
#     vocab_size = tokenizer.vocab_size
#     max_len = max(len(x["input_ids"]) for x in batch)
#     bsz = len(batch)
#     input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
#     attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
#     token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
#     for i, x in enumerate(batch):
