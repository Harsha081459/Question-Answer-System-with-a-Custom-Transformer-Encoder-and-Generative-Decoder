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



#         probs = self.attn_dropout(probs)
#         ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
#         out = self.out_proj(ctx)
#         out = self.proj_dropout(out)
#         return out
# 
# 
# class FeedForward(nn.Module):
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
#         self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
#         self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
# 
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
# 
# 
# class TransformerBlock(nn.Module):
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
#         self.attn = MultiHeadSelfAttention(cfg)
#         self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
#         self.ffn = FeedForward(cfg)
# 
