from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlm_pretraining import BertEncoder, ModelConfig


@dataclass
class DecoderConfig:
    vocab_size: int
    hidden_size: int = 512
    num_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 256
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12

    def to_dict(self) -> dict:
        return asdict(self)


class DecoderEmbeddings(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_embeddings(input_ids) + self.position_embeddings(pos)
        x = self.layer_norm(x)
        return self.dropout(x)


class HybridDecoderBlock(nn.Module):
    """
    Middle-path decoder block:
    - Uses low-level nn.MultiheadAttention + Linear + LayerNorm
    - Avoids nn.TransformerDecoder / nn.TransformerDecoderLayer
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.ffn_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.ffn_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm3 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,


#     def __init__(self, cfg: DecoderConfig):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=cfg.hidden_size,
#             num_heads=cfg.num_attention_heads,
#             dropout=cfg.dropout,
#             batch_first=True,
#         )
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=cfg.hidden_size,
#             num_heads=cfg.num_attention_heads,
#             dropout=cfg.dropout,
#             batch_first=True,
#         )
# 
#         self.ffn_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
#         self.ffn_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
# 
#         self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
#         self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
#         self.norm3 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
# 
#         self.dropout = nn.Dropout(cfg.dropout)
# 
#     def forward(
#         self,
#         x: torch.Tensor,
#         memory: torch.Tensor,
#         tgt_attn_mask: torch.Tensor,
#         tgt_key_padding_mask: torch.Tensor,
