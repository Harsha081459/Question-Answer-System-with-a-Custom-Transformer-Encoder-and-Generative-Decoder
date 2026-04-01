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


class GenerativeQAModel(nn.Module):
    def __init__(self, encoder_cfg: ModelConfig, decoder_cfg: DecoderConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.encoder = BertEncoder(encoder_cfg)

        if encoder_cfg.hidden_size != decoder_cfg.hidden_size:
            self.enc_to_dec = nn.Linear(encoder_cfg.hidden_size, decoder_cfg.hidden_size)
        else:
            self.enc_to_dec = nn.Identity()

        self.decoder_embeddings = DecoderEmbeddings(decoder_cfg)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_cfg.hidden_size,
            nhead=decoder_cfg.num_attention_heads,
            dim_feedforward=decoder_cfg.intermediate_size,


#         self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
#         self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
#         self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
#         self.dropout = nn.Dropout(cfg.dropout)
# 
#     def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
#         bsz, seq_len = input_ids.shape
#         pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
#         x = self.token_embeddings(input_ids) + self.position_embeddings(pos)
#         x = self.layer_norm(x)
#         return self.dropout(x)
# 
# 
# class GenerativeQAModel(nn.Module):
#     def __init__(self, encoder_cfg: ModelConfig, decoder_cfg: DecoderConfig):
#         super().__init__()
#         self.encoder_cfg = encoder_cfg
#         self.decoder_cfg = decoder_cfg
#         self.encoder = BertEncoder(encoder_cfg)
# 
#         if encoder_cfg.hidden_size != decoder_cfg.hidden_size:
#             self.enc_to_dec = nn.Linear(encoder_cfg.hidden_size, decoder_cfg.hidden_size)
#         else:
#             self.enc_to_dec = nn.Identity()
# 
#         self.decoder_embeddings = DecoderEmbeddings(decoder_cfg)
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=decoder_cfg.hidden_size,
#             nhead=decoder_cfg.num_attention_heads,
#             dim_feedforward=decoder_cfg.intermediate_size,
