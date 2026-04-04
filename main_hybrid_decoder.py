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
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm masked self-attention.
        x_ln = self.norm1(x)
        self_out, _ = self.self_attn(
            query=x_ln,
            key=x_ln,
            value=x_ln,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(self_out)

        # Pre-norm cross-attention (decoder -> encoder memory).
        x_ln = self.norm2(x)
        cross_out, _ = self.cross_attn(
            query=x_ln,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(cross_out)

        # Pre-norm FFN.
        x_ln = self.norm3(x)
        ffn = self.ffn_fc2(F.gelu(self.ffn_fc1(x_ln)))
        x = x + self.dropout(ffn)
        return x


class HybridDecoderStack(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.layers = nn.ModuleList([HybridDecoderBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x


class GenerativeQAModelHybrid(nn.Module):
    """Main generative QA model using the custom hybrid decoder stack."""

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
        self.decoder = HybridDecoderStack(decoder_cfg)
        self.final_ln = nn.LayerNorm(decoder_cfg.hidden_size, eps=decoder_cfg.layer_norm_eps)
        self.lm_head = nn.Linear(decoder_cfg.hidden_size, decoder_cfg.vocab_size, bias=False)
        self.lm_head.weight = self.decoder_embeddings.token_embeddings.weight

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)



#                 tgt_attn_mask=tgt_attn_mask,
#                 tgt_key_padding_mask=tgt_key_padding_mask,
#                 memory_key_padding_mask=memory_key_padding_mask,
#             )
#         return x
# 
# 
# class GenerativeQAModelHybrid(nn.Module):
#     """Main generative QA model using the custom hybrid decoder stack."""
# 
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
#         self.decoder = HybridDecoderStack(decoder_cfg)
#         self.final_ln = nn.LayerNorm(decoder_cfg.hidden_size, eps=decoder_cfg.layer_norm_eps)
#         self.lm_head = nn.Linear(decoder_cfg.hidden_size, decoder_cfg.vocab_size, bias=False)
#         self.lm_head.weight = self.decoder_embeddings.token_embeddings.weight
# 
#     def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
#         return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
# 
