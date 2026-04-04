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
            dropout=decoder_cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_cfg.num_layers)
        self.final_ln = nn.LayerNorm(decoder_cfg.hidden_size, eps=decoder_cfg.layer_norm_eps)
        self.lm_head = nn.Linear(decoder_cfg.hidden_size, decoder_cfg.vocab_size, bias=False)
        self.lm_head.weight = self.decoder_embeddings.token_embeddings.weight

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # Use a boolean mask to match key padding mask dtype in TransformerDecoder.
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.encoder(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
        return self.enc_to_dec(enc)

    def decode(
        self,
        memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        dec_inp = self.decoder_embeddings(decoder_input_ids)
        tgt_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)
        tgt_key_padding_mask = decoder_attention_mask == 0
        memory_key_padding_mask = encoder_attention_mask == 0
        dec_out = self.decoder(
            tgt=dec_inp,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        dec_out = self.final_ln(dec_out)
        return self.lm_head(dec_out)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> dict:
        memory = self.encode(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
        logits = self.decode(memory, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
        out = {"logits": logits}
        if labels is not None:


#         decoder_attention_mask: torch.Tensor,
#     ) -> torch.Tensor:
#         dec_inp = self.decoder_embeddings(decoder_input_ids)
#         tgt_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)
#         tgt_key_padding_mask = decoder_attention_mask == 0
#         memory_key_padding_mask = encoder_attention_mask == 0
#         dec_out = self.decoder(
#             tgt=dec_inp,
#             memory=memory,
#             tgt_mask=tgt_mask,
#             tgt_key_padding_mask=tgt_key_padding_mask,
#             memory_key_padding_mask=memory_key_padding_mask,
#         )
#         dec_out = self.final_ln(dec_out)
#         return self.lm_head(dec_out)
# 
#     def forward(
#         self,
#         encoder_input_ids: torch.Tensor,
#         encoder_token_type_ids: torch.Tensor,
#         encoder_attention_mask: torch.Tensor,
#         decoder_input_ids: torch.Tensor,
#         decoder_attention_mask: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#         label_smoothing: float = 0.0,
#     ) -> dict:
#         memory = self.encode(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
#         logits = self.decode(memory, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
#         out = {"logits": logits}
#         if labels is not None:
