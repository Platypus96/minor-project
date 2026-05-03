"""
DeepSC Model Architecture — standalone copy for local inference.

Copied from https://github.com/13274086/DeepSC so we don't need to clone
the full repo locally.  Only the model definitions + inference helpers.

Architecture:
  Encoder(Embedding → PositionalEncoding → N × EncoderLayer)
  ChannelEncoder(Linear → ReLU → Linear)          [d_model → 256 → 16]
  ChannelDecoder(Linear → ReLU → Linear → ReLU → Linear + LayerNorm)
  Decoder(Embedding → PositionalEncoding → N × DecoderLayer)
  Dense(Linear)                                     [d_model → vocab]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------ #
#  Building blocks                                                     #
# ------------------------------------------------------------------ #

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attn = self._attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.dropout(self.dense(x))

    @staticmethod
    def _attention(query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores += mask * -1e9
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        src_output = self.src_mha(x, memory, memory, trg_padding_mask)
        x = self.layernorm2(x + src_output)
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, src_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.dec_layers:
            x = layer(x, memory, look_ahead_mask, trg_padding_mask)
        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        return self.layernorm(x1 + x5)


# ------------------------------------------------------------------ #
#  Full DeepSC model                                                   #
# ------------------------------------------------------------------ #

class DeepSC(nn.Module):
    """
    Deep Semantic Communication model.

    Architecture:
      text → Encoder → ChannelEncoder → [channel] → ChannelDecoder → Decoder → Dense → text
    """

    def __init__(self, num_layers, src_vocab_size, trg_vocab_size,
                 src_max_len, trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
        )
        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)
        self.dense = nn.Linear(d_model, trg_vocab_size)


# ------------------------------------------------------------------ #
#  Inference helpers (from DeepSC utils.py)                            #
# ------------------------------------------------------------------ #

def power_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize signal to unit power (as done during training)."""
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x


def subsequent_mask(size: int) -> torch.Tensor:
    """Upper-triangular mask for autoregressive decoding."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask)
