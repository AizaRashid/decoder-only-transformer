"""
model.py

A decoder-only transformer built from the same core ideas as the classic
"Attention Is All You Need" decoder stack -- embeddings, sinusoidal position
encoding, causal self-attention, a feed-forward block, residual connections,
layer norm -- but written as our own implementation:

  - attention is MULTI-head (several attention "lenses" running in parallel,
    concatenated and mixed back down), not a single head
  - each attention block is followed by a position-wise feed-forward network
  - blocks are STACKED (n_layers of them), not just one
  - attention masking handles both the causal (no peeking ahead) constraint
    and padding (so batches of different-length examples don't leak into
    each other's loss or attention)

None of the code below is copied from any tutorial; variable names, control
flow, and comments reflect our own reasoning about why each piece exists.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    """
    Adds a fixed (non-learned) signal to each token embedding that encodes
    its position in the sequence, using alternating sine/cosine waves of
    increasing wavelength. We precompute the table once in __init__ (it
    never changes) and just slice it in forward().
    """

    def __init__(self, d_model, max_len=64):
        super().__init__()
        position_table = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # One frequency per pair of (sin, cos) columns; frequency shrinks as
        # the column index grows, which is what gives later dimensions a
        # longer "wavelength" and lets far-apart positions stay distinguishable.
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        position_table[:, 0::2] = torch.sin(positions * frequencies)
        position_table[:, 1::2] = torch.cos(positions * frequencies)
        # register_buffer -> moves with the model (e.g. to GPU) but isn't a
        # trainable parameter, since these values are fixed by definition.
        self.register_buffer("position_table", position_table)

    def forward(self, token_embeddings):
        seq_len = token_embeddings.size(1)
        return token_embeddings + self.position_table[:seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    """
    Splits d_model into n_heads independent slices, runs scaled dot-product
    attention separately in each slice, then concatenates and projects back
    to d_model. Running several smaller attention "views" in parallel (as
    opposed to one big one) lets different heads specialize -- e.g. one head
    might learn to track "what entity is this question about" while another
    tracks "has the question word already appeared".
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (batch, seq, d_model) -> (batch, n_heads, seq, head_dim)
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, attn_mask):
        """
        attn_mask: bool tensor, shape (batch, 1, seq, seq), True where a
        position is ALLOWED to attend (combines causal + padding rules).
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (batch, n_heads, seq, head_dim)
        batch, _, seq, _ = context.shape
        context = context.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
        return self.out_proj(context), attn_weights


class FeedForward(nn.Module):
    """Position-wise MLP applied identically to every token: expand, activate, contract."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    One transformer block: causal self-attention with a residual connection
    and layer norm, followed by a feed-forward network with its own residual
    connection and layer norm. Residuals let gradients skip straight through
    the block, which is what makes stacking many blocks trainable at all.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        attn_out, attn_weights = self.attention(x, attn_mask)
        x = self.attn_norm(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.ff_norm(x + self.dropout(ff_out))
        return x, attn_weights


class DecoderOnlyTransformer(nn.Module):
    """
    Full model: token embedding -> position encoding -> N stacked decoder
    blocks -> final layer norm -> linear projection back to vocab size.

    Kept as a plain nn.Module (no training-framework base class) so the
    training loop in train.py is explicit and easy to read line-by-line --
    useful for a portfolio piece meant to show understanding of what's
    actually happening during optimization, not just that a framework
    handles it.
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128,
                 max_len=32, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_encoding = SinusoidalPositionEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: the output projection reuses the embedding matrix.
        # The intuition is that "which token is this" and "which token comes
        # next" are two ends of the same association, so sharing weights
        # halves the parameters spent on the vocab dimension and tends to
        # regularize the model on small datasets like this one.
        self.output_proj.weight = self.token_embedding.weight

    def build_attention_mask(self, token_ids):
        """
        Combines two rules into one boolean mask of allowed attention:
          - causal: position i may only look at positions <= i
          - padding: no position may attend to a <pad> token
        Shape: (batch, 1, seq, seq), broadcast over heads.
        """
        batch, seq = token_ids.shape
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=token_ids.device))
        not_pad = (token_ids != self.pad_id).unsqueeze(1)  # (batch, 1, seq): key positions
        mask = causal.unsqueeze(0) & not_pad  # (batch, seq, seq)
        return mask.unsqueeze(1)  # (batch, 1, seq, seq)

    def forward(self, token_ids, return_attention=False):
        mask = self.build_attention_mask(token_ids)
        x = self.token_embedding(token_ids)
        x = self.position_encoding(x)
        x = self.embed_dropout(x)

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            if return_attention:
                all_attn_weights.append(attn_weights)

        x = self.final_norm(x)
        logits = self.output_proj(x)

        if return_attention:
            return logits, all_attn_weights
        return logits
