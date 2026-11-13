"""MafLM: a small ternary language model designed for the model-as-firmware compiler (Phase 3).

Two twins of the same size that differ only in the feedforward layer:
  ffn='dense'  x -> ReLU(W1 x) -> W2           (d -> hidden -> d)
  ffn='tree'   fast feedforward tree (Belcak & Wattenhofer 2023): depth-D binary tree of node neurons picks one of
               2^D leaves, each a small d -> leaf -> d block. Same total weights as the dense twin when
               2^D * leaf == hidden, but a token only reads D node rows plus one leaf.

Choices made for integer, bare-metal inference (see notes/lm.md):
  - byte vocabulary (256), no tokenizer tables;
  - ternary weights {-1, 0, +1} with one absmean scale per matrix (BitNet b1.58) for every linear layer;
    the embedding table is int8 (one row read per token, so its size barely matters for speed);
  - every linear layer normalizes its input per token (RMS, no learned gain) and quantizes it to int8 by absmax,
    so the board computes int8 x ternary -> int32 dot products;
  - attention with grouped key/value heads (smaller KV cache) and ALiBi positions (a per-head linear distance
    penalty: integer-friendly, and a rolling KV cache stays valid); q, k, v and the attention weights are
    fake-quantized (int8, uint8) the way the board will hold them;
  - ReLU everywhere, no biases.
Training uses straight-through estimators for all the rounding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 256
    d: int = 128
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    d_head: int = 32
    hidden: int = 640           # dense FFN width; for the tree, 2^depth * leaf
    ffn: str = "dense"          # 'dense' or 'tree'
    depth: int = 3              # tree depth (2^depth leaves)
    max_seq: int = 256          # training context (ALiBi has no position table, so this is not a hard limit)

    @property
    def leaf(self) -> int:
        return self.hidden // (2 ** self.depth)

    def to_dict(self) -> dict:
        return asdict(self)


def ste(x: Tensor, q: Tensor) -> Tensor:
    """Forward q, backward identity to x."""
    return x + (q - x).detach()


def quant_act(x: Tensor, bits: int = 8) -> Tensor:
    """Per-token (last dim) symmetric absmax quantization, returned dequantized, with STE."""
    qmax = 2 ** (bits - 1) - 1
    s = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-5) / qmax
    return ste(x, (x / s).round().clamp(-qmax - 1, qmax) * s)


def ternary(w: Tensor) -> tuple[Tensor, Tensor]:
    """BitNet b1.58 ternary codes and the absmean scale: w ~= codes * scale."""
    scale = w.detach().abs().mean().clamp_min(1e-5)
    return (w / scale).round().clamp(-1, 1), scale


class TLinear(nn.Module):
    """Ternary linear layer with per-token RMS normalization and int8 activations (no bias)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=d_in ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        x = x / x.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt()
        x = quant_act(x)
        codes, scale = ternary(self.weight)
        return F.linear(x, ste(self.weight, codes * scale))

    def codes(self) -> tuple[Tensor, float]:
        c, s = ternary(self.weight)
        return c.to(torch.int8), float(s)


def alibi_slopes(n: int) -> Tensor:
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / n) for i in range(n)])


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wq = TLinear(c.d, c.heads * c.d_head)
        self.wk = TLinear(c.d, c.kv_heads * c.d_head)
        self.wv = TLinear(c.d, c.kv_heads * c.d_head)
        self.wo = TLinear(c.heads * c.d_head, c.d)
        self.register_buffer("slopes", alibi_slopes(c.heads), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        c = self.c
        B, T, _ = x.shape
        q = quant_act(self.wq(x).view(B, T, c.heads, c.d_head)).transpose(1, 2)        # B H T dh
        k = quant_act(self.wk(x).view(B, T, c.kv_heads, c.d_head)).transpose(1, 2)
        v = quant_act(self.wv(x).view(B, T, c.kv_heads, c.d_head)).transpose(1, 2)
        rep = c.heads // c.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        pos = torch.arange(T, device=x.device)
        dist = (pos[:, None] - pos[None, :]).float()                                   # i - j
        bias = -self.slopes.to(x.device)[:, None, None] * dist                          # H T T
        bias = bias.masked_fill(dist < 0, float("-inf"))
        att = (q @ k.transpose(-1, -2)) * c.d_head ** -0.5 + bias
        p = att.float().softmax(-1)
        p = ste(p, (p * 255).round() / 255).to(v.dtype)                                 # uint8 weights on the board
        y = (p @ v).transpose(1, 2).reshape(B, T, c.heads * c.d_head)
        return self.wo(y)


class DenseFFN(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.w1 = TLinear(c.d, c.hidden)
        self.w2 = TLinear(c.hidden, c.d)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.relu(self.w1(x)))

    def aux_loss(self) -> Tensor | None:
        return None


class TreeFFN(nn.Module):
    """Fast feedforward tree. Node k (level by level, children 2k+1, 2k+2) is one ternary neuron; go right when
    its output >= 0. Training starts soft (the output is the mixture of all leaves weighted by the path
    probabilities), then switches to hard routing with a straight-through estimator to the nodes. A hardening loss
    (mean Bernoulli entropy of the decisions) ramps up during the soft phase, and a load-balancing loss
    (KL of the mean leaf usage over the batch from uniform) runs throughout: on MNIST the tree collapsed to one
    leaf without it (FINDINGS F25). Evaluation (no grad) is always hard."""

    def __init__(self, c: Config):
        super().__init__()
        self.depth, self.n_leaves, self.leaf = c.depth, 2 ** c.depth, c.leaf
        self.nodes = TLinear(c.d, self.n_leaves - 1)
        self.leaf_in = TLinear(c.d, self.n_leaves * self.leaf)       # rows j*leaf .. are leaf j
        self.leaf_out = nn.ModuleList(TLinear(self.leaf, c.d) for _ in range(self.n_leaves))
        # schedule, set by the trainer
        self.hard = False
        self.harden_w = 0.0
        self.balance_w = 1.0
        self._aux = None
        self.last_leaf = None

    def hard_leaf(self, logits: Tensor) -> Tensor:
        n = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
        for m in range(self.depth):
            node = (2 ** m - 1) + n
            go = (logits.gather(-1, node.unsqueeze(-1)).squeeze(-1) >= 0).long()
            n = 2 * n + go
        return n

    def forward(self, x: Tensor) -> Tensor:
        shp = x.shape[:-1]
        logits = self.nodes(x).float()                                   # ... (2^D - 1)
        p = torch.sigmoid(logits)
        probs = torch.ones(*shp, 1, device=x.device)
        for m in range(self.depth):
            pm = p[..., 2 ** m - 1: 2 ** (m + 1) - 1]
            probs = torch.stack([probs * (1 - pm), probs * pm], -1).reshape(*shp, 2 ** (m + 1))
        hard = self.hard or not torch.is_grad_enabled()
        idx = self.hard_leaf(logits)
        self.last_leaf = idx.detach()
        if torch.is_grad_enabled():
            pe = p.clamp(1e-6, 1 - 1e-6)
            ent = -(pe * pe.log() + (1 - pe) * (1 - pe).log()).mean()
            usage = probs.reshape(-1, self.n_leaves).mean(0)
            kl = (usage * (usage * self.n_leaves).clamp_min(1e-9).log()).sum()
            self._aux = self.balance_w * kl + (0.0 if hard else self.harden_w) * ent
        h = F.relu(self.leaf_in(x)).view(*shp, self.n_leaves, self.leaf)
        outs = torch.stack([self.leaf_out[j](h[..., j, :]) for j in range(self.n_leaves)], -2)   # ... L d
        if hard:
            w = F.one_hot(idx, self.n_leaves).to(probs.dtype)
            if torch.is_grad_enabled():
                w = w + probs - probs.detach()
        else:
            w = probs
        return (w.unsqueeze(-1).to(outs.dtype) * outs).sum(-2)

    def aux_loss(self) -> Tensor | None:
        return self._aux


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attn = Attention(c)
        self.ffn = TreeFFN(c) if c.ffn == "tree" else DenseFFN(c)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)          # each TLinear normalizes its own input (pre-norm)
        return x + self.ffn(x)


class MafLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Parameter(torch.randn(c.vocab, c.d))
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.unembed = TLinear(c.d, c.vocab)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0)))

    def embed_q(self) -> Tensor:
        """int8 embedding rows (per-row absmax), dequantized, with STE."""
        return quant_act(self.embed)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.embed_q()[ids]
        for b in self.blocks:
            x = b(x)
        return self.unembed(x) * self.logit_scale.exp()

    def aux_loss(self) -> Tensor:
        a = [b.ffn.aux_loss() for b in self.blocks]
        a = [t for t in a if t is not None]
        return torch.stack(a).mean() if a else torch.zeros((), device=self.embed.device)

    def trees(self) -> list[TreeFFN]:
        return [b.ffn for b in self.blocks if isinstance(b.ffn, TreeFFN)]

    def n_weights(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if n != "logit_scale")

    def weights_read_per_token(self) -> int:
        """Weights a single generated token reads (one embedding row; tree: the nodes plus one leaf)."""
        c = self.c
        n = c.d + self.unembed.weight.numel()
        for b in self.blocks:
            n += sum(m.weight.numel() for m in (b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo))
            if isinstance(b.ffn, TreeFFN):
                n += b.ffn.nodes.weight.numel() + 2 * c.d * c.leaf
            else:
                n += b.ffn.w1.weight.numel() + b.ffn.w2.weight.numel()
        return n
