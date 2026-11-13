"""Integer reference for MafLM: the exact arithmetic the board runs (golden model for the compiler and firmware).

Every value is an integer; the only floats are in export (computing the integer constants once) and in turning the
final integer logits into probabilities for the bits-per-byte evaluation. The rules (also in notes/lm.md):

Residual stream: int32, fixed point with F = 12 fractional bits.
Embedding:       x0 = (code[tok] * emul[tok] + 128) >> 8, codes int8, emul = round(rowmax / 127 * 2^(F+8)).

TLinear (ternary codes W, wmul = round(wscale * 2^(F+16) / 127)), input any integer vector x of length n:
  1. sh = max(0, bitlen(max|x|) - 15); xs = x >> sh            (inputs in 16 bits, so the sums below fit int64)
  2. m = max|xs|; if m == 0 the output is 0
  3. q = sign(xs) * ((254*|xs| + m) // (2m))                   int8 codes, absmax, round half away from zero
  4. acc = W . q                                               int32
  5. r = max(1, isqrt((sum xs^2 // n) << 14))                  rms(xs) * 2^7, floor (at least 1)
  6. G = (wmul * m * 2^7 + r // 2) // r                        per-vector multiplier, = wmul * m / rms
  7. y = (acc * G + 2^15) >> 16                                in residual units (2^-F)
  This is the float model's RMS-normalize -> absmax int8 -> ternary matmul, with the per-token scale folded into G.

Attention (H query heads, Hkv key/value heads, d_head dh, ALiBi slope 2^-(8(h+1)/H)):
  q, k, v = TLinear(x); each head slice is quantized to int8 codes with its own m (steps 1-3 on the slice, keeping m
  in residual units, i.e. m = max|slice| without the shift). The KV cache holds (codes, m) per token and head.
  dot  = qc . kc                                               int32
  a    = (dot * mk) >> 12                                      int64 product
  Cq   = (mq * KQ + 2^15) >> 16, KQ = round(2^(64-2F) / (127^2 sqrt(dh)))
  s    = (a * Cq + 2^23) >> 24  -  ((t - j) << (12 - 8(h+1)/H))  score in 2^-12 units, minus the ALiBi penalty
  z    = s - max_j s; u = (z * 5909) >> 12 (z log2 e, 2^-12 units); e = EXP2[(u & 4095) >> 4] >> (-(u >> 12))
         with EXP2[i] = round(2^15 * 2^(i/256)), and e = 0 when the shift is 31 or more
  p    = (255 * e + S // 2) // S, S = sum e                    uint8 attention weights
  w    = (p * mv) >> 12;  o = sum_j w_j * vc_j                 int32; o goes to Wo (TLinear is unit-free)
FFN dense: h = max(0, W1(x)); y = W2(h).  Tree: at node k go right when its acc >= 0 (exact sign of the float
logit); h = max(0, leaf_in rows of the leaf (x)); y = leaf_out[leaf](h).
Block: x += attn(x); x += ffn(x). Logits: unembed(x) in residual units; greedy picks the first maximum.

The reference runs batched on the GPU: matrix products of int8 codes are done in float32 (exact: every partial sum
is an integer below 2^24) or float64 (attention values), all other steps in int64.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import Config, MafLM, TLinear, TreeFFN  # noqa: E402

F = 12
EXP2 = np.array([round(2 ** 15 * 2 ** (i / 256)) for i in range(256)], dtype=np.int64)
LOG2E_Q12 = 5909


# ---------------------------------------------------------------------------------------------- export (floats -> ints)
def export(model: MafLM) -> dict:
    """Integer model: codes and multipliers. Keys are flat names, values numpy arrays or ints."""
    c = model.c
    out: dict = {"config": c.to_dict(), "F": F}
    e = model.embed.detach().float()
    rowmax = e.abs().amax(-1).clamp_min(1e-5)
    out["embed.codes"] = (e / (rowmax[:, None] / 127)).round().clamp(-128, 127).to(torch.int8).numpy()
    out["embed.mul"] = (rowmax / 127 * 2 ** (F + 8)).round().to(torch.int64).numpy()

    def lin(name: str, m: TLinear):
        codes, s = m.codes()
        out[name + ".codes"] = codes.numpy()
        out[name + ".wmul"] = int(round(s * 2 ** (F + 16) / 127))

    for i, b in enumerate(model.blocks):
        for nm in ("wq", "wk", "wv", "wo"):
            lin(f"b{i}.{nm}", getattr(b.attn, nm))
        if isinstance(b.ffn, TreeFFN):
            lin(f"b{i}.nodes", b.ffn.nodes)
            lin(f"b{i}.leaf_in", b.ffn.leaf_in)
            for j, lo in enumerate(b.ffn.leaf_out):
                lin(f"b{i}.leaf_out{j}", lo)
        else:
            lin(f"b{i}.w1", b.ffn.w1)
            lin(f"b{i}.w2", b.ffn.w2)
    lin("unembed", model.unembed)
    out["logit_scale"] = float(model.logit_scale.detach().exp())
    out["KQ"] = int(round(2 ** (64 - 2 * F) / (127 ** 2 * math.sqrt(c.d_head))))
    out["alibi_shift"] = [12 - 8 * (h + 1) // c.heads for h in range(c.heads)]
    assert all(8 * (h + 1) % c.heads == 0 for h in range(c.heads)), "ALiBi slopes must be powers of two"
    return out


def save_npz(im: dict, path: Path) -> None:
    """Flat .npz for the compiler: config as cfg.<name> (ints, ffn as 0 dense / 1 tree), the rest as is."""
    flat = {f"cfg.{k}": np.asarray(v if k != "ffn" else int(v == "tree")) for k, v in im["config"].items()}
    flat.update({k: np.asarray(v) for k, v in im.items() if k != "config"})
    np.savez(path, **flat)


def load_npz(path: Path) -> dict:
    z = np.load(path)
    im = {"config": {}}
    for k in z.files:
        if k.startswith("cfg."):
            v = int(z[k])
            im["config"][k[4:]] = ("tree" if v else "dense") if k == "cfg.ffn" else v
        else:
            v = z[k]
            im[k] = v.item() if v.ndim == 0 else (v.tolist() if k == "alibi_shift" else v)
    return im


def load_float(path: Path) -> MafLM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MafLM(Config(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


# ------------------------------------------------------------------------------------------------ integer primitives
def isqrt(v: torch.Tensor) -> torch.Tensor:
    r = v.double().sqrt().floor().long()
    for _ in range(2):
        r = torch.where(r * r > v, r - 1, r)
        r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


def bitlen(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v > 0, torch.floor(torch.log2(v.double().clamp_min(1))).long() + 1, torch.zeros_like(v))


def quant_codes(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """sign(x) * ((254|x| + m) // 2m), with m the per-vector max|x| (shape [..., 1]); 0 where m == 0."""
    mm = m.clamp_min(1)
    q = (254 * x.abs() + mm) // (2 * mm)
    return torch.where(m > 0, torch.sign(x) * q, torch.zeros_like(x))


def matmul_exact(q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """q [..., n] int (|q| <= 128) times codes [out, n] ternary -> int64 [..., out]. Exact in float32 (n <= 1024)."""
    assert codes.shape[1] <= 1024
    return torch.matmul(q.float(), codes.float().t()).round().long()


def tlinear(x: torch.Tensor, codes: torch.Tensor, wmul: int) -> torch.Tensor:
    n = x.shape[-1]
    sh = (bitlen(x.abs().amax(-1, keepdim=True)) - 15).clamp_min(0)
    xs = x >> sh
    m = xs.abs().amax(-1, keepdim=True)
    q = quant_codes(xs, m)
    acc = matmul_exact(q, codes)
    r = isqrt(((xs * xs).sum(-1, keepdim=True) // n) << 14).clamp_min(1)
    G = (wmul * m * 128 + r // 2) // r.clamp_min(1)
    y = (acc * G + 2 ** 15) >> 16
    return torch.where(m > 0, y, torch.zeros_like(y))


# ---------------------------------------------------------------------------------------------------- the model
class IntLM:
    def __init__(self, im: dict, device: str = "cuda"):
        self.im, self.dev = im, device
        self.c = Config(**im["config"])
        self.t = {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in im.items()
                  if isinstance(v, np.ndarray)}
        self.exp2 = torch.from_numpy(EXP2).to(device)
        torch.backends.cuda.matmul.allow_tf32 = False

    def lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return tlinear(x, self.t[name + ".codes"], self.im[name + ".wmul"])

    def heads(self, y: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, T, n*dh] residual-unit ints -> codes [B, n, T, dh], m [B, n, T, 1]"""
        B, T, _ = y.shape
        y = y.view(B, T, n, self.c.d_head).transpose(1, 2)
        m = y.abs().amax(-1, keepdim=True)
        return quant_codes(y, m), m

    def attn(self, x: torch.Tensor, i: int) -> torch.Tensor:
        c = self.c
        B, T, _ = x.shape
        qc, mq = self.heads(self.lin(x, f"b{i}.wq"), c.heads)
        kc, mk = self.heads(self.lin(x, f"b{i}.wk"), c.kv_heads)
        vc, mv = self.heads(self.lin(x, f"b{i}.wv"), c.kv_heads)
        rep = c.heads // c.kv_heads
        kc, mk, vc, mv = (t.repeat_interleave(rep, dim=1) for t in (kc, mk, vc, mv))
        dot = torch.matmul(qc.float(), kc.float().transpose(-1, -2)).round().long()           # B H T T, exact
        a = (dot * mk.transpose(-1, -2)) >> 12
        Cq = (mq * self.im["KQ"] + 2 ** 15) >> 16
        pos = torch.arange(T, device=self.dev)
        dist = pos[:, None] - pos[None, :]
        shift = torch.tensor(self.im["alibi_shift"], device=self.dev)[:, None, None]
        s = ((a * Cq + 2 ** 23) >> 24) - (dist.clamp_min(0)[None] << shift)
        causal = dist >= 0
        s = torch.where(causal, s, torch.full_like(s, -(2 ** 62)))
        z = s - s.amax(-1, keepdim=True)
        u = (z.clamp_min(-(2 ** 40)) * LOG2E_Q12) >> 12
        ip = -(u >> 12)
        e = self.exp2[(u & 4095) >> 4] >> ip.clamp_max(62)
        e = torch.where((ip < 31) & causal, e, torch.zeros_like(e))
        S = e.sum(-1, keepdim=True)
        p = (255 * e + S // 2) // S
        w = (p * mv.transpose(-1, -2)) >> 12                                                   # B H T(q) T(k)
        o = torch.matmul(w.double(), vc.double()).round().long()                               # exact in float64
        o = o.transpose(1, 2).reshape(B, T, c.heads * c.d_head)
        return self.lin(o, f"b{i}.wo")

    def ffn(self, x: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        c = self.c
        if c.ffn == "dense":
            h = self.lin(x, f"b{i}.w1").clamp_min(0)
            return self.lin(h, f"b{i}.w2"), None
        # tree: node decisions on the exact accumulator sign
        sh = (bitlen(x.abs().amax(-1, keepdim=True)) - 15).clamp_min(0)
        xs = x >> sh
        qx = quant_codes(xs, xs.abs().amax(-1, keepdim=True))
        node_acc = matmul_exact(qx, self.t[f"b{i}.nodes.codes"])
        n = torch.zeros(x.shape[:-1], dtype=torch.long, device=self.dev)
        for lvl in range(c.depth):
            go = (node_acc.gather(-1, ((2 ** lvl - 1) + n).unsqueeze(-1)).squeeze(-1) >= 0).long()
            n = 2 * n + go
        L = c.leaf
        h_all = self.lin(x, f"b{i}.leaf_in").clamp_min(0)                                     # all leaves (batched)
        h = h_all.view(*x.shape[:-1], 2 ** c.depth, L).gather(
            -2, n[..., None, None].expand(*n.shape, 1, L)).squeeze(-2)
        y = torch.zeros_like(x)
        for j in range(2 ** c.depth):
            sel = n == j
            if sel.any():
                y[sel] = self.lin(h[sel], f"b{i}.leaf_out{j}")
        return y, n

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, return_leaves: bool = False):
        x = (self.t["embed.codes"].long()[ids] * self.t["embed.mul"][ids][..., None] + 128) >> 8
        leaves = []
        for i in range(self.c.layers):
            x = x + self.attn(x, i)
            y, n = self.ffn(x, i)
            x = x + y
            leaves.append(n)
            assert x.abs().max().item() < 2 ** 30, "residual overflow"
        logits = self.lin(x, "unembed")
        return (logits, leaves) if return_leaves else logits

    def logits_float(self, logits_int: torch.Tensor) -> torch.Tensor:
        return logits_int.double() / 2 ** F * self.im["logit_scale"]


# ------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def bpb_int(il: IntLM, val: np.ndarray, seq: int, batch: int = 32) -> float:
    n = len(val) // seq
    chunks = torch.from_numpy(val[: n * seq].astype(np.int64)).view(n, seq)
    tot, cnt = 0.0, 0
    for i in range(0, n, batch):
        b = chunks[i:i + batch].to(il.dev)
        lf = il.logits_float(il.forward(b[:, :-1]))
        tot += torch.nn.functional.cross_entropy(lf.reshape(-1, 256), b[:, 1:].reshape(-1), reduction="sum").item()
        cnt += b[:, 1:].numel()
    return tot / cnt / math.log(2)


@torch.no_grad()
def bpb_float(model: MafLM, val: np.ndarray, seq: int, dev: str, batch: int = 32) -> float:
    model = model.to(dev).float()
    n = len(val) // seq
    chunks = torch.from_numpy(val[: n * seq].astype(np.int64)).view(n, seq)
    tot, cnt = 0.0, 0
    for i in range(0, n, batch):
        b = chunks[i:i + batch].to(dev)
        lf = model(b[:, :-1]).double()
        tot += torch.nn.functional.cross_entropy(lf.reshape(-1, 256), b[:, 1:].reshape(-1), reduction="sum").item()
        cnt += b[:, 1:].numel()
    return tot / cnt / math.log(2)


@torch.no_grad()
def generate_int(il: IntLM, prompt: bytes, n: int, window: int = 256) -> list[int]:
    ids = list(prompt)
    for _ in range(n):
        ctx = torch.tensor([ids[-window:]], device=il.dev)
        ids.append(int(il.forward(ctx)[0, -1].argmax()))                  # argmax returns the first maximum
    return ids[len(prompt):]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--npz", type=Path, help="write the integer model here")
    ap.add_argument("--eval-bytes", type=int, default=2_000_000)
    ap.add_argument("--seq", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_float(a.pt)
    im = export(model)
    if a.npz:
        save_npz(im, a.npz)
    il = IntLM(im, dev)
    val = np.fromfile(HERE.parent / "data/tinystories/TinyStoriesV2-GPT4-valid.txt", dtype=np.uint8)[: a.eval_bytes]
    bf = bpb_float(model, val, a.seq, dev)
    bi = bpb_int(il, val, a.seq)
    print(f"{a.pt.name}: float (fp32, fake-quant) {bf:.4f} bits/byte, integer {bi:.4f} (diff {bi - bf:+.4f})")


if __name__ == "__main__":
    main()
