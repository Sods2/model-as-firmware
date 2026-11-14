"""Token-by-token integer decoder with a KV cache, in plain numpy int64: the shape of the firmware's loop.

Independent of intref.py's batched GPU code (no float matmuls); used to check that the batched reference and
incremental decoding give bit-identical logits, and as the golden model for the board.

    .venv/Scripts/python.exe lm/intref_step.py --pt runs/check_tree.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intref import EXP2, LOG2E_Q12, IntLM, export, load_float  # noqa: E402
from model import Config  # noqa: E402


def isqrt(v: int) -> int:
    import math
    return math.isqrt(v)


def codes_of(x: np.ndarray, m: int) -> np.ndarray:
    if m == 0:
        return np.zeros_like(x)
    return np.sign(x) * ((254 * np.abs(x) + m) // (2 * m))


class StepLM:
    def __init__(self, im: dict):
        self.im = im
        self.c = Config(**im["config"])
        self.reset()

    def reset(self):
        self.k, self.v = [[] for _ in range(self.c.layers)], [[] for _ in range(self.c.layers)]   # (codes, m) per head
        self.pos = 0

    def lin(self, x: np.ndarray, name: str) -> np.ndarray:
        W, wmul = self.im[name + ".codes"].astype(np.int64), self.im[name + ".wmul"]
        n = x.shape[0]
        mx = int(np.abs(x).max())
        sh = max(0, mx.bit_length() - 15)
        xs = x >> sh
        m = int(np.abs(xs).max())
        if m == 0:
            return np.zeros(W.shape[0], np.int64)
        q = codes_of(xs, m)
        acc = W @ q
        r = max(1, isqrt((int((xs * xs).sum()) // n) << 14))
        G = (wmul * m * 128 + r // 2) // r
        return (acc * G + 2 ** 15) >> 16

    def attn(self, x: np.ndarray, i: int) -> np.ndarray:
        c, dh = self.c, self.c.d_head
        q = self.lin(x, f"b{i}.wq").reshape(c.heads, dh)
        k = self.lin(x, f"b{i}.wk").reshape(c.kv_heads, dh)
        v = self.lin(x, f"b{i}.wv").reshape(c.kv_heads, dh)
        self.k[i].append([(codes_of(k[g], int(np.abs(k[g]).max())), int(np.abs(k[g]).max())) for g in range(c.kv_heads)])
        self.v[i].append([(codes_of(v[g], int(np.abs(v[g]).max())), int(np.abs(v[g]).max())) for g in range(c.kv_heads)])
        T = len(self.k[i])
        out = np.zeros((c.heads, dh), np.int64)
        for h in range(c.heads):
            g = h // (c.heads // c.kv_heads)
            mq = int(np.abs(q[h]).max())
            qc = codes_of(q[h], mq)
            Cq = (mq * self.im["KQ"] + 2 ** 15) >> 16
            s = np.empty(T, np.int64)
            for j in range(T):
                kc, mk = self.k[i][j][g]
                a = (int(qc @ kc) * mk) >> 12
                s[j] = ((a * Cq + 2 ** 23) >> 24) - ((T - 1 - j) << self.im["alibi_shift"][h])
            z = s - s.max()
            u = (z * LOG2E_Q12) >> 12
            ip = -(u >> 12)
            e = np.where(ip < 31, EXP2[(u & 4095) >> 4] >> np.minimum(ip, 62), 0)
            S = int(e.sum())
            p = (255 * e + S // 2) // S
            for j in range(T):
                vc, mv = self.v[i][j][g]
                out[h] += ((int(p[j]) * mv) >> 12) * vc
        return self.lin(out.reshape(-1), f"b{i}.wo")

    def ffn(self, x: np.ndarray, i: int) -> tuple[np.ndarray, int | None]:
        c = self.c
        if c.ffn == "dense":
            return self.lin(np.maximum(self.lin(x, f"b{i}.w1"), 0), f"b{i}.w2"), None
        mx = int(np.abs(x).max())
        xs = x >> max(0, mx.bit_length() - 15)
        qx = codes_of(xs, int(np.abs(xs).max()))
        nodes = self.im[f"b{i}.nodes.codes"].astype(np.int64)
        n = 0
        for lvl in range(c.depth):
            n = 2 * n + int(nodes[(2 ** lvl - 1) + n] @ qx >= 0)
        L = c.leaf
        W = self.im[f"b{i}.leaf_in.codes"]
        sub = dict(self.im)
        sub["_li.codes"], sub["_li.wmul"] = W[n * L:(n + 1) * L], self.im[f"b{i}.leaf_in.wmul"]
        saved, self.im = self.im, sub
        h = np.maximum(self.lin(x, "_li"), 0)
        self.im = saved
        return self.lin(h, f"b{i}.leaf_out{n}"), n

    def step(self, tok: int) -> np.ndarray:
        im = self.im
        x = (im["embed.codes"][tok].astype(np.int64) * int(im["embed.mul"][tok]) + 128) >> 8
        for i in range(self.c.layers):
            x = x + self.attn(x, i)
            y, _ = self.ffn(x, i)
            x = x + y
        self.pos += 1
        return self.lin(x, "unembed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--len", type=int, default=64)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    im = export(load_float(a.pt))
    il = IntLM(im, "cuda" if torch.cuda.is_available() else "cpu")
    sl = StepLM(im)
    val = np.fromfile(HERE.parent / "data/tinystories/TinyStoriesV2-GPT4-valid.txt", dtype=np.uint8)
    ok = True
    for s in range(a.n):
        ids = val[s * 1000: s * 1000 + a.len].astype(np.int64)
        batched = il.forward(torch.from_numpy(ids)[None].to(il.dev))[0].cpu().numpy()
        sl.reset()
        step = np.stack([sl.step(int(t)) for t in ids])
        same = np.array_equal(batched, step)
        ok &= same
        print(f"sequence {s}: {a.len} positions, logits identical: {same}"
              + ("" if same else f" (first difference at position {np.argmax((batched != step).any(1))})"))
    print("bit-exact" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
