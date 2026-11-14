"""Train a MafLM twin on TinyStories V2 (bytes), with the bits-per-byte evaluation used for Atome (train/atome/ref.py).

    .venv/Scripts/python.exe lm/train.py --ffn dense --steps 10000 --out models/lm_dense.pt
    .venv/Scripts/python.exe lm/train.py --ffn tree  --steps 10000 --out models/lm_tree.pt

The training text is read straight from the .txt file as bytes (np.memmap), random windows of seq+1 bytes.
Evaluation: the first --eval-bytes of the validation file in non-overlapping windows, as ref.py does; the tree is
always evaluated with hard routing. Writes <out> (best validation) and <out>.json (config, args, log).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import Config, MafLM  # noqa: E402

DATA = HERE.parent / "data" / "tinystories"


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model: MafLM, val: np.ndarray, seq: int, dev, batch: int = 64) -> tuple[float, np.ndarray | None]:
    model.eval()
    n = len(val) // seq
    chunks = torch.from_numpy(val[: n * seq].astype(np.int64)).view(n, seq)
    tot, cnt = 0.0, 0
    usage = None
    for i in range(0, n, batch):
        b = chunks[i:i + batch].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(b[:, :-1])
        tot += F.cross_entropy(logits.float().reshape(-1, 256), b[:, 1:].reshape(-1), reduction="sum").item()
        cnt += b[:, 1:].numel()
        for j, t in enumerate(model.trees()):
            u = torch.bincount(t.last_leaf.flatten(), minlength=t.n_leaves).cpu().numpy()
            if usage is None:
                usage = np.zeros((len(model.trees()), t.n_leaves), np.int64)
            usage[j] += u
    model.train()
    return tot / cnt / math.log(2), usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffn", choices=["dense", "tree"], default="dense")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--soft-frac", type=float, default=0.3, help="tree: fraction of steps with soft routing")
    ap.add_argument("--harden", type=float, default=0.1, help="tree: hardening loss weight (reached at soft end)")
    ap.add_argument("--balance", type=float, default=0.1, help="tree: load-balancing loss weight")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-bytes", type=int, default=2_000_000)
    ap.add_argument("--train-file", type=Path, default=DATA / "TinyStoriesV2-GPT4-train.txt")
    ap.add_argument("--val-file", type=Path, default=DATA / "TinyStoriesV2-GPT4-valid.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = np.memmap(a.train_file, dtype=np.uint8, mode="r")
    val = np.fromfile(a.val_file, dtype=np.uint8)[: a.eval_bytes]

    cfg = Config(d=a.d, layers=a.layers, hidden=a.hidden, ffn=a.ffn, depth=a.depth, max_seq=a.seq)
    model = MafLM(cfg).to(dev)
    print(f"{a.ffn}: {model.n_weights():,} weights, {model.weights_read_per_token():,} read per token; "
          f"train {len(train) / 1e6:.0f} MB, device {dev}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    other = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    soft_steps = int(a.soft_frac * a.steps) if a.ffn == "tree" else 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    t0 = time.time()
    for step in range(a.steps + 1):
        for t in model.trees():
            t.hard = step >= soft_steps
            t.harden_w = a.harden * min(1.0, step / max(1, soft_steps))
            t.balance_w = a.balance
        if step % a.eval_every == 0 or step == a.steps:
            bpb, usage = evaluate(model, val, a.seq, dev)
            rec = {"step": step, "val_bpb": bpb, "val_ppl": 2 ** bpb, "time_s": round(time.time() - t0)}
            msg = f"  >> step {step}: val {bpb:.4f} bits/byte (ppl {2 ** bpb:.3f})"
            if usage is not None:
                frac = usage / usage.sum(1, keepdims=True)
                rec["leaf_usage"] = frac.round(4).tolist()
                ent = -(frac * np.log2(np.clip(frac, 1e-12, 1))).sum(1)
                msg += "  leaf entropy " + " ".join(f"{e:.2f}" for e in ent) + f" / {math.log2(frac.shape[1]):.0f}"
            print(msg, flush=True)
            log.append(rec)
            if bpb < best and (a.ffn == "dense" or step >= soft_steps):
                best = bpb
                torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(), "step": step,
                            "val_bpb": bpb}, a.out)
            if step == a.steps:
                break
        lr = lr_at(step, a.steps, a.lr, a.warmup, a.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        starts = rng.integers(0, len(train) - a.seq - 1, a.batch)
        b = torch.from_numpy(np.stack([train[s:s + a.seq + 1] for s in starts]).astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(b[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, 256), b[:, 1:].reshape(-1))
            aux = model.aux_loss()
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f} ({loss.item() / math.log(2):.3f} bpb)  aux {float(aux.detach()):.4f}"
                  f"  lr {lr:.1e}  {'' if a.ffn == 'dense' else 'hard' if step >= soft_steps else 'soft'}"
                  f"  {time.time() - t0:.0f}s", flush=True)

    a.out.with_suffix(".json").write_text(json.dumps(
        {"config": cfg.to_dict(), "args": {k: str(v) for k, v in vars(a).items()}, "weights": model.n_weights(),
         "read_per_token": model.weights_read_per_token(), "best_val_bpb": best, "log": log}, indent=1))
    print(f"best val {best:.4f} bits/byte (ppl {2 ** best:.3f}); wrote {a.out}")


if __name__ == "__main__":
    main()
