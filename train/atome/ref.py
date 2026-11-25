"""Reference numbers for an Atome LM checkpoint on the PC.

- bits per byte (and perplexity) on the TinyStories V2 validation file, with non-overlapping windows;
- greedy continuations for fixed prompts, using the same window the C engine is compiled with
  (Atome's generate() re-runs the whole window for every token, exactly like atome_predict_next).

    python atome/ref.py --checkpoint atome-lm/checkpoints/atome_1m_v1.pt \
        --max-seq 24 --json ../results/atome_ref.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "atome-lm"))
from atome_llm.core.atome_lm import AtomeLM  # noqa: E402

PROMPTS = ["Once", "The dog", "A girl", "One day, Lily"]


def load(path: Path) -> AtomeLM:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    kw = {k: cfg[k] for k in ("vocab_size", "d_model", "n_layers", "d_head", "top_k", "kernel_size") if k in cfg}
    if "quantizer" in cfg:
        kw["quantizer"] = cfg["quantizer"]
    m = AtomeLM(**kw)
    m.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob["model"])
    return m.eval()


@torch.no_grad()
def bpb(model, ids: torch.Tensor, seq_len: int, device: str, batch: int = 64) -> float:
    n = ids.numel() // seq_len
    chunks = ids[: n * seq_len].view(n, seq_len)
    tot, cnt = 0.0, 0
    for i in range(0, n, batch):
        b = chunks[i:i + batch].to(device)
        logits = model(b[:, :-1])
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
        cnt += b[:, 1:].numel()
    return tot / cnt / math.log(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE.parent / "data/tinystories/TinyStoriesV2-GPT4-valid.txt")
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--max-seq", type=int, default=24, help="engine window for generation")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(a.checkpoint)
    print(f"params {model.parameter_count():,}  d={model.d_model if hasattr(model, 'd_model') else '?'}")

    raw = a.data.read_bytes()[: a.max_bytes]
    ids = torch.tensor(list(raw), dtype=torch.long)
    out = {"checkpoint": str(a.checkpoint), "params": model.parameter_count(), "eval_bytes": len(raw), "bpb": {}}
    model.to(dev)
    for L in (24, 64, 256):
        v = bpb(model, ids, L, dev)
        out["bpb"][L] = v
        print(f"seq {L:3d}: {v:.4f} bits/byte  ppl {2 ** v:.3f}")

    model.to("cpu")
    out["generations"] = []
    for p in PROMPTS:
        x = torch.tensor([list(p.encode())], dtype=torch.long)
        y = model.generate(x, a.max_seq - x.size(1), max_seq=a.max_seq)
        toks = y[0, x.size(1):].tolist()
        text = bytes(toks).decode("utf-8", errors="replace")
        print(f"{p!r} -> {text!r}")
        out["generations"].append({"prompt": p, "max_seq": a.max_seq, "tokens": toks, "text": text})

    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
