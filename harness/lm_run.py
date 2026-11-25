"""Run MafLM firmware (firmware/rp2350/maflm) on the Pico 2: check it bit-exact against the integer reference, then
time greedy generation.

    python harness/lm_run.py --npz train/runs/check_tree.npz --csv results/phase3.csv

1. Exactness: --check sequences of TinyStories V2-valid, --check-len tokens each from an empty KV cache; for every
   position the board's argmax and logits checksum (sum_i logits[i] * (2i+1) mod 2^64) must equal the reference's.
2. Speed: for each prompt, generate until the KV cache window is full; report ms/token over the first 24 positions
   (Atome's window on the ESP32 and in results/phase3.csv) and over the whole window, plus flash bytes per token.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import serial
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train/lm"))
from intref import IntLM, load_npz  # noqa: E402

PROMPTS = ["Once", "The dog", "A girl", "One day, Lily"]


def line(s: serial.Serial) -> str:
    ln = s.readline().decode(errors="replace").strip()
    if not ln:
        raise TimeoutError("board stopped answering")
    return ln


def checksum(logits: np.ndarray) -> int:
    w = (2 * np.arange(256, dtype=np.uint64) + 1)
    return int((logits.astype(np.int64).view(np.uint64) * w).sum(dtype=np.uint64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--check", type=int, default=2)
    ap.add_argument("--check-len", type=int, default=256)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    im = load_npz(a.npz)
    il = IntLM(im, "cuda" if torch.cuda.is_available() else "cpu")
    val = np.fromfile(ROOT / "train/data/tinystories/TinyStoriesV2-GPT4-valid.txt", dtype=np.uint8)
    with serial.Serial(a.port, 115200, timeout=60) as s:
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(b"?")
        info = line(s)
        kv = dict(t.split("=", 1) for t in info.split()[1:])
        hz, window = int(kv["sys_hz"]), int(kv["window"])
        print(info)

        exact = True
        for k in range(a.check):
            ids = val[k * 7919: k * 7919 + a.check_len]
            ref = il.forward(torch.from_numpy(ids.astype(np.int64))[None].to(il.dev))[0].cpu().numpy()
            s.write(b"R")
            line(s)
            bad = None
            for t, tok in enumerate(ids):
                s.write(b"S" + bytes([int(tok)]))
                _, am, _, cs, *_ = line(s).split()
                if int(am) != int(ref[t].argmax()) or int(cs, 16) != checksum(ref[t]):
                    bad = t
                    break
            exact &= bad is None
            print(f"exactness, sequence {k}: {len(ids)} positions " +
                  ("identical" if bad is None else f"MISMATCH at position {bad}"))

        rows = []
        for p in PROMPTS:
            pb = p.encode()
            n = window - len(pb)
            s.write(b"G" + bytes([len(pb)]) + pb + bytes([min(n, 255)]))
            toks, cyc, miss = [], [], []
            while True:
                ln = line(s)
                if ln.startswith("T "):
                    _, t, c, acc, hit = ln.split()
                    toks.append(int(t))
                    cyc.append(int(c))
                    miss.append(int(acc) - int(hit))
                elif ln.startswith("E "):
                    break
            # positions: the first generated step runs at position len(prompt)
            pos = np.arange(len(pb), len(pb) + len(cyc))
            ms = np.array(cyc) / hz * 1e3
            first24 = ms[pos < 24]
            text = (pb + bytes(toks)).decode(errors="replace")
            print(f"{p!r:16} {len(ms)} tokens: {first24.mean():.2f} ms/token in the first 24 positions, "
                  f"{ms.mean():.2f} over the window ({1e3 / ms.mean():.1f} tok/s), last {ms[-1]:.2f}; "
                  f"flash {np.mean(miss) * 8 / 1024:.1f} KB/token")
            print(f"    {text[:150]!r}")
            rows.append((ms, first24, np.array(miss)))

    ms_all = np.concatenate([r[0] for r in rows])
    f24 = np.concatenate([r[1] for r in rows])
    miss_all = np.concatenate([r[2] for r in rows])
    print(f"all prompts: first 24 positions {f24.mean():.2f} ms/token ({1e3 / f24.mean():.1f} tok/s); whole window "
          f"{ms_all.mean():.2f} ms/token ({1e3 / ms_all.mean():.1f} tok/s); bit-exact={exact}")
    if a.csv:
        new = not a.csv.exists()
        with a.csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "build", "max_seq", "blob_bytes", "state_bytes", "tokens", "bit_exact",
                            "mean_ms_per_token", "tok_per_s", "flash_read_kb_per_token", "notes"])
            for label, arr, mi in (("24", f24, miss_all[:len(f24)]), (str(window), ms_all, miss_all)):
                w.writerow([date.today().isoformat(), kv["build"], label, "", "", len(arr), exact,
                            f"{arr.mean():.3f}", f"{1e3 / arr.mean():.2f}", f"{mi.mean() * 8 / 1024:.1f}", a.notes])


if __name__ == "__main__":
    main()
