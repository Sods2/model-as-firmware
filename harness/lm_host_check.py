"""Build the MafLM runtime for the PC and compare its logits with the integer reference (train/lm/intref.py).

    python harness/lm_host_check.py --name lm_tree --npz train/runs/lm_tree.npz

Compiles with the PC's C compiler: $HOST_CC, else C:/MinGW/bin/gcc.exe if it exists (what was used), else gcc on
PATH. Plain C, no DSP intrinsics: SMLAD is emulated in C.
Sequences are slices of TinyStories V2-valid, each run from a fresh KV cache.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train/lm"))
from intref import IntLM, load_npz  # noqa: E402

FW = ROOT / "firmware/rp2350/maflm"
GCC = (os.environ.get("HOST_CC") or ("C:/MinGW/bin/gcc.exe" if Path("C:/MinGW/bin/gcc.exe").exists() else None)
       or shutil.which("gcc") or "gcc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dual", action="store_true", help="compile the two-core partitioning (run on one core)")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        sys.exit(check(a, Path(td)))


def check(a, tmp: Path) -> int:
    exe = tmp / "host_main.exe"
    cmd = [GCC, "-O2", "-std=c11", f'-DLM_GEN_HEADER="maflm_{a.name}.h"', f"-DLM_DUAL={int(a.dual)}", f"-I{FW}", f"-I{FW / 'gen'}",
           str(FW / "host_main.c"), str(FW / "lm_rt.c"), str(FW / "gen" / f"maflm_{a.name}.c"), "-o", str(exe)]
    subprocess.run(cmd, check=True)

    il = IntLM(load_npz(a.npz), "cuda" if torch.cuda.is_available() else "cpu")
    val = np.fromfile(ROOT / "train/data/tinystories/TinyStoriesV2-GPT4-valid.txt", dtype=np.uint8)
    ok = True
    for s in range(a.n):
        ids = val[s * 5000: s * 5000 + a.len]
        (tmp / "t.bin").write_bytes(ids.tobytes())
        subprocess.run([str(exe), str(tmp / "t.bin"), str(tmp / "l.bin")], check=True, capture_output=True)
        c_logits = np.fromfile(tmp / "l.bin", dtype="<i8").reshape(-1, 256)
        ref = il.forward(torch.from_numpy(ids.astype(np.int64))[None].to(il.dev))[0].cpu().numpy()
        same = np.array_equal(c_logits, ref)
        ok &= same
        msg = "" if same else f", first difference at position {int(np.argmax((c_logits != ref).any(1)))}"
        print(f"sequence {s}: {len(ids)} positions, logits identical: {same}{msg}")
    print("PC build bit-exact" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
