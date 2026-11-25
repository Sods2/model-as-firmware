"""Compile a MafLM integer model (.npz) into C data for the RP2350 LM runtime (firmware/rp2350/maflm).

    python compiler/compile_lm.py --npz train/runs/check_tree.npz --name check_tree
"""
import argparse
from pathlib import Path

from mafc.lm_backend import generate, load_npz

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--window", type=int, default=256, help="KV cache length (tokens)")
    ap.add_argument("--out", type=Path, default=ROOT / "firmware/rp2350/maflm/gen")
    a = ap.parse_args()
    st = generate(load_npz(a.npz), a.name, a.out, a.window)
    print(f"{a.name}: packed weights {st['packed_bytes'] / 1024:.1f} KB, embedding {st['embed_bytes'] / 1024:.1f} KB, "
          f"KV cache {st['kv_bytes'] / 1024:.1f} KB -> {a.out}")


if __name__ == "__main__":
    main()
