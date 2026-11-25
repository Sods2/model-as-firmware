"""Run: python train/atome/fairness.py

Fairness check: Atome's corpus script builds its training text from TinyStoriesV2-GPT4-valid.txt (our test file),
cleaned (whitespace normalised, paragraphs deduplicated) and capped at --max-bytes (default 1,000,000).
Evaluate Atome and both MafLM twins on raw and cleaned slices to see (a) whether Atome saw our test text and
(b) how much the text format matters."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "train/atome"))
sys.path.insert(0, str(R / "train/lm"))
import ref  # noqa: E402
from model import Config, MafLM  # noqa: E402
from intref import bpb_float  # noqa: E402

spec = importlib.util.spec_from_file_location("bc", R / "train/atome-lm/scripts/build_corpus.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

raw_text = (R / "train/data/tinystories/TinyStoriesV2-GPT4-valid.txt").read_bytes()
clean, stats = bc.clean_pipeline([raw_text.decode("utf-8", errors="replace")], max_bytes=10 ** 12)
cb = clean.encode("utf-8")
print("raw", len(raw_text), "cleaned", len(cb), stats["paragraphs"], "paragraphs")
MB = 1_000_000
sets = {
    "raw 0-2 MB (current eval)": raw_text[:2 * MB],
    "raw last 2 MB": raw_text[-2 * MB:],
    "cleaned 0-1 MB (Atome's default corpus)": cb[:MB],
    "cleaned 1-3 MB": cb[MB:3 * MB],
    "cleaned last 2 MB": cb[-2 * MB:],
}
dev = "cuda"
atome = ref.load(R / "train/atome-lm/checkpoints/atome_1m_v1.pt").to(dev)
models = {"Atome": atome}
for t in ("dense", "tree"):
    blob = torch.load(R / f"train/runs/lm_{t}.pt", map_location="cpu")
    m = MafLM(Config(**blob["config"]))
    m.load_state_dict(blob["state_dict"])
    models[t] = m.eval()
torch.backends.cuda.matmul.allow_tf32 = False
print(f"{'set':42s}" + "".join(f"{k:>9s}" for k in models))
for name, data in sets.items():
    arr = np.frombuffer(data, dtype=np.uint8)
    row = []
    for k, m in models.items():
        if k == "Atome":
            row.append(ref.bpb(m, torch.from_numpy(arr.astype(np.int64)), 256, dev))
        else:
            with torch.no_grad():
                row.append(bpb_float(m, arr, 256, dev))
    print(f"{name:42s}" + "".join(f"{v:9.4f}" for v in row), flush=True)
print("first 300 cleaned bytes:", repr(cb[:300]))
print("first 300 raw bytes:", repr(raw_text[:300]))
