"""Run the Atome LM firmware (firmware/baselines/atome) on the Pico 2 and check it against the PC reference.

For each prompt in the reference JSON (train/atome/ref.py --json), asks the board for a greedy continuation up to
the engine window, compares the tokens, and reports cycles per token, tokens/s and flash bytes read per token
(XIP cache misses x 8 bytes). Appends one row per build to the CSV.

    python harness/atome_run.py --ref results/atome_ref_seq24.json --csv results/phase3.csv [--port COM3]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import date
from pathlib import Path

import serial


def ask(s: serial.Serial, data: bytes) -> list[str]:
    s.reset_input_buffer()
    s.write(data)
    lines = []
    while True:
        ln = s.readline().decode(errors="replace").strip()
        if not ln:
            raise TimeoutError("board stopped answering")
        lines.append(ln)
        if ln.startswith(("INFO", "E ", "L ")):
            return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    ref = json.loads(a.ref.read_text())
    with serial.Serial(a.port, 115200, timeout=120) as s:
        time.sleep(0.3)
        info = ask(s, b"?")[-1]
        kv = dict(t.split("=", 1) for t in info.split()[1:] if "=" in t)
        hz = int(kv["sys_hz"])
        print(info)
        rows, all_match = [], True
        for g in ref["generations"]:
            p = g["prompt"].encode()
            lines = ask(s, b"G" + bytes([len(p)]) + p + bytes([len(g["tokens"])]))
            toks, cyc, miss = [], [], []
            for ln in lines:
                if ln.startswith("T "):
                    _, t, c, acc, hit = ln.split()
                    toks.append(int(t))
                    cyc.append(int(c))
                    miss.append(int(acc) - int(hit))
            ok = toks == g["tokens"]
            all_match &= ok
            ms = [c / hz * 1e3 for c in cyc]
            text = bytes(toks).decode(errors="replace")
            print(f"{g['prompt']!r:16} -> {text!r:28} match={ok}  ms/token first {ms[0]:.0f}, last {ms[-1]:.0f}, "
                  f"mean {statistics.mean(ms):.0f}  -> {len(ms) / (sum(ms) / 1e3):.2f} tok/s  "
                  f"flash read {statistics.mean(miss) * 8 / 1024:.0f} KB/token")
            rows.append((ms, miss))

    ms_all = [m for r in rows for m in r[0]]
    miss_all = [m for r in rows for m in r[1]]
    tok_s = len(ms_all) / (sum(ms_all) / 1e3)
    print(f"all prompts: {tok_s:.2f} tok/s, mean {statistics.mean(ms_all):.1f} ms/token, "
          f"{statistics.mean(miss_all) * 8 / 1024:.0f} KB flash/token, bit-exact={all_match}")
    if a.csv:
        new = not a.csv.exists()
        with a.csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "build", "max_seq", "blob_bytes", "state_bytes", "tokens", "bit_exact",
                            "mean_ms_per_token", "tok_per_s", "flash_read_kb_per_token", "notes"])
            w.writerow([date.today().isoformat(), kv["build"], kv["max_seq"], kv["blob"], kv["state"], len(ms_all),
                        all_match, f"{statistics.mean(ms_all):.2f}", f"{tok_s:.3f}",
                        f"{statistics.mean(miss_all) * 8 / 1024:.1f}", a.notes])


if __name__ == "__main__":
    main()
