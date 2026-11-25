"""Report the zero fraction of every Ternary layer in a trained BitNetMCU model.

Usage (from train/BitNetMCU):  python ../sparsity_report.py <config.yaml> <model.pth>
"""
import os
import sys
# train/BitNetMCU (the folder) would shadow BitNetMCU.py; import from the repo itself.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "BitNetMCU"))
import yaml
import torch
from BitNetMCU import BitLinear
from exportquant import load_model

cfg_path, pth_path = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    params = yaml.safe_load(f)
model = load_model(params["model"], params)
model.load_state_dict(torch.load(pth_path, map_location="cpu"), strict=False)

with torch.no_grad():
    for name, m in model.named_modules():
        if isinstance(m, BitLinear) and m.QuantType == "Ternary":
            u, _, _ = m.weight_quant(m.weight)
            n = u.numel()
            z = (u == 0).sum().item()
            print(f"{name}: {tuple(u.shape)} zeros={z}/{n} = {100 * z / n:.1f}%  "
                  f"(+1: {(u == 1).sum().item()}, -1: {(u == -1).sum().item()})  target_frac={m.ternary_zero_frac}")
