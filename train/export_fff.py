"""Export a trained CNNMNIST or CNNMNIST_FFF model to integer weight codes (.npz), for idea B (fast feedforward tree).

BitNetMCU's exportquant.py writes only plain layer chains, so the tree layer needs its own export. The integer
codes are the ones BitNetMCU's own export uses: each layer's weight_quant() codes, with 4bitsym stored as the odd
integers 2u (+-1 .. +-15), as harness/golden.py decodes them from the header.

  CNNMNIST:      conv1, dw1, dw2 (C, 3, 3) int8; fc1, fc2, fc3 (n_out, n_in)
  CNNMNIST_FFF:  conv1, dw1, dw2; nodes (2^d - 1, n_in); leaf_in (2^d, leaf, n_in); leaf_out (2^d, d_out, leaf);
                 fc3 (10, d_out); depth, leaf; quant names per tensor

Usage (from the repo root):
  train/.venv/Scripts/python.exe train/export_fff.py --params train/configs/fff_cnn80.yaml --pth train/models/fff80.pth \
      --out train/export/fff80.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
BITNET = HERE / 'BitNetMCU'


def codes(layer):
    """Integer weight codes of one BitLinear/BitConv2d, as BitNetMCU's export encodes them."""
    u, _, _ = layer.weight_quant(layer.weight.data)
    u = u.detach().cpu().numpy().astype(np.float64)
    q = layer.QuantType
    if q == '4bitsym':
        v = 2 * u                          # +-0.5 .. +-7.5 -> odd +-1 .. +-15
    elif q in ('8bit', 'Ternary'):
        v = u
    else:
        raise NotImplementedError(q)
    assert np.allclose(v, np.round(v)), q
    return np.round(v).astype(np.int64), {'4bitsym': 'int4sym', '8bit': 'int8', 'Ternary': 'ternary'}[q]


def load(params_path, pth):
    params = yaml.safe_load(open(params_path))
    cwd = os.getcwd()
    sys.path.insert(0, str(BITNET))
    os.chdir(BITNET)
    try:
        from training import load_model
        model = load_model(params['model'], {**params, 'num_classes': 10})
        sd = torch.load(pth, map_location='cpu')
        model.load_state_dict(sd.get('state_dict', sd) if isinstance(sd, dict) else sd, strict=False)
        model.eval()
    finally:
        os.chdir(cwd)
    return params, model


def export(params, model):
    from BitNetMCU import BitConv2d
    convs = [m for m in model.modules() if isinstance(m, BitConv2d)]
    assert len(convs) == 3
    out = {}
    for name, m in zip(('conv1', 'dw1', 'dw2'), convs):
        w, q = codes(m)
        out[name] = w.reshape(w.shape[0], 3, 3)
        assert q == 'int8'
    if params['model'] == 'CNNMNIST':
        lin = [m for m in model.model if m.__class__.__name__ == 'BitLinear'] + [model.classifier]
        for name, m in zip(('fc1', 'fc2', 'fc3'), lin):
            out[name], out[name + '_quant'] = codes(m)
    elif params['model'] == 'CNNMNIST_FFF':
        f = model.fff
        d, l = f.depth, f.leaf
        out['depth'], out['leaf'] = np.int64(d), np.int64(l)
        out['nodes'], out['nodes_quant'] = codes(f.nodes)
        w, out['leaf_in_quant'] = codes(f.leaf_in)
        out['leaf_in'] = w.reshape(2 ** d, l, -1)              # row j*l + k of leaf_in is neuron k of leaf j
        lo = [codes(m) for m in f.leaf_out]
        out['leaf_out'] = np.stack([w for w, _ in lo])
        out['leaf_out_quant'] = lo[0][1]
        out['fc3'], out['fc3_quant'] = codes(model.classifier)
    else:
        raise NotImplementedError(params['model'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--pth', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    params, model = load(Path(args.params).resolve(), Path(args.pth).resolve())
    out = export(params, model)
    np.savez(args.out, **out)
    for k, v in out.items():
        print(f'  {k:15s} {getattr(v, "shape", "")} {v if np.ndim(v) == 0 else ""}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
