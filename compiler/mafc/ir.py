"""mafc IR: a small, explicit graph of integer ops.

The IR sits between front ends (which read a trained model) and back ends (which emit code for one
target). It is deliberately narrow: it only has the ops our models use, every op states its exact
integer arithmetic, and weights are stored as integer codes (not packed), so each back end chooses its
own storage format. The reference semantics are in interp.py; a back end is correct when its output
equals the interpreter's bit for bit.

Shapes are CHW for feature maps and (N,) for vectors. Activation dtypes are 'int8' or 'int32'.

Ops
  input        out (C, H, W) int8                                       the image
  conv3x3      (C_in, H, W) -> (C_out, H-2, W-2) int32                  valid 3x3 conv, stride 1, no bias,
               weights (C_out, 3, 3); attrs: mode 'broadcast' (C_in = 1) or 'depthwise' (C_in = C_out),
               shift: out = sum < 0 ? 0 : sum >> shift                  (ReLU fused)
  maxpool2     (C, H, W) -> (C, H/2, W/2)                               2x2 max, stride 2, dtype kept
  relu_norm    (any) -> (N,) int8                                       flatten (CHW order), ReLU, then the
               smallest power-of-two right shift that brings max <= 127, round half up, clip to 0..127
  linear       (N_in,) int8 -> (N_out,) int32                           y = W x, no bias;
               weights (N_out, N_in); attrs: quant 'ternary' {-1,0,1} | 'int4sym' odd {-15..15} | 'int8'
  fff          (N_in,) int8 -> (N_out,) int32                           idea B, a fast feedforward tree (no bias):
               tensors nodes (2^d - 1, N_in), leaf_in (2^d, L, N_in), leaf_out (2^d, N_out, L);
               attrs depth d, leaf L, quant_nodes / quant_in / quant_out.
               n = 0; for each level m: n = 2n + (nodes[2^m - 1 + n] . x >= 0);
               h = leaf_in[n] @ x; y = leaf_out[n] @ relu_norm(h)       (only leaf n is computed)
  argmax       (N,) -> scalar                                           index of the first maximum
"""
from dataclasses import dataclass, field
import json

import numpy as np

QUANT_RANGES = {'ternary': (-1, 1), 'int4sym': (-15, 15), 'int8': (-128, 127)}


@dataclass
class Node:
    name: str
    op: str
    in_shape: tuple
    out_shape: tuple
    dtype: str = 'int32'
    attrs: dict = field(default_factory=dict)
    weights: np.ndarray = None
    tensors: dict = None          # ops with several weight tensors (fff)

    def check(self):
        w = self.weights
        if self.op == 'conv3x3':
            c_in, h, wd = self.in_shape
            c_out = self.out_shape[0]
            assert self.attrs['mode'] in ('broadcast', 'depthwise')
            assert (c_in == 1) if self.attrs['mode'] == 'broadcast' else (c_in == c_out)
            assert w.shape == (c_out, 3, 3) and self.out_shape == (c_out, h - 2, wd - 2)
            assert w.min() >= -128 and w.max() <= 127
        elif self.op == 'linear':
            n_in = int(np.prod(self.in_shape))
            assert w.shape == (self.out_shape[0], n_in), (w.shape, self.in_shape)
            lo, hi = QUANT_RANGES[self.attrs['quant']]
            assert w.min() >= lo and w.max() <= hi
            if self.attrs['quant'] == 'int4sym':
                assert np.all(w % 2 == 1), 'int4sym weights are odd'
        elif self.op == 'fff':
            d, L = self.attrs['depth'], self.attrs['leaf']
            n_in, n_out = int(np.prod(self.in_shape)), self.out_shape[0]
            t = self.tensors
            assert t['nodes'].shape == (2 ** d - 1, n_in)
            assert t['leaf_in'].shape == (2 ** d, L, n_in) and t['leaf_out'].shape == (2 ** d, n_out, L)
            for key, q in (('nodes', 'quant_nodes'), ('leaf_in', 'quant_in'), ('leaf_out', 'quant_out')):
                lo, hi = QUANT_RANGES[self.attrs[q]]
                assert t[key].min() >= lo and t[key].max() <= hi, key
        elif self.op == 'maxpool2':
            c, h, wd = self.in_shape
            assert h % 2 == 0 and wd % 2 == 0 and self.out_shape == (c, h // 2, wd // 2)

    def describe(self):
        s = f'{self.name:8s} {self.op:9s} {str(self.in_shape):14s} -> {str(self.out_shape):14s} {self.dtype}'
        if self.weights is not None:
            w = self.weights
            q = self.attrs.get('quant', 'int8')
            s += f'  w {q} {w.size} ({(w == 0).mean() * 100:.0f}% zero)'
        if self.tensors is not None:
            s += '  ' + ', '.join(f'{k} {v.shape}' for k, v in self.tensors.items())
        extra = {k: v for k, v in self.attrs.items() if k != 'quant'}
        return s + (f'  {extra}' if extra else '')


@dataclass
class Graph:
    name: str
    nodes: list

    def check(self):
        assert self.nodes[0].op == 'input' and self.nodes[-1].op == 'argmax'
        for a, b in zip(self.nodes, self.nodes[1:]):
            assert int(np.prod(a.out_shape)) == int(np.prod(b.in_shape)), (a.name, b.name)
            b.check()
        return self

    def describe(self):
        return f'graph {self.name}\n' + '\n'.join('  ' + n.describe() for n in self.nodes)

    def to_json(self):
        """IR without weights, for inspection and diffs."""
        return json.dumps({'name': self.name, 'nodes': [
            {'name': n.name, 'op': n.op, 'in': list(n.in_shape), 'out': list(n.out_shape), 'dtype': n.dtype,
             'attrs': n.attrs, 'weights': None if n.weights is None else list(n.weights.shape),
             'tensors': None if n.tensors is None else {k: list(v.shape) for k, v in n.tensors.items()}}
            for n in self.nodes]}, indent=1)
