"""Worst-case value ranges for every tensor in the graph (interval analysis over all possible inputs).

bounds(graph) -> {node name: (lo, hi, acc_max)}, where lo/hi bound every output value (per tensor,
taken over all channels) and acc_max bounds |any partial sum| inside the op (for int32/SIMD safety).
The input image is any int8. These are proofs, not observations: a back end may use a narrower data
type for a tensor only when its range fits. interp.check_bounds() checks them against real data.
"""
import numpy as np


def _conv_range(k, lo, hi):
    """Per output channel: sum over the 3x3 taps of k*x with x in [lo_c, hi_c]."""
    kk = k.reshape(k.shape[0], -1).astype(np.int64)
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    if lo.ndim == 0:
        lo, hi = np.full(k.shape[0], lo), np.full(k.shape[0], hi)
    a, b = kk * lo[:, None], kk * hi[:, None]
    s_lo, s_hi = np.minimum(a, b).sum(axis=1), np.maximum(a, b).sum(axis=1)
    acc = np.maximum(np.abs(a), np.abs(b)).sum(axis=1)
    return s_lo, s_hi, acc


def bounds(graph):
    out = {}
    lo, hi = np.int64(-128), np.int64(127)          # per-channel arrays after the first conv
    for nd in graph.nodes:
        if nd.op == 'input':
            out[nd.name] = (-128, 127, 0)
        elif nd.op == 'conv3x3':
            s_lo, s_hi, acc = _conv_range(nd.weights, lo, hi)
            sh = nd.attrs['shift']
            lo, hi = np.maximum(s_lo, 0) >> sh, np.maximum(s_hi, 0) >> sh       # ReLU then shift
            out[nd.name] = (int(lo.min()), int(hi.max()), int(acc.max()))
        elif nd.op == 'maxpool2':
            out[nd.name] = out[prev]
        elif nd.op == 'relu_norm':
            lo, hi = np.int64(0), np.int64(127)
            out[nd.name] = (0, 127, 0)
        elif nd.op == 'linear':
            w = nd.weights.astype(np.int64)
            l, h = int(np.min(lo)), int(np.max(hi))
            a, b = w * l, w * h
            s_lo, s_hi = np.minimum(a, b).sum(axis=1), np.maximum(a, b).sum(axis=1)
            acc = np.maximum(np.abs(a), np.abs(b)).sum(axis=1)
            out[nd.name] = (int(s_lo.min()), int(s_hi.max()), int(acc.max()))
            lo, hi = np.int64(s_lo.min()), np.int64(s_hi.max())
        elif nd.op == 'fff':
            # every leaf is possible: the union over leaves; leaf inputs are relu_norm outputs, 0..127
            l, h = int(np.min(lo)), int(np.max(hi))
            acc = 0
            for key, (al, ah) in (('nodes', (l, h)), ('leaf_in', (l, h)), ('leaf_out', (0, 127))):
                w = nd.tensors[key].astype(np.int64)
                acc = max(acc, int((np.maximum(np.abs(w * al), np.abs(w * ah))).sum(axis=-1).max()))
            w = nd.tensors['leaf_out'].astype(np.int64)
            s_lo = (np.minimum(w * 0, w * 127)).sum(axis=-1)
            s_hi = (np.maximum(w * 0, w * 127)).sum(axis=-1)
            out[nd.name] = (int(s_lo.min()), int(s_hi.max()), acc)
            lo, hi = np.int64(s_lo.min()), np.int64(s_hi.max())
        elif nd.op == 'argmax':
            out[nd.name] = out[prev]
        prev = nd.name
    return out


def fits_int16(b):
    lo, hi, _ = b
    return -32768 <= lo and hi <= 32767
