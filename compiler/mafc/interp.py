"""Reference interpreter for the mafc IR (numpy, int64, batched). Defines what every op computes.

run(graph, imgs) returns the logits (the last int32 tensor before argmax), shape [N, n_classes].
It also asserts that no value an int32 accumulator would hold can overflow.
"""
import numpy as np

INT32_MAX = 2**31 - 1


def _conv3x3(x, node):
    k = node.weights
    m = x.shape[2] - 2
    s = np.zeros((x.shape[0], k.shape[0], m, m), dtype=np.int64)
    bound = np.zeros_like(s)
    for dy in range(3):
        for dx in range(3):
            t = k[:, dy, dx][None, :, None, None] * x[:, :, dy:dy + m, dx:dx + m]   # broadcast if C_in == 1
            s += t
            bound += np.abs(t)
    assert bound.max() <= INT32_MAX, f'{node.name}: int32 overflow possible'
    return np.where(s < 0, 0, s >> node.attrs['shift'])


def _relu_norm(x):
    x = x.reshape(x.shape[0], -1)
    mx = x.max(axis=1, keepdims=True)
    s, shift = mx >> 7, np.zeros_like(mx)
    while (s > 0).any():
        shift += s > 0
        s = np.where(s > 0, s >> 1, s)
    rnd = (np.int64(1) << shift) >> 1
    return np.where(x < 0, 0, np.minimum((x + rnd) >> shift, 127))


def _linear(x, node):
    w = node.weights
    assert (np.abs(x) @ np.abs(w).T).max() <= INT32_MAX, f'{node.name}: int32 overflow possible'
    return x @ w.T


def _fff(x, node):
    t, d = node.tensors, node.attrs['depth']
    n = np.zeros(x.shape[0], dtype=np.int64)
    for m in range(d):
        w = t['nodes'][2 ** m - 1 + n]                                     # [N, n_in], the node each image is at
        n = 2 * n + ((w * x).sum(axis=1) >= 0)
    win, wout = t['leaf_in'][n], t['leaf_out'][n]
    assert (np.abs(win) * np.abs(x)[:, None, :]).sum(axis=2).max() <= INT32_MAX, f'{node.name}: int32 overflow possible'
    h = _relu_norm((win * x[:, None, :]).sum(axis=2))
    return (wout * h[:, None, :]).sum(axis=2)


def run(graph, imgs):
    x = None
    logits = None
    for node in graph.nodes:
        if node.op == 'input':
            x = imgs.astype(np.int64).reshape((-1,) + tuple(node.out_shape))
        elif node.op == 'conv3x3':
            x = _conv3x3(x, node)
        elif node.op == 'maxpool2':
            n, c, h, w = x.shape
            x = x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))
        elif node.op == 'relu_norm':
            x = _relu_norm(x)
        elif node.op == 'linear':
            x = _linear(x.reshape(x.shape[0], -1), node)
            logits = x
        elif node.op == 'fff':
            x = _fff(x.reshape(x.shape[0], -1), node)
        elif node.op == 'argmax':
            pass
        else:
            raise ValueError(node.op)
    return logits
