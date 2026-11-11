"""Golden model: numpy emulation of firmware/baselines/ref_c, read straight from an exported model header.

It decodes the same packed weights the firmware uses and repeats the same integer arithmetic, so its
logits must match the C reference bit for bit. It also checks that no intermediate value can overflow
int32 (the C code accumulates in int32).
"""
import re
from pathlib import Path

import numpy as np

INT32_MAX = 2**31 - 1
CONV_SHIFT = 4


def parse_header(path):
    """Return ({define: value}, {array name: np.int64 array}) from a BitNetMCU model header."""
    text = Path(path).read_text()
    defines = {}
    for name, val in re.findall(r'^#define\s+(\w+)[ \t]+(\S+)', text, re.M):
        try:
            defines[name] = int(val, 0)
        except ValueError:
            defines[name] = val
    arrays = {}
    for ctype, name, body in re.findall(r'const\s+(\w+)\s+(\w+)\[\]\s*=\s*\{(.*?)\};', text, re.S):
        vals = [int(v, 0) for v in body.replace('\n', ' ').split(',') if v.strip()]
        arrays[name] = np.array(vals, dtype=np.int64)
    return defines, arrays


def decode_ternary(words, n_in, n_out):
    """uint16 words, 10 trits each as a base-3 fraction -> [n_out, n_in] in {-1, 0, +1}."""
    c = words.reshape(n_out, n_in // 10).copy()
    w = np.zeros((n_out, n_in // 10, 10), dtype=np.int64)
    for j in range(10):
        c *= 3
        d = (c >> 16) & 3
        w[:, :, j] = np.where(d == 0, 1, np.where(d == 1, -1, 0))
        c &= 0xFFFF
    return w.reshape(n_out, n_in)


def decode_4bitsym(words, n_in, n_out):
    """uint32 words, 8 nibbles each (MSB first, bit3 = sign, magnitude 2*m+1) -> [n_out, n_in]."""
    wd = (words & 0xFFFFFFFF).reshape(n_out, n_in // 8)
    w = np.zeros((n_out, n_in // 8, 8), dtype=np.int64)
    for j in range(8):
        nib = (wd >> (28 - 4 * j)) & 0xF
        w[:, :, j] = (2 * (nib & 7) + 1) * np.where(nib & 8, -1, 1)
    return w.reshape(n_out, n_in)


class Golden:
    def __init__(self, header):
        d, a = parse_header(header)
        self.name = Path(header).stem
        assert re.search(r'^#define\s+MODEL_CNNMNIST\b', Path(header).read_text(), re.M), 'CNNMNIST layout only'
        assert d['L11_bitperweight'] == 64 and d['L13_bitperweight'] == 4 and d['L15_bitperweight'] == 4
        self.n_ch = d['L7_out_channels']
        self.k1 = a['L2_weights'].reshape(self.n_ch, 3, 3)
        self.k2 = a['L4_weights'].reshape(self.n_ch, 3, 3)
        self.k3 = a['L7_weights'].reshape(self.n_ch, 3, 3)
        self.fc1 = decode_ternary(a['L11_weights'], d['L11_incoming_weights'], d['L11_outgoing_weights'])
        self.fc2 = decode_4bitsym(a['L13_weights'], d['L13_incoming_weights'], d['L13_outgoing_weights'])
        self.fc3 = decode_4bitsym(a['L15_weights'], d['L15_incoming_weights'], d['L15_outgoing_weights'])
        self.max_abs_bound = 0  # largest possible |partial sum| seen, to prove int32 is safe

    def _bound(self, v):
        self.max_abs_bound = max(self.max_abs_bound, int(np.abs(v).max()))
        assert self.max_abs_bound <= INT32_MAX, 'int32 overflow possible'

    def _conv(self, x, k):
        """x [N, C, n, n] int64, k [C, 3, 3] -> relu(conv) >> CONV_SHIFT, [N, C, n-2, n-2]."""
        m = x.shape[2] - 2
        s = np.zeros((x.shape[0], self.n_ch, m, m), dtype=np.int64)
        bound = np.zeros_like(s)
        for dy in range(3):
            for dx in range(3):
                kk = k[:, dy, dx][None, :, None, None]
                term = kk * x[:, :, dy:dy + m, dx:dx + m]
                s += term
                bound += np.abs(term)
        self._bound(bound)
        return np.where(s < 0, 0, s >> CONV_SHIFT)

    @staticmethod
    def _pool(x):
        n, c, h, w = x.shape
        return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))

    @staticmethod
    def _relu_norm(x):
        """Per row: smallest power-of-two shift that brings max below 128, round half up, clip to 0..127."""
        mx = x.max(axis=1, keepdims=True)
        s = mx >> 7
        shift = np.zeros_like(mx)
        while (s > 0).any():
            shift += (s > 0)
            s = np.where(s > 0, s >> 1, s)
        rnd = (np.int64(1) << shift) >> 1
        y = np.minimum((x + rnd) >> shift, 127)
        return np.where(x < 0, 0, y)

    def _fc(self, a, w):
        self._bound(np.abs(a) @ np.abs(w).T)
        return a @ w.T

    def logits(self, imgs):
        """imgs [N, 256] int8 -> logits [N, 10] int64."""
        x = imgs.astype(np.int64).reshape(-1, 1, 16, 16)
        x = np.broadcast_to(x, (x.shape[0], self.n_ch, 16, 16))
        x = self._conv(x, self.k1)
        x = self._conv(x, self.k2)
        x = self._pool(x)
        x = self._conv(x, self.k3)
        x = self._pool(x)
        a = self._relu_norm(x.reshape(x.shape[0], -1))
        a = self._relu_norm(self._fc(a, self.fc1))
        a = self._relu_norm(self._fc(a, self.fc2))
        return self._fc(a, self.fc3)


# ---------------------------------------------------------------------------------------------
# CMSIS-NN int8 baseline (firmware/baselines/cmsis_nn), built with CMSIS_NN_USE_SINGLE_ROUNDING.
# Same integer weight codes as above; activations are int8 with static, calibrated per-layer scales.
# ---------------------------------------------------------------------------------------------

def requant_single_rounding(acc, mult, shift, lo, hi):
    """CMSIS-NN arm_nn_requantize with CMSIS_NN_USE_SINGLE_ROUNDING: round(acc * mult / 2^(31 - shift)), half up."""
    v = (acc.astype(np.int64) * np.int64(mult)) >> (30 - shift)
    v = (v + 1) >> 1
    return np.clip(v, lo, hi)


def conv3x3_valid(x, k):
    """x [N, C or 1, n, n], k [C, 3, 3] (depthwise, or broadcast single input channel) -> int64 sums."""
    m = x.shape[2] - 2
    s = np.zeros((x.shape[0], k.shape[0], m, m), dtype=np.int64)
    for dy in range(3):
        for dx in range(3):
            s += k[:, dy, dx][None, :, None, None] * x[:, :, dy:dy + m, dx:dx + m]
    return s


def pool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


class GoldenS8:
    """numpy emulation of the CMSIS-NN build, read from its generated header (model_s8_<name>.h)."""

    LAYERS = ('CONV1', 'DW1', 'DW2', 'FC1', 'FC2', 'FC3')

    def __init__(self, header):
        d, a = parse_header(header)
        self.name = Path(header).stem
        c = d['S8_CH']
        self.k1 = a['s8_conv1_w'].reshape(c, 3, 3)                         # OHWI with I = 1
        self.k2 = a['s8_dw1_w'].reshape(3, 3, c).transpose(2, 0, 1)         # HWC -> CHW
        self.k3 = a['s8_dw2_w'].reshape(3, 3, c).transpose(2, 0, 1)
        fc1 = a['s8_fc1_w'].reshape(d['S8_FC1_OUT'], -1)                    # columns in NHWC order
        # back to NCHW column order: nhwc index (y*2+x)*C + ch  <->  nchw index ch*4 + y*2 + x
        idx = np.array([(p * c + ch) for ch in range(c) for p in range(4)])
        self.fc1 = fc1[:, idx]
        self.fc2 = a['s8_fc2_w'].reshape(d['S8_FC2_OUT'], -1)
        self.fc3 = a['s8_fc3_w'].reshape(d['S8_FC3_OUT'], -1)
        self.q = {l: (d[f'S8_{l}_MULT'], d[f'S8_{l}_SHIFT']) for l in self.LAYERS}

    def _rq(self, acc, layer, lo=0, hi=127):
        m, s = self.q[layer]
        return requant_single_rounding(acc, m, s, lo, hi)

    def logits(self, imgs):
        """imgs [N, 256] int8 -> int8 logits [N, 10] (as int64)."""
        x = imgs.astype(np.int64).reshape(-1, 1, 16, 16)
        x = self._rq(conv3x3_valid(x, self.k1), 'CONV1')
        x = self._rq(conv3x3_valid(x, self.k2), 'DW1')
        x = pool2(x)
        x = self._rq(conv3x3_valid(x, self.k3), 'DW2')
        x = pool2(x).reshape(x.shape[0], -1)
        x = self._rq(x @ self.fc1.T, 'FC1')
        x = self._rq(x @ self.fc2.T, 'FC2')
        return self._rq(x @ self.fc3.T, 'FC3', -128, 127)
