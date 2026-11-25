"""Back end: mafc IR -> one C file for the RP2350 (Cortex-M33), implementing ref_infer() from
firmware/baselines/ref_c/ref_infer.h, so the same pico_main.c, harness and golden models apply.

v0 strategy ("all-data"):
  - every layer becomes its own C function with all sizes as literals (no runtime shape logic);
  - weights are const arrays in a packed format chosen per quant type (ternary: 10 trits per uint16,
    int4sym: 8 nibbles per uint32, int8: bytes), placed in flash unless the build copies to RAM;
  - activations live in static buffers at fixed addresses, sized at compile time;
  - a channel-fusion pass runs the leading run of per-channel ops (broadcast/depthwise conv, pool)
    one channel at a time, so only two small per-channel planes are needed;
  - each conv/pool/linear node is a profiling stage (relu_norm counts toward the linear after it).
"""
from pathlib import Path

import numpy as np

from . import __version__
from .bounds import bounds, fits_int16
from .ir import Node

TERNARY_DIGIT = {1: 0, -1: 1, 0: 2}   # base-3 digit for each weight value (matches the decoder below)


# --------------------------------------------------------------------------- weight packing

def pack_ternary(w):
    """[n_out, n_in] in {-1,0,1} -> uint16 words, 10 trits each, rows padded with zeros.
    Word = ceil(N * 2^16 / 3^10) for the base-3 number N, so the decoder (c *= 3; digit = c >> 16) is exact."""
    n_out, n_in = w.shape
    pad = (-n_in) % 10
    wp = np.concatenate([w, np.zeros((n_out, pad), dtype=w.dtype)], axis=1)
    d = np.vectorize(TERNARY_DIGIT.get)(wp).reshape(n_out, -1, 10).astype(np.int64)
    n = (d * (3 ** np.arange(9, -1, -1))).sum(axis=2)
    words = (n * 65536 + 59048) // 59049
    assert words.max() < 65536
    return words.reshape(-1), n_in + pad


def unpack_ternary(words, n_in_padded, n_out):
    c = words.reshape(n_out, -1).astype(np.int64)
    out = np.zeros((n_out, c.shape[1], 10), dtype=np.int64)
    for j in range(10):
        c = c * 3
        dg = (c >> 16) & 3
        out[:, :, j] = np.where(dg == 0, 1, np.where(dg == 1, -1, 0))
        c &= 0xFFFF
    return out.reshape(n_out, n_in_padded)


def pack_int4sym(w):
    """[n_out, n_in] odd in -15..15 -> uint32 words, 8 nibbles MSB first: bit3 = sign, bits2:0 = (|w|-1)/2."""
    n_out, n_in = w.shape
    assert n_in % 8 == 0, 'int4sym rows must be a multiple of 8 (no zero code to pad with)'
    nib = np.where(w < 0, 8, 0) | ((np.abs(w) - 1) // 2)
    nib = nib.reshape(n_out, -1, 8).astype(np.uint64)
    words = np.zeros(nib.shape[:2], dtype=np.uint64)
    for j in range(8):
        words |= nib[:, :, j] << np.uint64(28 - 4 * j)
    return words.reshape(-1)


def unpack_int4sym(words, n_in, n_out):
    wd = words.reshape(n_out, -1).astype(np.int64)
    out = np.zeros((n_out, wd.shape[1], 8), dtype=np.int64)
    for j in range(8):
        nib = (wd >> (28 - 4 * j)) & 0xF
        out[:, :, j] = (2 * (nib & 7) + 1) * np.where(nib & 8, -1, 1)
    return out.reshape(n_out, n_in)


# --------------------------------------------------------------------------- C text helpers

def c_array(ctype, name, vals, fmt='{}', per_line=16):
    vals = [int(v) for v in np.asarray(vals).ravel()]
    lines = [', '.join(fmt.format(v) for v in vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f'static const {ctype} {name}[{len(vals)}] = {{\n    ' + ',\n    '.join(lines) + '\n};\n'


# --------------------------------------------------------------------------- per-op code

def relu_shift(v, sh, ctype):
    """ReLU then >> sh, written clamp-first. Same result as `v < 0 ? 0 : v >> sh` (0 >> sh == 0), but gcc
    compiles it branch-free (`bic r, r, r, asr #31`). The select form became a branch per output when the
    store was int16 (notes/compiler.md, "Why conv1 got slower")."""
    return f'({ctype})(({v} > 0 ? {v} : 0) >> {sh})'


def emit_conv_ch(node, in_ctype, out_ctype='int32_t'):
    """One channel of a 3x3 valid conv + ReLU + shift. in: one n x n plane, out: (n-2) x (n-2)."""
    n = node.in_shape[1]
    m = n - 2
    sh = node.attrs['shift']
    return f'''// {node.name}: conv3x3 ({node.attrs['mode']}), one channel, {n}x{n} -> {m}x{m}, ReLU, >> {sh}
static inline void {node.name}_ch(const {in_ctype} *restrict in, const int8_t *restrict k, {out_ctype} *restrict out) {{
    const int32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4], k5 = k[5], k6 = k[6], k7 = k[7], k8 = k[8];
    for (int y = 0; y < {m}; y++) {{
        const {in_ctype} *r0 = in + y * {n}, *r1 = r0 + {n}, *r2 = r1 + {n};
        {out_ctype} *o = out + y * {m};
        for (int x = 0; x < {m}; x++) {{
            int32_t s = k0 * r0[x] + k1 * r0[x + 1] + k2 * r0[x + 2]
                      + k3 * r1[x] + k4 * r1[x + 1] + k5 * r1[x + 2]
                      + k6 * r2[x] + k7 * r2[x + 1] + k8 * r2[x + 2];
            o[x] = {relu_shift('s', sh, out_ctype)};
        }}
    }}
}}
'''


def emit_pool_ch(node, in_ctype='int32_t', out_ctype='int32_t'):
    n = node.in_shape[1]
    m = n // 2
    return f'''// {node.name}: maxpool 2x2, one channel, {n}x{n} -> {m}x{m}
static inline void {node.name}_ch(const {in_ctype} *restrict in, {out_ctype} *restrict out) {{
    for (int y = 0; y < {m}; y++) {{
        const {in_ctype} *r0 = in + 2 * y * {n}, *r1 = r0 + {n};
        for (int x = 0; x < {m}; x++) {{
            int32_t a = r0[2 * x], b = r0[2 * x + 1], c = r1[2 * x], d = r1[2 * x + 1];
            int32_t ab = a > b ? a : b, cd = c > d ? c : d;
            out[y * {m} + x] = ab > cd ? ab : cd;
        }}
    }}
}}
'''


def pack_smlad_kernel(k):
    """[C, 3, 3] int8 -> [C, 6] uint32: per kernel row r the halfword pairs A_r = (k[r][0], k[r][1]) and
    B_r = (k[r][1], k[r][2]) (first element in the low half): A_0, A_1, A_2, B_0, B_1, B_2."""
    k = k.astype(np.int64)
    a = ((k[:, :, 1] & 0xFFFF) << 16) | (k[:, :, 0] & 0xFFFF)
    b = ((k[:, :, 2] & 0xFFFF) << 16) | (k[:, :, 1] & 0xFFFF)
    return np.concatenate([a, b], axis=1)


def emit_conv_ch_smlad(node, out_ctype, ch=None, kwords=None, shared=False):
    """One channel of a 3x3 conv on an int16 plane, two outputs per step with SMLAD.
    For outputs x and x+1 of one kernel row r (inputs i0..i3 = in[x..x+3], words p = (i0, i1), q = (i2, i3),
    kernel words A = (k0, k1), B = (k1, k2)):
      s0 += i0*k0 + i1*k1  (SMLAD p, A)    + i2*k2          (SMLABT q, B)
      s1 += i1*k0          (SMLATB p, A)   + i2*k1 + i3*k2  (SMLAD q, B)
    4 MACs and 2 loads per row, no repacking. Row 0 starts the sums with SMUAD (no zero register).
    Exact: the inputs are proven to fit int16 and the partial sums int32 (bounds.py).
    ReLU + shift is one USAT: s < 0 gives 0, and the upper clamp is never reached (output range proven).
    An int16 output pair is stored as one word: both values are in [0, 2^15), so v0 | v1 << 16 is exactly the
    two int16s side by side (little-endian).
    The x loop is kept rolled (`#pragma GCC unroll 1`): unrolled, gcc hoists the loads and spills to the stack.
    ch/kwords: the weights-as-code variant (--conv code): one function per channel, named <node>_ch<ch>,
    with its 6 kernel words as immediates instead of loaded from <node>_kp."""
    n = node.in_shape[1]
    m = n - 2
    sh = node.attrs['shift']
    assert n % 2 == 0 and m % 2 == 0, 'SMLAD kernel needs an even plane width (aligned word pairs)'
    rows = []
    for r in range(3):
        s0 = f'SMUAD(p, a{r})' if r == 0 else f'SMLAD(p, a{r}, s0)'
        s1 = f'SMUAD(q, b{r})' if r == 0 else f'SMLAD(q, b{r}, s1)'
        rows.append(f'''            {{
                uint32_t p, q;
                memcpy(&p, row + {r * n} + x, 4);
                memcpy(&q, row + {r * n} + x + 2, 4);
                s0 = {s0};
                s0 = SMLABT(q, b{r}, s0);
                s1 = {s1};
                s1 = SMLATB(p, a{r}, s1);
            }}''')
    body = '\n'.join(rows)
    bits = 15 if out_ctype == 'int16_t' else 31
    v0, v1 = f'USAT(s0 >> {sh}, {bits})', f'USAT(s1 >> {sh}, {bits})'
    if out_ctype == 'int16_t':
        store = (f"            uint32_t w = (uint32_t){v0} | ((uint32_t){v1} << 16);\n"
                 f"            memcpy(o + x, &w, 4);")
    else:
        store = (f"            o[x] = ({out_ctype}){v0};\n"
                 f"            o[x + 1] = ({out_ctype}){v1};")
    if ch is None:
        # shared (two-core builds): noinline, one copy for both cores. Inlined, gcc compiled core 1's copy (inside
        # core1_main) with spills (dw1 loop 30 instructions vs 24), and core 0 waited ~24k cycles at the join (F19).
        # One-core builds keep it inline (the call costs ~1%).
        qual = '__attribute__((noinline)) static void' if shared else 'static inline void'
        head = (f'{qual} {node.name}_ch(const int16_t *restrict in, const uint32_t *restrict kp, '
                f'{out_ctype} *restrict out) {{\n'
                '    const uint32_t a0 = kp[0], a1 = kp[1], a2 = kp[2], b0 = kp[3], b1 = kp[4], b2 = kp[5];')
        title = f'// {node.name}: conv3x3 ({node.attrs["mode"]}), one channel'
    else:
        kw = [f'0x{int(v):08X}u' for v in kwords]
        head = (f'static void {node.name}_ch{ch}(const int16_t *restrict in, {out_ctype} *restrict out) {{\n'
                f'    const uint32_t a0 = {kw[0]}, a1 = {kw[1]}, a2 = {kw[2]}, b0 = {kw[3]}, b1 = {kw[4]}, b2 = {kw[5]};')
        title = f'// {node.name}: conv3x3 ({node.attrs["mode"]}), channel {ch}, weights as code'
    return f'''{title}, int16 {n}x{n} -> {m}x{m}, ReLU, >> {sh}; SMLAD, 2 outputs per step
{head}
    for (int y = 0; y < {m}; y++) {{
        const int16_t *row = in + y * {n};
        {out_ctype} *o = out + y * {m};
        #pragma GCC unroll 1
        for (int x = 0; x < {m}; x += 2) {{
            int32_t s0, s1;
{body}
{store}
        }}
    }}
}}
'''


def emit_conv_pool_ch_smlad(node, pool, out_ctype, shared=False):
    """One channel of a 3x3 conv on an int16 plane fused with the 2x2 max pool that follows it (--fuse-pool).
    Each step computes the 2x2 block of conv sums under one pooled output, two at a time with the same SMLAD
    rows as emit_conv_ch_smlad (row y, then row y+1), keeps their maximum, and applies ReLU + shift once.
    Exact: v -> USAT(v >> sh) is monotone non-decreasing, so max over the block of USAT(s >> sh) equals
    USAT(max(s) >> sh). The conv output plane is never stored, and the pool pass disappears."""
    n = node.in_shape[1]
    m = n - 2
    mp = m // 2
    sh = node.attrs['shift']
    assert n % 2 == 0 and m % 2 == 0 and pool.in_shape[1] == m
    def rows(base):
        out = []
        for r in range(3):
            s0 = f'SMUAD(p, a{r})' if r == 0 else f'SMLAD(p, a{r}, s0)'
            s1 = f'SMUAD(q, b{r})' if r == 0 else f'SMLAD(q, b{r}, s1)'
            out.append(f'''            {{
                uint32_t p, q;
                memcpy(&p, r + {base + r * n}, 4);
                memcpy(&q, r + {base + r * n} + 2, 4);
                s0 = {s0};
                s0 = SMLABT(q, b{r}, s0);
                s1 = {s1};
                s1 = SMLATB(p, a{r}, s1);
            }}''')
        return '\n'.join(out)
    bits = 15 if out_ctype == 'int16_t' else 31
    qual = '__attribute__((noinline)) static void' if shared else 'static inline void'
    steps = f'''{rows(0)}
            mx = s0 > s1 ? s0 : s1;
{rows(n)}
            s0 = s0 > s1 ? s0 : s1;
            mx = mx > s0 ? mx : s0;'''
    # Rows y and y+1 read input rows 1 and 2 at the same addresses; gcc already loads them once (no stores in
    # between), so a hand-interleaved two-row version measured the same cycle count (notes/compiler.md, step 11).
    return f'''// {node.name} + {pool.name}: conv3x3 ({node.attrs["mode"]}) fused with maxpool 2x2, one channel, int16 {n}x{n} -> {mp}x{mp}, ReLU, >> {sh}
{qual} {node.name}_{pool.name}_ch(const int16_t *restrict in, const uint32_t *restrict kp, {out_ctype} *restrict out) {{
    const uint32_t a0 = kp[0], a1 = kp[1], a2 = kp[2], b0 = kp[3], b1 = kp[4], b2 = kp[5];
    for (int y = 0; y < {mp}; y++) {{
        const int16_t *row = in + y * {2 * n};
        {out_ctype} *o = out + y * {mp};
        #pragma GCC unroll 1
        for (int x = 0; x < {mp}; x++) {{
            const int16_t *r = row + 2 * x;
            int32_t s0, s1, mx;
{steps}
            o[x] = ({out_ctype})USAT(mx >> {sh}, {bits});
        }}
    }}
}}
'''


def emit_conv_ch_smlad_rows2(node, shared=False):
    """emit_conv_ch_smlad with two output rows per step (--row-pairs): outputs (y, x..x+1) and (y+1, x..x+1).
    Input rows y+1 and y+2 feed both output rows, so 4 row loads serve 4 outputs instead of 6. Order: input row 0
    starts the top sums, row 1 adds to the top and starts the bottom, row 2 finishes and stores the top, row 3
    finishes the bottom. int16 output only (stored as word pairs); needs an even output height."""
    n = node.in_shape[1]
    m = n - 2
    sh = node.attrs['shift']
    assert n % 2 == 0 and m % 2 == 0
    def mac(t, kr, first):
        a, b = f'a{kr}', f'b{kr}'
        s0 = f'SMUAD(p, {a})' if first else f'SMLAD(p, {a}, {t}0)'
        s1 = f'SMUAD(q, {b})' if first else f'SMLAD(q, {b}, {t}1)'
        return [f'{t}0 = {s0};', f'{t}0 = SMLABT(q, {b}, {t}0);', f'{t}1 = {s1};', f'{t}1 = SMLATB(p, {a}, {t}1);']

    def store(t, off):
        return [f'{{ uint32_t w = (uint32_t)USAT({t}0 >> {sh}, 15) | ((uint32_t)USAT({t}1 >> {sh}, 15) << 16);',
                f'  memcpy(o + {off}x, &w, 4); }}']
    lines = []
    for ir in range(4):
        blk = ['uint32_t p, q;', f'memcpy(&p, row + {ir * n} + x, 4);', f'memcpy(&q, row + {ir * n} + x + 2, 4);']
        if ir <= 2:
            blk += mac('t', ir, ir == 0)
        if ir >= 1:
            blk += mac('u', ir - 1, ir == 1)
        if ir == 2:
            blk += store('t', '')
        if ir == 3:
            blk += store('u', f'{m} + ')
        lines += ['            {'] + ['                ' + l for l in blk] + ['            }']
    body = '\n'.join(lines) + '\n'
    qual = '__attribute__((noinline)) static void' if shared else 'static inline void'
    return f'''// {node.name}: conv3x3 ({node.attrs["mode"]}), one channel, int16 {n}x{n} -> {m}x{m}, ReLU, >> {sh}; SMLAD, 2 rows x 2 outputs per step
{qual} {node.name}_ch(const int16_t *restrict in, const uint32_t *restrict kp, int16_t *restrict out) {{
    const uint32_t a0 = kp[0], a1 = kp[1], a2 = kp[2], b0 = kp[3], b1 = kp[4], b2 = kp[5];
    for (int y = 0; y < {m}; y += 2) {{
        const int16_t *row = in + y * {n};
        int16_t *o = out + y * {m};
        #pragma GCC unroll 1
        for (int x = 0; x < {m}; x += 2) {{
            int32_t t0, t1, u0, u1;
{body}        }}
    }}
}}
'''


IMG16_DECL = '''static int16_t img16[REF_IMG_PIXELS] __attribute__((aligned(4)));   // the image widened to int16, read by conv1's SMLAD kernel
'''

WIDEN_IMG = '''
// The image widened to int16 once per inference: conv1 is a broadcast conv, so every output channel reads
// the same plane and the SMLAD kernel needs (int16, int16) word pairs.
static inline void widen_img(const int8_t *restrict img) {
    for (int i = 0; i < REF_IMG_PIXELS; i++) img16[i] = img[i];
}
'''

SIMD_HELPERS = '''
// ---- DSP helpers: SMLAD/SMUAD/SMLABT/SMLATB/USAT/SXTB16 on the M33, plain C on the PC (same results) ----
#include <string.h>
#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP
#include <arm_acle.h>
#define SMLAD(a, b, acc)  __smlad((a), (b), (acc))
#define SMUAD(a, b)       __smuad((a), (b))
#define SMLABT(a, b, acc) __smlabt((int32_t)(a), (int32_t)(b), (acc))
#define SMLATB(a, b, acc) __smlatb((int32_t)(a), (int32_t)(b), (acc))
#define USAT(x, bits)     __usat((x), (bits))
#define SXTB16(x)         __sxtb16(x)
// gcc doesn't fold __ror into sxtb16 (one extra instruction), so the rotated form is written directly
static inline uint32_t SXTB16_ROR8(uint32_t x) { uint32_t r; __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(x)); return r; }
#else
static inline uint32_t SXTB16(uint32_t x) {
    return ((uint32_t)(uint16_t)(int16_t)(int8_t)x) | ((uint32_t)(uint16_t)(int16_t)(int8_t)(x >> 16) << 16);
}
static inline uint32_t SXTB16_ROR8(uint32_t x) { return SXTB16((x >> 8) | (x << 24)); }
static inline int32_t SMLAD(uint32_t a, uint32_t b, int32_t acc) {
    return acc + (int16_t)a * (int16_t)b + (int16_t)(a >> 16) * (int16_t)(b >> 16);
}
static inline int32_t SMUAD(uint32_t a, uint32_t b) { return SMLAD(a, b, 0); }
static inline int32_t SMLABT(uint32_t a, uint32_t b, int32_t acc) { return acc + (int16_t)a * (int16_t)(b >> 16); }
static inline int32_t SMLATB(uint32_t a, uint32_t b, int32_t acc) { return acc + (int16_t)(a >> 16) * (int16_t)b; }
static inline int32_t USAT(int32_t x, int bits) {
    const int32_t hi = (int32_t)((1u << bits) - 1);
    return x < 0 ? 0 : x > hi ? hi : x;
}
#endif
'''


def emit_relu_norm(node):
    n = node.out_shape[0]
    return f'''// {node.name}: ReLU + dynamic power-of-two requantization to 0..127, {n} values
static inline void {node.name}(const int32_t *restrict in, int8_t *restrict out) {{
    int32_t mx = -INT32_MAX;
    for (int i = 0; i < {n}; i++) if (in[i] > mx) mx = in[i];
    int shift = 0;
    for (int32_t s = mx >> 7; s > 0; s >>= 1) shift++;
    const int32_t rnd = (1 << shift) >> 1;
    for (int i = 0; i < {n}; i++) {{
        int32_t v = in[i];
        if (v < 0) {{ out[i] = 0; continue; }}
        v = (v + rnd) >> shift;
        out[i] = (int8_t)(v > 127 ? 127 : v);
    }}
}}
'''


def emit_linear(node, rows=False):
    """Returns (weights C text, function C text). rows=True: the function takes a row range (o0, o1)
    so two cores can each compute part of the outputs."""
    q = node.attrs['quant']
    w = node.weights
    n_out, n_in = w.shape
    if q == 'ternary':
        words, n_pad = pack_ternary(w)
        assert np.array_equal(unpack_ternary(words, n_pad, n_out)[:, :n_in], w), 'ternary packing round trip'
        data = c_array('uint16_t', f'{node.name}_w', words, '0x{:04X}', 12)
        chunks = n_pad // 10
        tail = n_in - (chunks - 1) * 10   # trits used in the last word of a row
        fn = f'''// {node.name}: linear {n_in} -> {n_out}, ternary weights, 10 trits per uint16 (base-3 fraction)
static inline void {node.name}(const int8_t *restrict in, int32_t *restrict out) {{
    const uint16_t *w = {node.name}_w;
    for (int o = 0; o < {n_out}; o++) {{
        const int8_t *a = in;
        int32_t sum = 0;
        for (int k = 0; k < {chunks}; k++) {{
            uint32_t c = *w++;
            const int nt = k == {chunks - 1} ? {tail} : 10;
            for (int j = 0; j < nt; j++) {{
                c *= 3;
                uint32_t d = c >> 16;
                if (d == 0) sum += a[j];
                else if (d == 1) sum -= a[j];
                c &= 0xFFFF;
            }}
            a += 10;
        }}
        out[o] = sum;
    }}
}}
'''
    elif q == 'int4sym':
        words = pack_int4sym(w)
        assert np.array_equal(unpack_int4sym(words, n_in, n_out), w), 'int4sym packing round trip'
        data = c_array('uint32_t', f'{node.name}_w', words, '0x{:08X}', 8)
        fn = f'''// {node.name}: linear {n_in} -> {n_out}, int4sym weights (+-1..+-15, odd), 8 per uint32
static inline void {node.name}(const int8_t *restrict in, int32_t *restrict out) {{
    const uint32_t *w = {node.name}_w;
    for (int o = 0; o < {n_out}; o++) {{
        const int8_t *a = in;
        int32_t sum = 0;
        for (int k = 0; k < {n_in // 8}; k++) {{
            uint32_t word = *w++;
            for (int j = 0; j < 8; j++) {{
                uint32_t nib = word >> 28;
                int32_t v = (int32_t)(2 * (nib & 7) + 1) * *a++;
                sum += (nib & 8) ? -v : v;
                word <<= 4;
            }}
        }}
        out[o] = sum;
    }}
}}
'''
    elif q == 'int8':
        data = c_array('int8_t', f'{node.name}_w', w)
        fn = f'''// {node.name}: linear {n_in} -> {n_out}, int8 weights
static inline void {node.name}(const int8_t *restrict in, int32_t *restrict out) {{
    const int8_t *w = {node.name}_w;
    for (int o = 0; o < {n_out}; o++) {{
        int32_t sum = 0;
        for (int i = 0; i < {n_in}; i++) sum += (int32_t)*w++ * in[i];
        out[o] = sum;
    }}
}}
'''
    else:
        raise ValueError(q)
    if rows:
        stride = {'ternary': (n_in + (-n_in) % 10) // 10, 'int4sym': n_in // 8, 'int8': n_in}[q]
        fn = fn.replace('(const int8_t *restrict in, int32_t *restrict out) {',
                        '(const int8_t *restrict in, int32_t *restrict out, int o0, int o1) {')
        fn = fn.replace(f'w = {node.name}_w;', f'w = {node.name}_w + o0 * {stride};')
        fn = fn.replace(f'for (int o = 0; o < {n_out}; o++)', 'for (int o = o0; o < o1; o++)')
        assert 'o0 * ' in fn and 'o = o0' in fn
    return data, fn


# --------------------------------------------------------------------------- graph-level

def channel_fused_prefix(nodes):
    """Indices of the leading run of per-channel ops (after the input) that can run one channel at a time."""
    seg = []
    for i, nd in enumerate(nodes[1:], start=1):
        if nd.op == 'conv3x3' or nd.op == 'maxpool2':
            seg.append(i)
        else:
            break
    return seg


PAR_MIN_MACS = 4096   # linear layers smaller than this stay on core 0 (the handshake would cost more)


TERNARY_FORMATS = ('base3', 'sparse', 'code', 'dense')
INT4_FORMATS = ('packed', 'code', 'dense')
INT8_FORMATS = ('plain', 'code', 'dense')   # plain = emit_linear's int8 loop
STREAM_BLOCK_ROWS = 8   # rows per streamed weight block (double-buffered)


DENSE_ROW_BLOCK = 4   # dense format: rows computed together, sharing each activation load (FINDINGS F24)


def dense_rows(nm, k, R=DENSE_ROW_BLOCK):
    """C text of {nm}_rows(w, out, n): n rows of a dense layer (k words of 4 int8 weights per row), R rows at a
    time so each pair of int16 activations is loaded once per R rows; then the leftover rows one at a time."""
    assert R >= 1 and R & (R - 1) == 0, 'row block must be a power of two'
    acc = ', '.join(f's{r} = 0' for r in range(R))
    body = ''.join(f'''            {{ uint32_t x = w[{r * k} + j];
              s{r} = SMLAD(SXTB16(x), a01, s{r});
              s{r} = SMLAD(SXTB16_ROR8(x), a23, s{r}); }}
''' for r in range(R))
    stores = ''.join(f'        out[o + {r}] = s{r};\n' for r in range(R))
    return f'''// {nm}_rows: n rows of weights starting at w. {R} rows at a time: each activation pair is loaded once and
// feeds all {R} rows (CMSIS-NN's vec_mat_mult does the same with 2 rows; our activations are already int16).
__attribute__((always_inline)) static inline void {nm}_rows(const uint32_t *restrict w, int32_t *restrict out, int n) {{
    int o = 0;
    for (; o + {R - 1} < n; o += {R}) {{
        const int16_t *a = {nm}_a16;
        int32_t {acc};
        #pragma GCC unroll 1
        for (int j = 0; j < {k}; j++) {{
            uint32_t a01, a23;
            memcpy(&a01, a + 4 * j, 4);
            memcpy(&a23, a + 4 * j + 2, 4);
{body}        }}
{stores}        w += {R} * {k};
    }}
    const int left = (n - o) & {R - 1};                           // leftover rows (< {R}), one at a time
    for (int r = 0; r < left; r++, o++) {{
        const int16_t *a = {nm}_a16;
        int32_t s0 = 0;
        for (int j = 0; j < {k}; j++) {{
            uint32_t x = w[j], a01, a23;
            memcpy(&a01, a + 4 * j, 4);
            memcpy(&a23, a + 4 * j + 2, 4);
            s0 = SMLAD(SXTB16(x), a01, s0);
            s0 = SMLAD(SXTB16_ROR8(x), a23, s0);
        }}
        out[o] = s0;
        w += {k};
    }}
}}
'''


def emit_linear_fmt(node, fmt, rows, stream=False):
    """A ternary, int4sym or int8 linear layer in one of the alternative formats (base3 / packed / plain are emit_linear). Returns
    (weights C text, function C text, C statement to run before the call or None, extra globals C text).
    Every function takes (in, out, o0, o1) when rows is True, else (in, out).
      sparse: per row, the input indices of the +1 weights then the -1 weights (uint16), summed with a loop.
      code:   idea A, the weights compiled into the instructions: one straight-line sum per row. For int4sym,
              the inputs are summed per weight magnitude first and each group is multiplied once:
              3 * (in[2] - in[9] ...) + 5 * (...) + ...  (one add/sub per weight, at most 8 multiplies per row).
      dense:  the weights as int8, 4 per word, unpacked with SXTB16 and multiplied with SMLAD against an int16
              copy of the input (widened once, before the call). Ternary, int4sym or int8.
              stream=True: the weights are streamed from flash in blocks by DMA (idea C, --stream).
    sparse is ternary only."""
    w = node.weights
    n_out, n_in = w.shape
    nm = node.name
    sig = '(const int8_t *restrict in, int32_t *restrict out, int o0, int o1)' if rows else \
          '(const int8_t *restrict in, int32_t *restrict out)'
    lo, hi = ('o0', 'o1') if rows else ('0', str(n_out))
    nnz = int(np.count_nonzero(w))
    q = node.attrs['quant']
    assert q in ('ternary', 'int4sym', 'int8') and (fmt != 'sparse' or q == 'ternary')
    if fmt == 'sparse':
        idx, start, mid = [], [], []
        for o in range(n_out):
            start.append(len(idx))
            idx += list(np.nonzero(w[o] == 1)[0])
            mid.append(len(idx))
            idx += list(np.nonzero(w[o] == -1)[0])
        start.append(len(idx))
        assert len(idx) == nnz and max(idx) < 2**16 and len(idx) < 2**16
        data = (c_array('uint16_t', f'{nm}_ix', idx, '{}', 16) + c_array('uint16_t', f'{nm}_start', start, '{}', 16) +
                c_array('uint16_t', f'{nm}_mid', mid, '{}', 16))
        fn = f'''// {nm}: linear {n_in} -> {n_out}, ternary as sparse index lists ({nnz} nonzeros: +1 indices, then -1 indices, per row)
static inline void {nm}{sig} {{
    for (int o = {lo}; o < {hi}; o++) {{
        const uint16_t *p = {nm}_ix + {nm}_start[o], *m = {nm}_ix + {nm}_mid[o], *e = {nm}_ix + {nm}_start[o + 1];
        int32_t s = 0;
        #pragma GCC unroll 4
        while (p < m) s += in[*p++];
        #pragma GCC unroll 4
        while (p < e) s -= in[*p++];
        out[o] = s;
    }}
}}
'''
        return data, fn, None, ''
    if fmt == 'code':
        lines = []
        for o in range(n_out):
            groups = []
            for mag in sorted({abs(int(v)) for v in w[o] if v}):
                terms = [f'+ in[{i}]' for i in np.nonzero(w[o] == mag)[0]] + \
                        [f'- in[{i}]' for i in np.nonzero(w[o] == -mag)[0]]
                groups.append((mag, ' '.join(terms).lstrip('+ ')))
            if not groups:
                expr = '0'
            elif len(groups) == 1 and groups[0][0] == 1:
                expr = groups[0][1]                      # ternary: a plain sum
            else:
                expr = ' + '.join(f'({g})' if mag == 1 else f'{mag} * ({g})' for mag, g in groups)
            text, line = [], ''          # wrap at about 110 columns, between tokens
            for tok in expr.split(' '):
                if len(line) + len(tok) > 110:
                    text.append(line)
                    line = ''
                line += (' ' if line else '') + tok
            text.append(line)
            body = ('\n' + ' ' * 12).join(text)
            guard = f'if ({o} >= o0 && {o} < o1) ' if rows else ''
            lines.append(f'    {guard}out[{o}] =\n            {body};')
        # The function always lives in SRAM (copied at boot, also in flash builds): run from flash, its ~90 KB of
        # straight-line code streams through the 16 KB XIP cache and is 12x slower than from SRAM (F16).
        what = 'one load and one add/sub each' if q == 'ternary' else \
            'summed per weight magnitude, then one multiply per magnitude'
        fn = f'''// {nm}: linear {n_in} -> {n_out}, {q} weights compiled into code (idea A): {nnz} nonzeros, {what}
#ifndef MAFC_IN_SRAM
#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
#include "pico.h"
#define MAFC_IN_SRAM(f) __not_in_flash_func(f)   // section .time_critical.<f>, copied to SRAM at boot
#else
#define MAFC_IN_SRAM(f) f
#endif
#endif
__attribute__((noinline)) static void MAFC_IN_SRAM({nm}){sig} {{
''' + '\n'.join(lines) + '\n}\n'
        return '', fn, None, ''
    if fmt == 'dense':
        assert n_in % 4 == 0, 'dense format needs n_in % 4 == 0'
        b = w.astype(np.int64).reshape(n_out, n_in // 4, 4) & 0xFF
        # bytes [w0, w2, w1, w3]: SXTB16(word) = (w0, w1), SXTB16(word ror 8) = (w2, w3)
        words = (b[:, :, 0] | (b[:, :, 2] << 8) | (b[:, :, 1] << 16) | (b[:, :, 3] << 24)).reshape(-1)
        k = n_in // 4
        data = c_array('uint32_t', f'{nm}_w', words, '0x{:08X}', 8)
        glob = f'static int16_t {nm}_a16[{n_in}] __attribute__((aligned(4)));   // {nm} input widened to int16\n'

        if stream:
            assert not rows, 'a streamed layer runs on core 0 only (the XIP stream FIFO is a single resource)'
            R = STREAM_BLOCK_ROWS
            nb = (n_out + R - 1) // R
            glob += f'static uint32_t {nm}_sbuf[2][{R * k}];   // {nm}: double buffer for streamed weight blocks\n'
            fn = f'''// {nm}: linear {n_in} -> {n_out}, {q} as int8, 4 per word (bytes w0 w2 w1 w3); SXTB16 + SMLAD on the int16 input.
// Streamed (idea C, FINDINGS F21): when the weights are in flash, the XIP stream FIFO feeds a DMA channel that
// fills one SRAM buffer with the next {R} rows while the CPU computes the previous {R}. When they are in SRAM
// (copy_to_ram build), the rows are read directly. On the PC, memcpy plays the DMA, so the block logic is checked.
#ifndef MAFC_IN_SRAM
#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
#include "pico.h"
#define MAFC_IN_SRAM(f) __not_in_flash_func(f)   // section .time_critical.<f>, copied to SRAM at boot
#else
#define MAFC_IN_SRAM(f) f
#endif
#endif
#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
#include "hardware/dma.h"
#include "hardware/structs/xip.h"
#include "hardware/structs/xip_aux.h"
#include "hardware/regs/addressmap.h"
#endif
static inline void {nm}_widen(const int8_t *restrict in) {{
    for (int i = 0; i < {n_in}; i++) {nm}_a16[i] = in[i];
}}
''' + dense_rows(nm, k) + f'''__attribute__((noinline)) static void MAFC_IN_SRAM({nm})(const int8_t *restrict in, int32_t *restrict out) {{
    (void)in;
#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
    const uintptr_t wa = (uintptr_t){nm}_w;
    if (wa >= XIP_BASE && wa < XIP_NOCACHE_NOALLOC_BASE) {{       // weights in flash: stream them
        static int ch = -1;
        if (ch < 0) ch = dma_claim_unused_channel(true);
        xip_ctrl_hw->stream_ctr = 0;
        while (!(xip_ctrl_hw->stat & XIP_STAT_FIFO_EMPTY_BITS)) (void)xip_aux_hw->stream;
        xip_ctrl_hw->stream_addr = (uint32_t)wa;
        xip_ctrl_hw->stream_ctr = {n_out * k};
        dma_channel_config c = dma_channel_get_default_config(ch);
        channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
        channel_config_set_read_increment(&c, false);
        channel_config_set_write_increment(&c, true);
        channel_config_set_dreq(&c, DREQ_XIP_STREAM);
        const void *src = (const void *)&xip_aux_hw->stream;
        dma_channel_configure(ch, &c, {nm}_sbuf[0], src, {min(R, n_out) * k}, true);
        for (int b = 0; b < {nb}; b++) {{
            const int r0 = b * {R}, nr = {n_out} - r0 < {R} ? {n_out} - r0 : {R};
            dma_channel_wait_for_finish_blocking(ch);
            if (b + 1 < {nb}) {{
                const int r1 = r0 + {R}, nr1 = {n_out} - r1 < {R} ? {n_out} - r1 : {R};
                dma_channel_configure(ch, &c, {nm}_sbuf[(b + 1) & 1], src, nr1 * {k}, true);
            }}
            {nm}_rows({nm}_sbuf[b & 1], out + r0, nr);
        }}
        return;
    }}
    {nm}_rows({nm}_w, out, {n_out});                          // weights in SRAM: read them directly
#else
    for (int b = 0; b < {nb}; b++) {{                          // PC: the same blocks, memcpy as the DMA
        const int r0 = b * {R}, nr = {n_out} - r0 < {R} ? {n_out} - r0 : {R};
        memcpy({nm}_sbuf[b & 1], {nm}_w + r0 * {k}, (size_t)nr * {k} * 4);
        {nm}_rows({nm}_sbuf[b & 1], out + r0, nr);
    }}
#endif
}}
'''
            return data, fn, f'{nm}_widen', glob
        fn = f'''// {nm}: linear {n_in} -> {n_out}, {q} as int8, 4 per word (bytes w0 w2 w1 w3); SXTB16 + SMLAD on the int16 input, 2 rows at a time
static inline void {nm}_widen(const int8_t *restrict in) {{
    for (int i = 0; i < {n_in}; i++) {nm}_a16[i] = in[i];
}}
''' + dense_rows(nm, k) + f'''static inline void {nm}{sig} {{
    (void)in;
    {nm}_rows({nm}_w + {lo} * {k}, out + {lo}, {hi} - {lo});
}}
'''
        return data, fn, f'{nm}_widen', glob
    raise ValueError(fmt)


def emit_fff(node, simd, ternary, int4, int8):
    """Idea B: a fast feedforward tree layer. Returns (weights C text, functions C text, globals C text).
    Routing nodes: int8 rows, SMLAD on an int16 copy of the input (simd) or a plain loop. Each leaf's two layers are
    ordinary linear layers in the format chosen for their weight type (--ternary / --int4 / --int8), one function per
    leaf, called through a table: only the reached leaf's code runs and only its weights are read."""
    nm, t, a = node.name, node.tensors, node.attrs
    d, L = a['depth'], a['leaf']
    n_in, n_out = int(np.prod(node.in_shape)), node.out_shape[0]
    fmt_of = {'ternary': ternary if ternary != 'base3' else None, 'int4sym': int4 if int4 != 'packed' else None,
              'int8': int8 if int8 != 'plain' else None}
    data, funcs, glob = [], [], ''
    # routing nodes
    assert a['quant_nodes'] == 'int8'
    if simd:
        assert n_in % 4 == 0
        b = t['nodes'].astype(np.int64).reshape(-1, n_in // 4, 4) & 0xFF
        words = (b[:, :, 0] | (b[:, :, 2] << 8) | (b[:, :, 1] << 16) | (b[:, :, 3] << 24)).reshape(-1)
        k = n_in // 4
        data.append(c_array('uint32_t', f'{nm}_nw', words, '0x{:08X}', 8))
        glob += f'static int16_t {nm}_a16[{n_in}] __attribute__((aligned(4)));   // {nm} input widened to int16\n'
        funcs.append(f"""// {nm}: routing node k of the tree (int8 weights, 4 per word, bytes w0 w2 w1 w3): logit = nodes[k] . in
static inline int32_t {nm}_node(const int8_t *restrict in, int k) {{
    (void)in;
    const uint32_t *w = {nm}_nw + k * {k};
    const int16_t *a = {nm}_a16;
    int32_t s = 0;
    for (int j = 0; j < {k}; j++) {{
        uint32_t ww = w[j], a01, a23;
        memcpy(&a01, a + 4 * j, 4);
        memcpy(&a23, a + 4 * j + 2, 4);
        s = SMLAD(SXTB16(ww), a01, s);
        s = SMLAD(SXTB16_ROR8(ww), a23, s);
    }}
    return s;
}}
""")
    else:
        data.append(c_array('int8_t', f'{nm}_nw', t['nodes'].reshape(-1)))
        funcs.append(f"""// {nm}: routing node k of the tree (int8 weights): logit = nodes[k] . in
static inline int32_t {nm}_node(const int8_t *restrict in, int k) {{
    const int8_t *w = {nm}_nw + k * {n_in};
    int32_t s = 0;
    for (int i = 0; i < {n_in}; i++) s += (int32_t)w[i] * in[i];
    return s;
}}
""")
    # leaves: two linear layers each, emitted by the ordinary linear emitters
    h_dense = False
    for part, key, qk, shp_in, shp_out in (('in', 'leaf_in', 'quant_in', n_in, L), ('out', 'leaf_out', 'quant_out', L, n_out)):
        q = a[qk]
        fmt = fmt_of[q]
        for j in range(2 ** d):
            pn = Node(f'{nm}_{part}{j}', 'linear', (shp_in,), (shp_out,), 'int32', {'quant': q}, t[key][j])
            if fmt:
                dd, f, pre, g = emit_linear_fmt(pn, fmt, rows=False)
                if fmt == 'dense':
                    # all leaves share one int16 copy of their input: the tree input for 'in', the hidden layer for 'out'
                    shared = f'{nm}_a16' if part == 'in' else f'{nm}_h16'
                    f = f.replace(f'{pn.name}_a16', shared)
                    if part == 'out':
                        h_dense = True
                    assert part == 'out' or simd
            else:
                if q == 'int4sym':
                    assert shp_in % 8 == 0, f'{nm}: packed int4 needs a leaf input width divisible by 8; use --int4 code or dense'
                dd, f = emit_linear(pn)
            data.append(dd)
            funcs.append(f)
        funcs.append(f'static void (*const {nm}_{part}_tab[{2 ** d}])(const int8_t *restrict, int32_t *restrict) = {{\n    ' +
                     ', '.join(f'{nm}_{part}{j}' for j in range(2 ** d)) + '};\n')
    if h_dense:
        glob += f'static int16_t {nm}_h16[{L}] __attribute__((aligned(4)));   // {nm} leaf hidden layer widened to int16\n'
    glob += f'static int32_t {nm}_h[{L}];\nstatic int8_t {nm}_h8[{L}];\n'
    funcs.append(emit_relu_norm(Node(f'{nm}_hn', 'relu_norm', (L,), (L,), 'int8')))
    widen = f'    for (int i = 0; i < {n_in}; i++) {nm}_a16[i] = in[i];   // for the routing nodes and dense leaves\n' if simd else ''
    hwiden = f'    for (int i = 0; i < {L}; i++) {nm}_h16[i] = {nm}_h8[i];\n' if h_dense else ''
    funcs.append(f"""// {nm}: fast feedforward tree (idea B): depth {d}, {2 ** d} leaves of {L} neurons, {n_in} -> {n_out}.
// The path takes {d} routing dot products; then only the reached leaf runs: {n_in} -> {L} (relu_norm) -> {n_out}.
static void {nm}(const int8_t *restrict in, int32_t *restrict out) {{
{widen}    int n = 0;
    for (int m = 0; m < {d}; m++) n = 2 * n + ({nm}_node(in, (1 << m) - 1 + n) >= 0);
    {nm}_in_tab[n](in, {nm}_h);
    {nm}_hn({nm}_h, {nm}_h8);
{hwiden}    {nm}_out_tab[n]({nm}_h8, out);
}}
""")
    return ''.join(data), '\n'.join(funcs), glob


CONV_FORMATS = ('data', 'code')


def generate(graph, source='', cores=1, simd=False, ternary='base3', int4='packed', conv='data', stream=False,
             int8='plain', fuse_pool=False, row_pairs=False):
    """cores=2: the channel-fused conv segment and the big linear layers are split across both M33 cores.
    Core 1 runs a job loop fed by the SIO FIFO; on the PC build the core-1 jobs simply run inline first."""
    assert cores in (1, 2)
    nodes = graph.nodes
    assert nodes[0].op == 'input' and nodes[0].in_shape[0] == 1 and nodes[0].in_shape[1:] == (16, 16), \
        'this back end expects a 1x16x16 int8 input (the harness protocol)'
    seg = channel_fused_prefix(nodes)
    assert seg and nodes[seg[0]].op == 'conv3x3' and nodes[seg[0]].attrs['mode'] == 'broadcast'
    assert all(nodes[i].op != 'conv3x3' or nodes[i].attrs['mode'] == 'depthwise' for i in seg[1:])
    rest = nodes[seg[-1] + 1:]
    assert rest[-1].op == 'argmax' and rest[-2].op == 'linear' and rest[-2].out_shape == (10,), \
        'this back end expects the graph to end in linear(->10) + argmax'

    n_ch = nodes[seg[0]].out_shape[0]
    plane = max(int(np.prod(nodes[i].out_shape[1:])) for i in seg)
    feat_ch = int(np.prod(nodes[seg[-1]].out_shape[1:]))
    par_linear = set()
    if cores == 2:
        par_linear = {nd.name for nd in rest if nd.op == 'linear' and nd.weights.size >= PAR_MIN_MACS}
    # --stream: layers in a dense format stream their weights from flash (idea C); one stream FIFO, so core 0 only
    streamed = set()
    if stream:
        streamed = {nd.name for nd in rest if nd.op == 'linear' and
                    {'ternary': ternary, 'int4sym': int4, 'int8': int8}.get(nd.attrs['quant']) == 'dense'}
        assert streamed, '--stream needs a dense-format layer (--ternary/--int4/--int8 dense)'
        par_linear -= streamed
    # Typed plan for the fused segment. With simd, a tensor is stored as int16 when bounds.py proves it
    # fits, and a depthwise conv reading an int16 plane uses the SMLAD kernel. The last op writes feat (int32).
    bd = bounds(graph) if simd else {}
    plan = []          # (node, in_ctype, out_ctype, kernel)
    in_t = 'int8_t'
    for j, i in enumerate(seg):
        nd = nodes[i]
        last = j == len(seg) - 1
        out_t = 'int16_t' if simd and not last and fits_int16(bd[nd.name]) else 'int32_t'
        if nd.op == 'maxpool2':
            out_t = in_t if not last else 'int32_t'
        kern = 'plain'
        even = nd.in_shape[1] % 2 == 0 and (nd.in_shape[1] - 2) % 2 == 0
        if simd and nd.op == 'conv3x3' and even and (
                (nd.attrs['mode'] == 'depthwise' and in_t == 'int16_t') or
                (nd.attrs['mode'] == 'broadcast' and in_t == 'int8_t')):
            # broadcast: the int8 image is widened to int16 once per inference (img16), then every
            # output channel runs the same SMLAD kernel on it
            assert bd[nd.name][2] <= 2**31 - 1
            kern = 'smlad'
        plan.append((nd, in_t, out_t, kern))
        in_t = out_t
    # --fuse-pool: an SMLAD conv followed by a 2x2 max pool becomes one kernel (emit_conv_pool_ch_smlad); the
    # merged entry takes the pool's output type, and the pool node is remembered in fused_pool
    fused_pool = {}
    if fuse_pool:
        assert simd and conv == 'data', '--fuse-pool fuses the SMLAD conv kernel with data weights: needs --simd, --conv data'
        merged = []
        for e in plan:
            if merged and e[0].op == 'maxpool2' and merged[-1][3] == 'smlad':
                cn, cit, _, _ = merged.pop()
                fused_pool[cn.name] = e[0]
                merged.append((cn, cit, e[2], 'smlad_pool'))
            else:
                merged.append(e)
        assert fused_pool, '--fuse-pool: no SMLAD conv followed by a max pool'
        plan = merged

    def sname(nd):
        return f'{nd.name}+{fused_pool[nd.name].name}' if nd.name in fused_pool else nd.name
    stage_names = [sname(nd) for nd, _, _, _ in plan] + (['join'] if cores == 2 else []) + \
                  [nd.name for nd in rest if nd.op in ('linear', 'fff')]
    stage_id = {n: k for k, n in enumerate(stage_names)}
    uses16 = any(o == 'int16_t' for _, _, o, _ in plan)
    img16 = plan[0][3] == 'smlad'       # conv1 reads a widened int16 copy of the image

    weights, funcs = [], []
    for nd, it, ot, kern in plan:
        if nd.op == 'conv3x3':
            if kern == 'smlad_pool':
                weights.append(c_array('uint32_t', f'{nd.name}_kp', pack_smlad_kernel(nd.weights), '0x{:08X}', 6))
                funcs.append(emit_conv_pool_ch_smlad(nd, fused_pool[nd.name], ot, shared=cores == 2))
            elif kern == 'smlad' and row_pairs and ot == 'int16_t' and nd.attrs['mode'] == 'broadcast':
                weights.append(c_array('uint32_t', f'{nd.name}_kp', pack_smlad_kernel(nd.weights), '0x{:08X}', 6))
                funcs.append(emit_conv_ch_smlad_rows2(nd, shared=cores == 2))
            elif kern == 'smlad':
                kp = pack_smlad_kernel(nd.weights)
                if conv == 'code':
                    # weights as code: one specialised function per channel, called through a table
                    for c in range(kp.shape[0]):
                        funcs.append(emit_conv_ch_smlad(nd, ot, ch=c, kwords=kp[c]))
                    funcs.append(f'static void (*const {nd.name}_tab[{kp.shape[0]}])(const int16_t *restrict, {ot} *restrict) = {{\n    ' +
                                 ', '.join(f'{nd.name}_ch{c}' for c in range(kp.shape[0])) + '};\n')
                else:
                    weights.append(c_array('uint32_t', f'{nd.name}_kp', kp, '0x{:08X}', 6))
                    funcs.append(emit_conv_ch_smlad(nd, ot, shared=cores == 2))
            else:
                weights.append(c_array('int8_t', f'{nd.name}_w', nd.weights.reshape(-1)))
                funcs.append(emit_conv_ch(nd, it, ot) if simd else
                             emit_conv_ch(nd, 'int8_t' if nd.attrs['mode'] == 'broadcast' else 'int32_t'))
        else:
            funcs.append(emit_pool_ch(nd, it, ot) if simd else emit_pool_ch(nd))
    assert ternary in TERNARY_FORMATS and int4 in INT4_FORMATS and int8 in INT8_FORMATS and conv in CONV_FORMATS
    assert conv == 'data' or simd, '--conv code specialises the SMLAD conv kernels: needs --simd'
    assert (ternary != 'dense' and int4 != 'dense' and int8 != 'dense') or simd, \
        'the dense formats use SXTB16/SMLAD: need --simd'
    pre_call, extra_globals = {}, ''
    for nd in rest:
        if nd.op == 'relu_norm':
            funcs.append(emit_relu_norm(nd))
        elif nd.op == 'linear':
            fmt = {'ternary': ternary if ternary != 'base3' else None,
                   'int4sym': int4 if int4 != 'packed' else None,
                   'int8': int8 if int8 != 'plain' else None}.get(nd.attrs['quant'])
            if fmt:
                d, f, pre, glob = emit_linear_fmt(nd, fmt, rows=nd.name in par_linear, stream=nd.name in streamed)
                if pre:
                    pre_call[nd.name] = pre
                extra_globals += glob
            else:
                d, f = emit_linear(nd, rows=nd.name in par_linear)
            weights.append(d)
            funcs.append(f)
        elif nd.op == 'fff':
            d, f, glob = emit_fff(nd, simd, ternary, int4, int8)
            weights.append(d)
            funcs.append(f)
            extra_globals += glob

    tail_max = max([int(np.prod(nd.out_shape)) for nd in rest if nd.op in ('relu_norm', 'linear', 'fff')] + [feat_ch * n_ch])

    def seg_loop(lo, hi, pa, pb, prof, indent='    ', p16=None):
        """p16: names of the two int16 planes for this core (needed when the plan uses int16)."""
        out = [f'{indent}for (int c = {lo}; c < {hi}; c++) {{']
        cur = 'img16' if img16 else 'img'
        pools = {'int32_t': [pa, pb], 'int16_t': p16 or []}
        used_planes.update(b for b in [pa, pb] + (p16 or []))
        for j, (nd, it, ot, kern) in enumerate(plan):
            if j == len(plan) - 1:
                dst = f'feat + {feat_ch} * c'
            else:
                dst = next(b for b in pools[ot] if b != cur)
                used_dst.add(dst)
            if kern == 'smlad_pool':
                out.append(f'{indent}    {nd.name}_{fused_pool[nd.name].name}_ch({cur}, {nd.name}_kp + 6 * c, {dst});')
            elif nd.op == 'conv3x3' and kern == 'smlad':
                if conv == 'code':
                    out.append(f'{indent}    {nd.name}_tab[c]({cur}, {dst});')
                else:
                    out.append(f'{indent}    {nd.name}_ch({cur}, {nd.name}_kp + 6 * c, {dst});')
            elif nd.op == 'conv3x3':
                out.append(f'{indent}    {nd.name}_ch({cur}, {nd.name}_w + 9 * c, {dst});')
            else:
                out.append(f'{indent}    {nd.name}_ch({cur}, {dst});')
            if prof:
                out.append(f'{indent}    PROF_STAGE({stage_id[sname(nd)]});')
            cur = dst
        out.append(f'{indent}}}')
        return out

    half = n_ch // 2
    dual = ''
    job_of = {}
    body = ['    PROF_BEGIN();']
    if img16:
        body.append('    widen_img(img);   // counted in the conv1 stage')
    used_planes, used_dst = set(), set()
    p16_0 = ['ch16_a', 'ch16_b'] if uses16 else None
    p16_1 = ['ch1_16_a', 'ch1_16_b'] if uses16 else None
    if cores == 1:
        body += seg_loop(0, n_ch, 'ch_a', 'ch_b', True, p16=p16_0)
    else:
        # core-1 jobs: 1 = its half of the conv segment (channels 0..half), 2.. = the upper half of the rows
        # of each parallel linear layer
        jobs = ['        case 1: {'] + ([] if img16 else ['            const int8_t *img = job_img;']) + \
               seg_loop(0, half, 'ch1_a', 'ch1_b', False, '            ', p16=p16_1) + ['            break;', '        }']
        for nd in rest:
            if nd.name in par_linear:
                k = len(job_of) + 2
                job_of[nd.name] = k
                h = nd.out_shape[0] // 2
                jobs.append(f'        case {k}: {nd.name}(act, acc, {h}, {nd.out_shape[0]}); break;')
        jobs_text = '\n'.join(jobs)
        dual = f'''
// ---- second core ----
// Core 1 waits on the SIO FIFO for a job number, runs it, and pushes the number back. Each FIFO
// transfer is fenced with a DMB so the other core sees every buffer write made before it.
{"" if img16 else "static const int8_t *volatile job_img;" + chr(10)}static int32_t ch1_a[{plane}], ch1_b[{plane}];   // core 1's own per-channel planes
{f"static int16_t ch1_16_a[{plane}] __attribute__((aligned(4))), ch1_16_b[{plane}] __attribute__((aligned(4)));  // core 1's int16 planes" + chr(10) if uses16 else ""}
static void run_job(uint32_t job) {{
    switch (job) {{
{jobs_text}
        default: break;
    }}
}}

#if defined(PICO_ON_DEVICE) && PICO_ON_DEVICE
#include "pico/multicore.h"
#include "hardware/sync.h"
static void core1_main(void) {{
    for (;;) {{
        uint32_t job = multicore_fifo_pop_blocking();
        __dmb();
        run_job(job);
        __dmb();
        multicore_fifo_push_blocking(job);
    }}
}}
void ref_init(void) {{ multicore_launch_core1(core1_main); }}
static inline void start_job(uint32_t job) {{ __dmb(); multicore_fifo_push_blocking(job); }}
static inline void join_job(void) {{ (void)multicore_fifo_pop_blocking(); __dmb(); }}
#else
// PC build: no second core; core 1's share runs first, inline. Same partitioning, so the same results.
void ref_init(void) {{}}
static inline void start_job(uint32_t job) {{ run_job(job); }}
static inline void join_job(void) {{}}
#endif
'''
        body += ([] if img16 else ['    job_img = img;']) + ['    start_job(1);'] + seg_loop(half, n_ch, 'ch_a', 'ch_b', True, p16=p16_0) + \
                ['    join_job();', f'    PROF_STAGE({stage_id["join"]});']
    src = 'feat'
    for nd in rest:
        if nd.op == 'relu_norm':
            body.append(f'    {nd.name}({src}, act);')
            src = 'act'
        elif nd.op == 'linear':
            dst = 'logits' if nd is rest[-2] else 'acc'
            if nd.name in pre_call:     # runs on core 0 before core 1 is started, counted in this stage
                body.append(f'    {pre_call[nd.name]}({src});')
            if nd.name in par_linear:
                assert src == 'act' and dst == 'acc'
                body += [f'    start_job({job_of[nd.name]});',
                         f'    {nd.name}(act, acc, 0, {nd.out_shape[0] // 2});',
                         '    join_job();']
            else:
                body.append(f'    {nd.name}({src}, {dst});')
            body.append(f'    PROF_STAGE({stage_id[nd.name]});')
            src = dst
        elif nd.op == 'fff':
            assert src == 'act'
            body.append(f'    {nd.name}(act, acc);')
            body.append(f'    PROF_STAGE({stage_id[nd.name]});')
            src = 'acc'
    body += [
        '    uint32_t best = 0;',
        '    for (uint32_t i = 1; i < 10; i++) if (logits[i] > logits[best]) best = i;',
        '    return best;',
    ]

    variant = ('v0 (one core' if cores == 1 else 'v1 (two cores') + (', SMLAD on int16 planes' if simd else '') + \
              (f', ternary as {ternary}' if ternary != 'base3' else '') + (f', int4 as {int4}' if int4 != 'packed' else '') + (', conv weights as code' if conv == 'code' else '') + (', dense layers streamed from flash' if stream else '') + (f', int8 layers {int8}' if int8 != 'plain' else '') + (', conv + max pool fused' if fuse_pool else '') + (', conv1 two rows per step' if row_pairs else '')
    names = ', '.join(f'"{n}"' for n in stage_names)
    regen = f'compiler/compile.py --model {graph.name}' + (' --cores 2' if cores == 2 else '') + (' --simd' if simd else '') + \
            (f' --ternary {ternary}' if ternary != 'base3' else '') + (f' --int4 {int4}' if int4 != 'packed' else '') + (' --conv code' if conv == 'code' else '') + (' --stream' if stream else '') + (f' --int8 {int8}' if int8 != 'plain' else '') + (' --fuse-pool' if fuse_pool else '') + (' --row-pairs' if row_pairs else '')
    desc = '\n'.join('//   ' + l for l in graph.describe().splitlines())
    head = f'''// Generated by mafc {__version__} (compiler/), back end rp2350_c {variant}, all weights as packed data).
// Source: {source}
// Do not edit: regenerate with  python {regen}
//
{desc}
#include <stdint.h>
#include "ref_infer.h"

const char ref_model_name[] = "mafc:{graph.name}";
const int ref_n_stages = {len(stage_names)};
const char *const ref_stage_names[REF_MAX_STAGES] = {{{names}}};
uint32_t ref_stage_cycles[REF_MAX_STAGES];

// ---- activations: static, fixed addresses ----
static int32_t ch_a[{plane}], ch_b[{plane}];   // per-channel planes of the fused conv segment (core 0)
{f"static int16_t ch16_a[{plane}] __attribute__((aligned(4))), ch16_b[{plane}] __attribute__((aligned(4)));  // int16 planes (bound-proven)" + chr(10) if uses16 else ""}{IMG16_DECL if img16 else ""}static int32_t feat[{feat_ch * n_ch}];          // conv segment output, CHW
static int32_t acc[{tail_max}];
static int8_t act[{tail_max}];
{extra_globals}
// ---- weights ----
'''
    assert len(stage_names) <= 16
    if simd:
        for decl_old, names, ctype, extra in (
                (f'static int32_t ch_a[{plane}], ch_b[{plane}];', ['ch_a', 'ch_b'], 'int32_t', ''),
                (f'static int32_t ch1_a[{plane}], ch1_b[{plane}];', ['ch1_a', 'ch1_b'], 'int32_t', ''),
                (f'static int16_t ch16_a[{plane}] __attribute__((aligned(4))), ch16_b[{plane}] __attribute__((aligned(4)));',
                 ['ch16_a', 'ch16_b'], 'int16_t', ' __attribute__((aligned(4)))'),
                (f'static int16_t ch1_16_a[{plane}] __attribute__((aligned(4))), ch1_16_b[{plane}] __attribute__((aligned(4)));',
                 ['ch1_16_a', 'ch1_16_b'], 'int16_t', ' __attribute__((aligned(4)))')):
            keep = [n for n in names if n in used_dst]
            new = f'static {ctype} ' + ', '.join(f'{n}[{plane}]{extra}' for n in keep) + ';' if keep else '// (unused plane pair removed)'
            head = head.replace(decl_old, new)
            dual = dual.replace(decl_old, new)
        bnote = '\n'.join(f'//   {nd.name}: values in [{bd[nd.name][0]}, {bd[nd.name][1]}], |partial sums| <= {bd[nd.name][2]}'
                          f' -> {ot}{" (SMLAD)" if kern == "smlad" else ""}{f" (SMLAD, fused with {fused_pool[nd.name].name}: conv plane never stored)" if kern == "smlad_pool" else ""}' for nd, it, ot, kern in plan)
        head = head.replace('// ---- weights ----', '// ---- proven value ranges (compiler/mafc/bounds.py) ----\n' + bnote +
                            '\n' + SIMD_HELPERS + (WIDEN_IMG if img16 else '') + '\n// ---- weights ----')
    return (head + ''.join(weights) + '\n// ---- layers ----\n' + '\n'.join(funcs) + dual +
            '\nuint32_t ref_infer(const int8_t img[REF_IMG_PIXELS], int32_t logits[REF_N_CLASSES]) {\n' +
            '\n'.join(body) + '\n}\n')


def write(graph, out_dir, source='', cores=1, simd=False, ternary='base3', int4='packed', conv='data', stream=False,
          int8='plain', fuse_pool=False, row_pairs=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / (f'mafc_{graph.name}' + ('_dual' if cores == 2 else '') + ('_simd' if simd else '') +
                        (f'_{ternary}' if ternary != 'base3' else '') + (f'_i4{int4}' if int4 != 'packed' else '') +
                        (f'_i8{int8}' if int8 != 'plain' else '') +
                        ('_cvcode' if conv == 'code' else '') + ('_stream' if stream else '') + ('_fpool' if fuse_pool else '') + ('_rp' if row_pairs else '') + '.c')
    c_path.write_text(generate(graph, source, cores, simd, ternary, int4, conv, stream, int8, fuse_pool, row_pairs), encoding='utf-8', newline='\n')
    (out_dir / f'mafc_{graph.name}.ir.json').write_text(graph.to_json(), encoding='utf-8', newline='\n')
    return c_path
