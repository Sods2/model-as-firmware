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
    """[C, 3, 3] int8 -> [C, 6] uint32: per kernel row r, (k[r][1] << 16) | (k[r][0] & 0xFFFF), then k[r][2] (low half)."""
    k = k.astype(np.int64)
    pairs = ((k[:, :, 1] & 0xFFFF) << 16) | (k[:, :, 0] & 0xFFFF)
    singles = k[:, :, 2] & 0xFFFF
    return np.concatenate([pairs, singles], axis=1)


def emit_conv_ch_smlad(node, out_ctype):
    """One channel of a depthwise 3x3 conv on an int16 plane, two outputs per step with SMLAD.
    For outputs x and x+1 of one kernel row r (inputs i0..i3 = in[x..x+3]):
      s0 += k0*i0 + k1*i1  (SMLAD on the word (i0, i1))   + k2*i2  (SMLABB, low half of (i2, i3))
      s1 += k0*i1 + k1*i2  (SMLAD on the repacked (i1, i2)) + k2*i3  (SMLATB, high half of (i2, i3))
    Exact: the inputs are proven to fit int16 and the partial sums int32 (bounds.py)."""
    n = node.in_shape[1]
    m = n - 2
    sh = node.attrs['shift']
    assert n % 2 == 0 and m % 2 == 0, 'SMLAD kernel needs an even plane width (aligned word pairs)'
    rows = []
    for r in range(3):
        rows.append(f'''            {{
                uint32_t p, q;
                memcpy(&p, in + (y + {r}) * {n} + x, 4);
                memcpy(&q, in + (y + {r}) * {n} + x + 2, 4);
                s0 = SMLAD(p, kp[{r}], s0);
                s0 = SMLABB(q, kp[{3 + r}], s0);
                s1 = SMLAD((p >> 16) | (q << 16), kp[{r}], s1);
                s1 = SMLATB(q, kp[{3 + r}], s1);
            }}''')
    body = '\n'.join(rows)
    return f'''// {node.name}: conv3x3 (depthwise), one channel, int16 {n}x{n} -> {m}x{m}, ReLU, >> {sh}; SMLAD, 2 outputs per step
static inline void {node.name}_ch(const int16_t *restrict in, const uint32_t *restrict kp, {out_ctype} *restrict out) {{
    for (int y = 0; y < {m}; y++) {{
        {out_ctype} *o = out + y * {m};
        for (int x = 0; x < {m}; x += 2) {{
            int32_t s0 = 0, s1 = 0;
{body}
            o[x] = {relu_shift('s0', sh, out_ctype)};
            o[x + 1] = {relu_shift('s1', sh, out_ctype)};
        }}
    }}
}}
'''


SIMD_HELPERS = '''
// ---- DSP helpers: SMLAD/SMLABB/SMLATB on the M33, plain C on the PC (same results) ----
#include <string.h>
#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP
#include <arm_acle.h>
#define SMLAD(a, b, acc)  __smlad((a), (b), (acc))
#define SMLABB(a, b, acc) __smlabb((int32_t)(a), (int32_t)(b), (acc))
#define SMLATB(a, b, acc) __smlatb((int32_t)(a), (int32_t)(b), (acc))
#else
static inline int32_t SMLAD(uint32_t a, uint32_t b, int32_t acc) {
    return acc + (int16_t)a * (int16_t)b + (int16_t)(a >> 16) * (int16_t)(b >> 16);
}
static inline int32_t SMLABB(uint32_t a, uint32_t b, int32_t acc) { return acc + (int16_t)a * (int16_t)b; }
static inline int32_t SMLATB(uint32_t a, uint32_t b, int32_t acc) { return acc + (int16_t)(a >> 16) * (int16_t)b; }
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


def generate(graph, source='', cores=1, simd=False):
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
    stage_names = [nodes[i].name for i in seg] + (['join'] if cores == 2 else []) + \
                  [nd.name for nd in rest if nd.op == 'linear']
    stage_id = {n: k for k, n in enumerate(stage_names)}

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
        if (simd and nd.op == 'conv3x3' and nd.attrs['mode'] == 'depthwise' and in_t == 'int16_t'
                and nd.in_shape[1] % 2 == 0 and (nd.in_shape[1] - 2) % 2 == 0):
            assert bd[nd.name][2] <= 2**31 - 1
            kern = 'smlad'
        plan.append((nd, in_t, out_t, kern))
        in_t = out_t
    uses16 = any(o == 'int16_t' for _, _, o, _ in plan)

    weights, funcs = [], []
    for nd, it, ot, kern in plan:
        if nd.op == 'conv3x3':
            if kern == 'smlad':
                weights.append(c_array('uint32_t', f'{nd.name}_kp', pack_smlad_kernel(nd.weights), '0x{:08X}', 6))
                funcs.append(emit_conv_ch_smlad(nd, ot))
            else:
                weights.append(c_array('int8_t', f'{nd.name}_w', nd.weights.reshape(-1)))
                funcs.append(emit_conv_ch(nd, it, ot) if simd else
                             emit_conv_ch(nd, 'int8_t' if nd.attrs['mode'] == 'broadcast' else 'int32_t'))
        else:
            funcs.append(emit_pool_ch(nd, it, ot) if simd else emit_pool_ch(nd))
    for nd in rest:
        if nd.op == 'relu_norm':
            funcs.append(emit_relu_norm(nd))
        elif nd.op == 'linear':
            d, f = emit_linear(nd, rows=nd.name in par_linear)
            weights.append(d)
            funcs.append(f)

    tail_max = max([int(np.prod(nd.out_shape)) for nd in rest if nd.op in ('relu_norm', 'linear')] + [feat_ch * n_ch])

    def seg_loop(lo, hi, pa, pb, prof, indent='    ', p16=None):
        """p16: names of the two int16 planes for this core (needed when the plan uses int16)."""
        out = [f'{indent}for (int c = {lo}; c < {hi}; c++) {{']
        cur = 'img'
        pools = {'int32_t': [pa, pb], 'int16_t': p16 or []}
        used_planes.update(b for b in [pa, pb] + (p16 or []))
        for j, (nd, it, ot, kern) in enumerate(plan):
            if j == len(plan) - 1:
                dst = f'feat + {feat_ch} * c'
            else:
                dst = next(b for b in pools[ot] if b != cur)
                used_dst.add(dst)
            if nd.op == 'conv3x3' and kern == 'smlad':
                out.append(f'{indent}    {nd.name}_ch({cur}, {nd.name}_kp + 6 * c, {dst});')
            elif nd.op == 'conv3x3':
                out.append(f'{indent}    {nd.name}_ch({cur}, {nd.name}_w + 9 * c, {dst});')
            else:
                out.append(f'{indent}    {nd.name}_ch({cur}, {dst});')
            if prof:
                out.append(f'{indent}    PROF_STAGE({stage_id[nd.name]});')
            cur = dst
        out.append(f'{indent}}}')
        return out

    half = n_ch // 2
    dual = ''
    job_of = {}
    body = ['    PROF_BEGIN();']
    used_planes, used_dst = set(), set()
    p16_0 = ['ch16_a', 'ch16_b'] if uses16 else None
    p16_1 = ['ch1_16_a', 'ch1_16_b'] if uses16 else None
    if cores == 1:
        body += seg_loop(0, n_ch, 'ch_a', 'ch_b', True, p16=p16_0)
    else:
        # core-1 jobs: 1 = its half of the conv segment (channels 0..half), 2.. = the upper half of the rows
        # of each parallel linear layer
        jobs = ['        case 1: {', '            const int8_t *img = job_img;'] + \
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
static const int8_t *volatile job_img;
static int32_t ch1_a[{plane}], ch1_b[{plane}];   // core 1's own per-channel planes
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
        body += ['    job_img = img;', '    start_job(1);'] + seg_loop(half, n_ch, 'ch_a', 'ch_b', True, p16=p16_0) + \
                ['    join_job();', f'    PROF_STAGE({stage_id["join"]});']
    src = 'feat'
    for nd in rest:
        if nd.op == 'relu_norm':
            body.append(f'    {nd.name}({src}, act);')
            src = 'act'
        elif nd.op == 'linear':
            dst = 'logits' if nd is rest[-2] else 'acc'
            if nd.name in par_linear:
                assert src == 'act' and dst == 'acc'
                body += [f'    start_job({job_of[nd.name]});',
                         f'    {nd.name}(act, acc, 0, {nd.out_shape[0] // 2});',
                         '    join_job();']
            else:
                body.append(f'    {nd.name}({src}, {dst});')
            body.append(f'    PROF_STAGE({stage_id[nd.name]});')
            src = dst
    body += [
        '    uint32_t best = 0;',
        '    for (uint32_t i = 1; i < 10; i++) if (logits[i] > logits[best]) best = i;',
        '    return best;',
    ]

    variant = ('v0 (one core' if cores == 1 else 'v1 (two cores') + (', SMLAD on int16 planes' if simd else '')
    names = ', '.join(f'"{n}"' for n in stage_names)
    regen = f'compiler/compile.py --model {graph.name}' + (' --cores 2' if cores == 2 else '') + (' --simd' if simd else '')
    desc = '\n'.join('//   ' + l for l in graph.describe().splitlines())
    head = f'''// Generated by mafc {__version__} (compiler/), back end rp2350_c {variant}, all weights as packed data).
// Source: {source}
// Do not edit: regenerate with  train/.venv/Scripts/python.exe {regen}
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
{f"static int16_t ch16_a[{plane}] __attribute__((aligned(4))), ch16_b[{plane}] __attribute__((aligned(4)));  // int16 planes (bound-proven)" + chr(10) if uses16 else ""}static int32_t feat[{feat_ch * n_ch}];          // conv segment output, CHW
static int32_t acc[{tail_max}];
static int8_t act[{tail_max}];

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
                          f' -> {ot}{" (SMLAD)" if kern == "smlad" else ""}' for nd, it, ot, kern in plan)
        head = head.replace('// ---- weights ----', '// ---- proven value ranges (compiler/mafc/bounds.py) ----\n' + bnote +
                            '\n' + SIMD_HELPERS + '\n// ---- weights ----')
    return (head + ''.join(weights) + '\n// ---- layers ----\n' + '\n'.join(funcs) + dual +
            '\nuint32_t ref_infer(const int8_t img[REF_IMG_PIXELS], int32_t logits[REF_N_CLASSES]) {\n' +
            '\n'.join(body) + '\n}\n')


def write(graph, out_dir, source='', cores=1, simd=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / (f'mafc_{graph.name}' + ('_dual' if cores == 2 else '') + ('_simd' if simd else '') + '.c')
    c_path.write_text(generate(graph, source, cores, simd), encoding='utf-8', newline='\n')
    (out_dir / f'mafc_{graph.name}.ir.json').write_text(graph.to_json(), encoding='utf-8', newline='\n')
    return c_path
