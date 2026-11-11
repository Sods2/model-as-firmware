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

def emit_conv_ch(node, in_ctype):
    """One channel of a 3x3 valid conv + ReLU + shift. in: one n x n plane, out: (n-2) x (n-2)."""
    n = node.in_shape[1]
    m = n - 2
    sh = node.attrs['shift']
    return f'''// {node.name}: conv3x3 ({node.attrs['mode']}), one channel, {n}x{n} -> {m}x{m}, ReLU, >> {sh}
static inline void {node.name}_ch(const {in_ctype} *restrict in, const int8_t *restrict k, int32_t *restrict out) {{
    const int32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4], k5 = k[5], k6 = k[6], k7 = k[7], k8 = k[8];
    for (int y = 0; y < {m}; y++) {{
        const {in_ctype} *r0 = in + y * {n}, *r1 = r0 + {n}, *r2 = r1 + {n};
        int32_t *o = out + y * {m};
        for (int x = 0; x < {m}; x++) {{
            int32_t s = k0 * r0[x] + k1 * r0[x + 1] + k2 * r0[x + 2]
                      + k3 * r1[x] + k4 * r1[x + 1] + k5 * r1[x + 2]
                      + k6 * r2[x] + k7 * r2[x + 1] + k8 * r2[x + 2];
            o[x] = s < 0 ? 0 : (s >> {sh});
        }}
    }}
}}
'''


def emit_pool_ch(node):
    n = node.in_shape[1]
    m = n // 2
    return f'''// {node.name}: maxpool 2x2, one channel, {n}x{n} -> {m}x{m}
static inline void {node.name}_ch(const int32_t *restrict in, int32_t *restrict out) {{
    for (int y = 0; y < {m}; y++) {{
        const int32_t *r0 = in + 2 * y * {n}, *r1 = r0 + {n};
        for (int x = 0; x < {m}; x++) {{
            int32_t a = r0[2 * x], b = r0[2 * x + 1], c = r1[2 * x], d = r1[2 * x + 1];
            int32_t ab = a > b ? a : b, cd = c > d ? c : d;
            out[y * {m} + x] = ab > cd ? ab : cd;
        }}
    }}
}}
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


def emit_linear(node):
    """Returns (weights C text, function C text)."""
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


def generate(graph, source=''):
    nodes = graph.nodes
    assert nodes[0].op == 'input' and nodes[0].in_shape[0] == 1 and nodes[0].in_shape[1:] == (16, 16), \
        'v0 back end expects a 1x16x16 int8 input (the harness protocol)'
    seg = channel_fused_prefix(nodes)
    assert seg and nodes[seg[0]].op == 'conv3x3' and nodes[seg[0]].attrs['mode'] == 'broadcast'
    assert all(nodes[i].op != 'conv3x3' or nodes[i].attrs['mode'] == 'depthwise' for i in seg[1:])
    rest = nodes[seg[-1] + 1:]
    assert rest[-1].op == 'argmax' and rest[-2].op == 'linear' and rest[-2].out_shape == (10,), \
        'v0 back end expects the graph to end in linear(->10) + argmax'

    n_ch = nodes[seg[0]].out_shape[0]
    plane = max(int(np.prod(nodes[i].out_shape[1:])) for i in seg)
    feat_ch = int(np.prod(nodes[seg[-1]].out_shape[1:]))
    stage_nodes = [nd for nd in nodes if nd.op in ('conv3x3', 'maxpool2', 'linear')]
    stage_id = {nd.name: k for k, nd in enumerate(stage_nodes)}

    weights, funcs = [], []
    for i in seg:
        nd = nodes[i]
        if nd.op == 'conv3x3':
            weights.append(c_array('int8_t', f'{nd.name}_w', nd.weights.reshape(-1)))
            funcs.append(emit_conv_ch(nd, 'int8_t' if nd.attrs['mode'] == 'broadcast' else 'int32_t'))
        else:
            funcs.append(emit_pool_ch(nd))
    for nd in rest:
        if nd.op == 'relu_norm':
            funcs.append(emit_relu_norm(nd))
        elif nd.op == 'linear':
            d, f = emit_linear(nd)
            weights.append(d)
            funcs.append(f)

    # buffers for the non-fused tail: one int32 accumulator and one int8 activation buffer, reused
    tail_max = max([int(np.prod(nd.out_shape)) for nd in rest if nd.op in ('relu_norm', 'linear')] + [feat_ch * n_ch])

    body = ['    PROF_BEGIN();', f'    for (int c = 0; c < {n_ch}; c++) {{']
    cur = 'img'
    bufs = ['ch_a', 'ch_b']
    for j, i in enumerate(seg):
        nd = nodes[i]
        last = j == len(seg) - 1
        dst = f'feat + {feat_ch} * c' if last else bufs[j % 2]
        if nd.op == 'conv3x3':
            body.append(f'        {nd.name}_ch({cur}, {nd.name}_w + 9 * c, {dst});')
        else:
            body.append(f'        {nd.name}_ch({cur}, {dst});')
        body.append(f'        PROF_STAGE({stage_id[nd.name]});')
        cur = dst
    body.append('    }')
    src = 'feat'
    for nd in rest:
        if nd.op == 'relu_norm':
            body.append(f'    {nd.name}({src}, act);')
            src = 'act'
        elif nd.op == 'linear':
            is_last = nd is rest[-2]
            dst = 'logits' if is_last else 'acc'
            body.append(f'    {nd.name}({src}, {dst});')
            body.append(f'    PROF_STAGE({stage_id[nd.name]});')
            src = dst
    body += [
        '    uint32_t best = 0;',
        '    for (uint32_t i = 1; i < 10; i++) if (logits[i] > logits[best]) best = i;',
        '    return best;',
    ]

    names = ', '.join(f'"{nd.name}"' for nd in stage_nodes)
    head = f'''// Generated by mafc {__version__} (compiler/), back end rp2350_c v0 (all weights as packed data).
// Source: {source}
// Do not edit: regenerate with  train/.venv/Scripts/python.exe compiler/compile.py --model {graph.name}
//
{chr(10).join('//   ' + l for l in graph.describe().splitlines())}
#include <stdint.h>
#include "ref_infer.h"

const char ref_model_name[] = "mafc:{graph.name}";
const int ref_n_stages = {len(stage_nodes)};
const char *const ref_stage_names[REF_MAX_STAGES] = {{{names}}};
uint32_t ref_stage_cycles[REF_MAX_STAGES];

// ---- activations: static, fixed addresses ----
static int32_t ch_a[{plane}], ch_b[{plane}];   // per-channel planes of the fused conv segment
static int32_t feat[{feat_ch * n_ch}];          // conv segment output, CHW
static int32_t acc[{tail_max}];
static int8_t act[{tail_max}];

// ---- weights ----
'''
    assert len(stage_nodes) <= 16
    return (head + ''.join(weights) + '\n// ---- layers ----\n' + '\n'.join(funcs) +
            '\nuint32_t ref_infer(const int8_t img[REF_IMG_PIXELS], int32_t logits[REF_N_CLASSES]) {\n' +
            '\n'.join(body) + '\n}\n')


def write(graph, out_dir, source=''):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / f'mafc_{graph.name}.c'
    c_path.write_text(generate(graph, source), encoding='utf-8', newline='\n')
    (out_dir / f'mafc_{graph.name}.ir.json').write_text(graph.to_json(), encoding='utf-8', newline='\n')
    return c_path
