"""Front end: a BitNetMCU-exported model header (train/export/<model>.h) -> mafc IR.

Reads the integer weight codes the exporter wrote, so the IR holds exactly the trained, quantized
model. The header parser and weight decoders are shared with harness/golden.py, whose decoding is
proven correct by the plain-C reference (which decodes the packed words independently on the board).
"""
import sys
from pathlib import Path

import numpy as np

from .ir import Graph, Node

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'harness'))
from golden import decode_4bitsym, decode_ternary, parse_header  # noqa: E402

QUANT = {64: 'ternary', 4: 'int4sym'}


def load(header):
    d, a = parse_header(header)
    text = Path(header).read_text()
    if '#define MODEL_CNNMNIST' not in text:
        raise NotImplementedError('only the CNNMNIST layout is supported by this front end')
    c = d['L7_out_channels']
    n = d['L2_incoming_x']

    def fc(layer):
        bpw, n_in, n_out = d[f'{layer}_bitperweight'], d[f'{layer}_incoming_weights'], d[f'{layer}_outgoing_weights']
        dec = {64: decode_ternary, 4: decode_4bitsym}[bpw]
        return dec(a[f'{layer}_weights'], n_in, n_out), QUANT[bpw]

    w1, q1 = fc('L11')
    w2, q2 = fc('L13')
    w3, q3 = fc('L15')
    s = n
    nodes = [Node('input', 'input', (1, s, s), (1, s, s), 'int8')]
    nodes.append(Node('conv1', 'conv3x3', (1, s, s), (c, s - 2, s - 2), 'int32', {'mode': 'broadcast', 'shift': 4},
                      a['L2_weights'].reshape(c, 3, 3)))
    s -= 2
    nodes.append(Node('dw1', 'conv3x3', (c, s, s), (c, s - 2, s - 2), 'int32', {'mode': 'depthwise', 'shift': 4},
                      a['L4_weights'].reshape(c, 3, 3)))
    s -= 2
    nodes.append(Node('pool1', 'maxpool2', (c, s, s), (c, s // 2, s // 2), 'int32'))
    s //= 2
    nodes.append(Node('dw2', 'conv3x3', (c, s, s), (c, s - 2, s - 2), 'int32', {'mode': 'depthwise', 'shift': 4},
                      a['L7_weights'].reshape(c, 3, 3)))
    s -= 2
    nodes.append(Node('pool2', 'maxpool2', (c, s, s), (c, s // 2, s // 2), 'int32'))
    s //= 2
    n_feat = c * s * s
    nodes.append(Node('norm0', 'relu_norm', (c, s, s), (n_feat,), 'int8'))
    nodes.append(Node('fc1', 'linear', (n_feat,), (w1.shape[0],), 'int32', {'quant': q1}, w1))
    nodes.append(Node('norm1', 'relu_norm', (w1.shape[0],), (w1.shape[0],), 'int8'))
    nodes.append(Node('fc2', 'linear', (w1.shape[0],), (w2.shape[0],), 'int32', {'quant': q2}, w2))
    nodes.append(Node('norm2', 'relu_norm', (w2.shape[0],), (w2.shape[0],), 'int8'))
    nodes.append(Node('fc3', 'linear', (w2.shape[0],), (w3.shape[0],), 'int32', {'quant': q3}, w3))
    nodes.append(Node('argmax', 'argmax', (w3.shape[0],), (), 'int32'))
    for nd in nodes:
        if nd.weights is not None:
            nd.weights = nd.weights.astype(np.int64)
    return Graph(Path(header).stem, nodes).check()
