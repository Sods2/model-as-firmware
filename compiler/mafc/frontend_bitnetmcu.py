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
from golden import QUANT_NAMES, decode_fc, parse_header  # noqa: E402


def load_fff(npz):
    """Idea B: a CNNMNIST_FFF model exported by train/export_fff.py (.npz of integer codes) -> mafc IR."""
    z = np.load(npz)
    k1 = z['conv1']
    c, s = k1.shape[0], 16
    d, L = int(z['depth']), int(z['leaf'])
    q = {str(k)[:-6]: str(z[k]) for k in z.files if k.endswith('_quant')}
    nodes = [Node('input', 'input', (1, s, s), (1, s, s), 'int8')]
    nodes.append(Node('conv1', 'conv3x3', (1, s, s), (c, s - 2, s - 2), 'int32', {'mode': 'broadcast', 'shift': 4}, k1))
    s -= 2
    nodes.append(Node('dw1', 'conv3x3', (c, s, s), (c, s - 2, s - 2), 'int32', {'mode': 'depthwise', 'shift': 4}, z['dw1']))
    s -= 2
    nodes.append(Node('pool1', 'maxpool2', (c, s, s), (c, s // 2, s // 2), 'int32'))
    s //= 2
    nodes.append(Node('dw2', 'conv3x3', (c, s, s), (c, s - 2, s - 2), 'int32', {'mode': 'depthwise', 'shift': 4}, z['dw2']))
    s -= 2
    nodes.append(Node('pool2', 'maxpool2', (c, s, s), (c, s // 2, s // 2), 'int32'))
    s //= 2
    n_feat = c * s * s
    n_out = z['leaf_out'].shape[1]
    nodes.append(Node('norm0', 'relu_norm', (c, s, s), (n_feat,), 'int8'))
    nodes.append(Node('fff', 'fff', (n_feat,), (n_out,), 'int32',
                      {'depth': d, 'leaf': L, 'quant_nodes': q['nodes'], 'quant_in': q['leaf_in'], 'quant_out': q['leaf_out']},
                      tensors={k: z[k].astype(np.int64) for k in ('nodes', 'leaf_in', 'leaf_out')}))
    nodes.append(Node('norm2', 'relu_norm', (n_out,), (n_out,), 'int8'))
    nodes.append(Node('fc3', 'linear', (n_out,), (z['fc3'].shape[0],), 'int32', {'quant': q['fc3']}, z['fc3']))
    nodes.append(Node('argmax', 'argmax', (z['fc3'].shape[0],), (), 'int32'))
    for nd in nodes:
        if nd.weights is not None:
            nd.weights = nd.weights.astype(np.int64)
    return Graph(Path(npz).stem, nodes).check()


def load(header):
    d, a = parse_header(header)
    text = Path(header).read_text()
    if '#define MODEL_CNNMNIST' not in text:
        raise NotImplementedError('only the CNNMNIST layout is supported by this front end')
    c = d['L7_out_channels']
    n = d['L2_incoming_x']

    def fc(layer):
        w, bpw = decode_fc(d, a, layer)
        return w, QUANT_NAMES[bpw]

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
