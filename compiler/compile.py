"""mafc command line: compile a trained model to RP2350 C.

  train/.venv/Scripts/python.exe compiler/compile.py --model tern80
      reads  train/export/tern80.h
      writes firmware/rp2350/mafc/gen/mafc_tern80.c (+ .ir.json), checks the IR against the golden model
  ... --cores 2   writes gen/mafc_tern80_dual.c (conv channels and big linear layers split across both cores)
  ... --simd      adds _simd: int16 planes where compiler/mafc/bounds.py proves they fit, SMLAD depthwise convs
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from mafc import backend_rp2350_c, frontend_bitnetmcu, interp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tern80')
    ap.add_argument('--out', default=str(ROOT / 'firmware' / 'rp2350' / 'mafc' / 'gen'))
    ap.add_argument('--cores', type=int, default=1, choices=[1, 2], help='2 = split work across both M33 cores')
    ap.add_argument('--simd', action='store_true', help='int16 planes where bounds prove it, SMLAD depthwise convs')
    ap.add_argument('--ternary', default='base3', choices=backend_rp2350_c.TERNARY_FORMATS,
                    help='format of ternary linear layers: base3 (10 trits/uint16, decode loop), sparse (index lists), '
                         'code (weights compiled into instructions, idea A), dense (int8 + SXTB16/SMLAD, needs --simd)')
    ap.add_argument('--int4', default='packed', choices=backend_rp2350_c.INT4_FORMATS,
                    help='format of int4sym linear layers (fc2, fc3): packed (8 per uint32, decode loop), '
                         'code (weights compiled into instructions), dense (int8 + SXTB16/SMLAD, needs --simd)')
    ap.add_argument('--int8', default='plain', choices=backend_rp2350_c.INT8_FORMATS,
                    help='format of int8 linear layers: plain (int8 array, multiply loop), '
                         'code (weights compiled into instructions), dense (SXTB16/SMLAD, needs --simd)')
    ap.add_argument('--conv', default='data', choices=backend_rp2350_c.CONV_FORMATS,
                    help='conv weights: data (packed words loaded per channel) or code (one specialised function '
                         'per channel, weights as immediates; needs --simd)')
    ap.add_argument('--stream', action='store_true',
                    help='layers in a dense format stream their weights from flash by DMA (idea C, flash builds)')
    ap.add_argument('--check', type=int, default=1000, help='test images to check IR vs golden (0 = skip)')
    args = ap.parse_args()

    header = ROOT / 'train' / 'export' / f'{args.model}.h'
    graph = frontend_bitnetmcu.load(header)
    print(graph.describe())
    if args.check:
        sys.path.insert(0, str(ROOT / 'harness'))
        from golden import Golden
        from run import load_mnist
        _, _, img8 = load_mnist(args.check)
        ok = (interp.run(graph, img8) == Golden(header).logits(img8)).all()
        print(f'IR interpreter == golden on {args.check} images: {ok}')
        if not ok:
            sys.exit(1)
    path = backend_rp2350_c.write(graph, args.out, source=str(header.relative_to(ROOT)).replace('\\', '/'), cores=args.cores, simd=args.simd, ternary=args.ternary, int4=args.int4, conv=args.conv, stream=args.stream, int8=args.int8)
    print(f'wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
