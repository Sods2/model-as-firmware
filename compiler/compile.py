"""mafc command line: compile a trained model to RP2350 C.

  train/.venv/Scripts/python.exe compiler/compile.py --model tern80
      reads  train/export/tern80.h
      writes firmware/rp2350/mafc/gen/mafc_tern80.c (+ .ir.json), checks the IR against the golden model
  ... --cores 2   writes gen/mafc_tern80_dual.c (conv channels and big linear layers split across both cores)
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
    path = backend_rp2350_c.write(graph, args.out, source=str(header.relative_to(ROOT)).replace('\\', '/'), cores=args.cores)
    print(f'wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
