"""Phase 0 test harness: run MNIST test images through a target, compare, and log one CSV row.

Targets:
  golden            numpy emulation of the C reference (harness/golden.py)
  host  --exe X     the C reference compiled for the PC
  pico  --port COMn the firmware on the board, over USB serial

Every target is checked against:
  - the golden model: logits must match bit for bit (this is the Gate 0 correctness check)
  - the PyTorch model the weights came from: prediction agreement, in %
  - the true labels: accuracy, in %

Usage (from the repo root, with the venv's python; see docs/SETUP.md):
  python harness/run.py --model tern80 --target golden --csv scratch.csv
  python harness/run.py --model tern80 --target pico --port COM3 --elf <ref_c.elf>     # /dev/ttyACM0 on Linux
"""
import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BITNET = ROOT / 'train' / 'BitNetMCU'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import Golden, GoldenFFF, GoldenS8  # noqa: E402

CONFIGS = {'tern80': 'ternary_cnn80.yaml', 'tern80z70': 'ternary_cnn80_z70.yaml', 'tern80z90': 'ternary_cnn80_z90.yaml',
           'int8_80': 'int8_cnn80.yaml', 'fff80': 'fff_cnn80.yaml'}
FFF_MODELS = {'fff80'}   # idea B: tree models, exported to train/export/<model>.npz by train/export_fff.py


def find_tool(name, env):
    """A toolchain program: $<env>, else the Pico VS Code extension's GCC 15.2.Rel1 (the version the results were
    measured with), else PATH. None if it isn't anywhere."""
    if os.environ.get(env):
        return os.environ[env]
    sdk = Path(os.environ.get('PICO_SDK_HOME', Path.home() / '.pico-sdk')) / 'toolchain/15_2_Rel1/bin'
    for c in (sdk / f'{name}.exe', sdk / name):
        if c.exists():
            return str(c)
    return shutil.which(name)


SIZE_TOOL = find_tool('arm-none-eabi-size', 'ARM_SIZE')
SUMMARY_COLS = ['date', 'build', 'model', 'target', 'n_images', 'accuracy_pct', 'golden_match_pct',
                'pytorch_agree_pct', 'latency_ms_mean', 'cycles_mean', 'cycles_min', 'cycles_max', 'sys_mhz',
                'flash_kb', 'ram_kb', 'flash_read_kb', 'energy_mj', 'notes']


def load_mnist(n, train=False):
    """MNIST (test set by default) as BitNetMCU feeds it: 16x16, normalized float, plus the int8 version the MCU gets."""
    import torch
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    try:
        ds = datasets.MNIST(root=str(BITNET / 'data'), train=train, transform=tf, download=False)
    except RuntimeError as e:
        raise SystemExit(f'MNIST not found in {BITNET / "data"}. Download it once with the command in '
                         f'docs/SETUP.md ("MNIST test images"). ({e})') from None
    x = torch.stack([ds[i][0] for i in range(n)])
    y = np.array([ds[i][1] for i in range(n)])
    flat = x.view(n, -1).numpy()
    scale = 127.0 / np.maximum(np.abs(flat).max(axis=-1, keepdims=True), 1e-5)
    img8 = np.round(flat * scale).clip(-128, 127).astype(np.int8)
    return x, y, img8


def pytorch_preds(model_name, x):
    """Predictions of the trained (quantization-aware) PyTorch model on the float inputs. None when the BitNetMCU
    clone (which defines the model classes) isn't there: the PyTorch agreement check is then skipped."""
    if not (BITNET / 'exportquant.py').exists():
        print(f'note: {BITNET} not found, so the PyTorch agreement check is skipped (docs/SETUP.md, "BitNetMCU")')
        return None
    import torch
    import yaml
    cwd = os.getcwd()
    sys.path.insert(0, str(BITNET))
    os.chdir(BITNET)  # BitNetMCU's modules expect to run from their folder
    try:
        with open(ROOT / 'train' / 'configs' / CONFIGS[model_name]) as f:
            params = yaml.safe_load(f)
        if model_name in FFF_MODELS:     # training's loader passes the tree parameters; eval (no grad) = hard routing
            from training import load_model
            model = load_model(params['model'], {**params, 'num_classes': 10})
        else:
            from exportquant import load_model
            model = load_model(params['model'], params)
        model.load_state_dict(torch.load(ROOT / 'train' / 'models' / f'{model_name}.pth', map_location='cpu'),
                              strict=False)
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x[i:i + 500]).argmax(1) for i in range(0, len(x), 500)]).numpy()
    finally:
        os.chdir(cwd)


def parse_r(line):
    parts = line.split()
    assert parts[0] == 'R' and len(parts) == 13, f'bad reply: {line!r}'
    return int(parts[1]), int(parts[2]), [int(v) for v in parts[3:]]


def run_host(exe, img8):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'images.bin'
        p.write_bytes(img8.tobytes())
        out = subprocess.run([str(Path(exe).resolve()), str(p)], capture_output=True, text=True, check=True).stdout
    rows = [parse_r(l) for l in out.splitlines() if l.strip()]
    assert len(rows) == len(img8), f'host returned {len(rows)} results for {len(img8)} images'
    return rows, {}


def run_pico(port, img8, stages_out=None):
    import serial
    rows = []
    with serial.Serial(port, 115200, timeout=5) as s:
        time.sleep(0.2)
        s.reset_input_buffer()
        s.write(b'?')
        info = s.readline().decode().strip()
        assert info.startswith('INFO'), f'no INFO reply from board: {info!r}'
        meta = dict(kv.split('=', 1) for kv in info.split()[1:])
        print(f'board: {info}')
        t0 = time.time()
        for i, im in enumerate(img8):
            s.write(b'I' + im.tobytes())
            rows.append(parse_r(s.readline().decode().strip()))
            if (i + 1) % 1000 == 0:
                print(f'  {i + 1}/{len(img8)}  ({time.time() - t0:.0f} s)')
        # XIP cache counters summed over the run (firmware since 12 Nov 2026; older firmware doesn't answer)
        s.write(b'X')
        line = s.readline().decode().strip()
        if line.startswith('X '):
            n, acc, hit = (int(v) for v in line.split()[1:4])
            meta['xip'] = (n, acc, hit)
        if stages_out is not None:
            # per-stage cycles of the first test image, re-run so the numbers belong to a known input
            s.write(b'I' + img8[0].tobytes())
            s.readline()
            s.write(b'T')
            line = s.readline().decode().strip()
            if line.startswith('T'):
                stages_out.update({k: int(v) for k, v in (kv.split('=') for kv in line.split()[1:])})
    return rows, meta


XIP_LINE = 8                                   # bytes per RP2350 XIP cache line (one miss fetches one line)
SRAM_LO, SRAM_HI = 0x20000000, 0x20080000     # main SRAM; the scratch banks above it hold only the stacks


def elf_sizes(elf, in_ram=False):
    """(flash KB, RAM KB). Flash = text + data from arm-none-eabi-size (everything stored in flash).
    RAM = the sizes of all sections linked at main-SRAM addresses (size -A), excluding the heap and the
    stacks. This counts code copied to SRAM at boot (copy_to_ram builds, and functions placed in
    .time_critical.* in flash builds), which Berkeley `size` reports as text.
    (Until 11 Nov 2026 RAM was data + bss (+ text for copy_to_ram builds); see results/README.md.)
    in_ram is unused, kept for callers."""
    out = subprocess.run([str(SIZE_TOOL), str(elf)], capture_output=True, text=True, check=True).stdout
    text, data, bss = (int(v) for v in out.splitlines()[1].split()[:3])
    out_a = subprocess.run([str(SIZE_TOOL), '-A', str(elf)], capture_output=True, text=True, check=True).stdout
    ram = 0
    for line in out_a.splitlines():
        f = line.split()
        if len(f) == 3 and f[1].isdigit() and f[2].isdigit() and SRAM_LO <= int(f[2]) < SRAM_HI and f[0] != '.heap':
            ram += int(f[1])
    return round((text + data) / 1024, 2), round(ram / 1024, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tern80', choices=sorted(CONFIGS))
    ap.add_argument('--target', required=True, choices=['golden', 'host', 'pico'])
    ap.add_argument('--build', default=None, help='build name for the CSV (default: what the board reports, else ref_c)')
    ap.add_argument('--golden', default='ref', choices=['ref', 's8'],
                    help='golden model: ref = plain-C reference arithmetic, s8 = CMSIS-NN int8 build')
    ap.add_argument('--exe', help='host executable (target host)')
    ap.add_argument('--port', help='serial port (target pico)')
    ap.add_argument('--elf', help='firmware ELF, for flash/RAM size')
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--notes', default='')
    ap.add_argument('--csv', default=str(ROOT / 'results' / 'phase0.csv'))
    args = ap.parse_args()

    x, y, img8 = load_mnist(args.n)
    if args.golden == 'ref' and args.model in FFF_MODELS:
        golden = GoldenFFF(ROOT / 'train' / 'export' / f'{args.model}.npz')
    elif args.golden == 'ref':
        golden = Golden(ROOT / 'train' / 'export' / f'{args.model}.h')
    else:
        golden = GoldenS8(ROOT / 'firmware' / 'baselines' / 'cmsis_nn' / f'model_s8_{args.model}.h')
    g_logits = golden.logits(img8)
    g_pred = g_logits.argmax(1)
    t_pred = pytorch_preds(args.model, x)

    meta, stages = {}, {}
    if args.target == 'golden':
        pred, cycles, logits = g_pred, np.zeros(args.n, dtype=np.int64), g_logits
    else:
        stages = {}
        rows, meta = run_host(args.exe, img8) if args.target == 'host' else run_pico(args.port, img8, stages)
        pred = np.array([r[0] for r in rows])
        cycles = np.array([r[1] for r in rows], dtype=np.int64)
        logits = np.array([r[2] for r in rows], dtype=np.int64)

    if args.build is None:
        args.build = meta.get('build', 'ref_c')
    match = (logits == g_logits).all(axis=1)
    sys_hz = int(meta.get('sys_hz', 0))
    acc = (pred == y).mean() * 100
    summary = {
        'date': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'build': args.build, 'model': args.model, 'target': args.target, 'n_images': args.n,
        'accuracy_pct': round(acc, 2),
        'golden_match_pct': round(match.mean() * 100, 2),
        'pytorch_agree_pct': round((pred == t_pred).mean() * 100, 2) if t_pred is not None else '',
        'latency_ms_mean': round(cycles.mean() / sys_hz * 1e3, 3) if sys_hz else '',
        'cycles_mean': int(cycles.mean()) if sys_hz else '',
        'cycles_min': int(cycles.min()) if sys_hz else '',
        'cycles_max': int(cycles.max()) if sys_hz else '',
        'sys_mhz': sys_hz // 1_000_000 if sys_hz else '',
        'flash_kb': '', 'ram_kb': '', 'flash_read_kb': '', 'energy_mj': '',
        'notes': args.notes,
    }
    if args.elf and SIZE_TOOL:
        summary['flash_kb'], summary['ram_kb'] = elf_sizes(args.elf, in_ram='_ram' in args.build)
    elif args.elf:
        print('note: arm-none-eabi-size not found (set ARM_SIZE), so flash/RAM sizes are left empty')
    xip = meta.get('xip')
    if xip and xip[0]:
        # flash_read_kb: XIP cache misses x 8-byte lines, per inference (FINDINGS F22). Uncached XIP accesses
        # would count as misses too; the measured builds make none.
        n_img, x_acc, x_hit = xip
        summary['flash_read_kb'] = round((x_acc - x_hit) * XIP_LINE / n_img / 1024, 2)

    print(f'model {args.model}, target {args.target}, {args.n} images')
    torch_acc = f'   (PyTorch model: {(t_pred == y).mean() * 100:.2f} %)' if t_pred is not None else ''
    print(f'  accuracy           {acc:.2f} %{torch_acc}')
    print(f'  logits == golden   {match.sum()}/{args.n}')
    print(f'  agrees w/ PyTorch  ' + (f'{summary["pytorch_agree_pct"]} %' if t_pred is not None else 'n/a (BitNetMCU not cloned)'))
    if hasattr(golden, 'max_abs_bound'):
        print(f'  int32 headroom     max |partial sum| bound = {golden.max_abs_bound} ({golden.max_abs_bound / 2**31 * 100:.2f} % of int32)')
    if sys_hz:
        print(f'  cycles             mean {summary["cycles_mean"]}, min {summary["cycles_min"]}, max {summary["cycles_max"]}'
              f'  ->  {summary["latency_ms_mean"]} ms at {summary["sys_mhz"]} MHz')
    if args.elf and SIZE_TOOL:
        print(f'  flash {summary["flash_kb"]} KB, RAM {summary["ram_kb"]} KB')
    if xip and xip[0]:
        n_img, x_acc, x_hit = xip
        print(f'  XIP per inference  {x_acc / n_img:.0f} accesses, {(x_acc - x_hit) / n_img:.0f} misses '
              f'(hit rate {x_hit / max(x_acc, 1) * 100:.2f} %)  ->  flash read {summary["flash_read_kb"]} KB')
    if not match.all():
        bad = np.flatnonzero(~match)[:5]
        print(f'  FIRST MISMATCHES: images {bad.tolist()}')

    if stages and sum(stages.values()) > 0:
        total = sum(stages.values())
        print('  stages (image 0):  ' + ', '.join(f'{k} {v} ({v / total * 100:.0f}%)' for k, v in stages.items()))
        # one file per stage set, so columns always line up (dual-core builds add a 'join' stage)
        suffix = ('_fff' if 'fff' in stages else '') + ('_fpool' if any('+' in k for k in stages) else '') + ('_stages_dual.csv' if 'join' in stages else '_stages.csv')
        sp = Path(args.csv).with_name(Path(args.csv).stem + suffix)
        new_s = not sp.exists()
        if not new_s:
            with open(sp, newline='') as f:
                assert next(csv.reader(f))[3:] == list(stages), f'stage columns of {sp} differ from this build'
        with open(sp, 'a', newline='') as f:
            w = csv.writer(f)
            if new_s:
                w.writerow(['date', 'build', 'model'] + list(stages))
            w.writerow([dt.datetime.now().strftime('%Y-%m-%d %H:%M'), args.build, args.model] + list(stages.values()))

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if new:
            w.writeheader()
        w.writerow(summary)
    print(f'  logged to {out}')
    return 0 if match.all() else 1


if __name__ == '__main__':
    sys.exit(main())
