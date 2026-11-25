"""Energy per inference with a USB power meter between the PC and the Pico (Phase 0 step 2b).

The board must run firmware built from pico_main.c with the 'L' command (12 Nov 2026 or later). Two phases of equal
length, read off the meter by hand:
  1. idle: the board waits for a command (USB connected, nothing running);
  2. busy: the board runs ref_infer() back to back on one test image for --secs seconds ('L' command), no USB traffic.
Energy per inference above idle = (P_busy - P_idle) x T / inferences; the total (including idle power) is also logged.

Meter readings, typed in when asked (--reading):
  mwh  energy accumulated over the phase (reset the meter's counter before each phase)
  mw   average power during the phase
  ma   average current during the phase (power = mA x --volts)
A meter with 1 mWh resolution gives power to +-3600/secs mW, so use long phases (default 600 s) with --reading mwh.
--no-meter skips the prompts and only reports inferences per second (to check the firmware and the loop).

Usage (from the repo root):
  python harness/energy.py --port COM3 --model tern80 --secs 600 --reading mwh
"""
import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import load_mnist, parse_r  # noqa: E402

COLS = ['date', 'build', 'model', 'secs', 'inferences', 'latency_ms', 'reading', 'idle_raw', 'busy_raw',
        'p_idle_mw', 'p_busy_mw', 'energy_mj_above_idle', 'energy_mj_total', 'notes']


def ask(prompt):
    while True:
        v = input(prompt).strip().replace(',', '.')
        try:
            return float(v)
        except ValueError:
            print('  a number, please')


def to_mw(raw, kind, secs, volts):
    if kind == 'mwh':
        return raw * 3600.0 / secs
    if kind == 'ma':
        return raw * volts
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--model', default='tern80')
    ap.add_argument('--build', default=None, help='build name for the CSV (default: what the board reports)')
    ap.add_argument('--secs', type=int, default=600, help='length of each phase, seconds (max 65535)')
    ap.add_argument('--reading', choices=['mwh', 'mw', 'ma'], default='mwh')
    ap.add_argument('--volts', type=float, default=5.0, help='bus voltage, for --reading ma')
    ap.add_argument('--no-meter', action='store_true', help='no prompts: only run the busy loop and report the rate')
    ap.add_argument('--skip-idle', type=float, default=None, metavar='P_IDLE_MW',
                    help='reuse an idle power measured before (mW) instead of measuring it again')
    ap.add_argument('--notes', default='')
    ap.add_argument('--csv', default=str(ROOT / 'results' / 'energy.csv'))
    args = ap.parse_args()
    assert 1 <= args.secs <= 65535

    import serial
    _, _, img8 = load_mnist(1)
    with serial.Serial(args.port, 115200, timeout=5) as s:
        time.sleep(0.2)
        s.reset_input_buffer()
        s.write(b'?')
        info = s.readline().decode().strip()
        assert info.startswith('INFO'), f'no INFO reply from board: {info!r}'
        meta = dict(kv.split('=', 1) for kv in info.split()[1:])
        build = args.build or meta.get('build', '?')
        print(f'board: {info}')
        s.write(b'I' + img8[0].tobytes())
        parse_r(s.readline().decode().strip())

        idle_raw = p_idle = None
        if not args.no_meter:
            if args.skip_idle is not None:
                p_idle = args.skip_idle
            else:
                input(f'IDLE phase ({args.secs} s): reset the meter, then press Enter ')
                time.sleep(args.secs)
                idle_raw = ask(f'  idle reading ({args.reading}): ')
                p_idle = to_mw(idle_raw, args.reading, args.secs, args.volts)
            input(f'BUSY phase ({args.secs} s): reset the meter, then press Enter ')
        s.write(b'L' + args.secs.to_bytes(2, 'little'))
        s.timeout = args.secs + 30
        line = s.readline().decode().strip()
        assert line.startswith('L '), f'bad reply to L: {line!r} (firmware without the L command?)'
        count, us = (int(v) for v in line.split()[1:3])
    t = us / 1e6
    lat_ms = t / count * 1e3
    print(f'  busy loop: {count} inferences in {t:.2f} s -> {lat_ms:.3f} ms each ({count / t:.1f}/s)')
    if args.no_meter:
        return
    busy_raw = ask(f'  busy reading ({args.reading}): ')
    p_busy = to_mw(busy_raw, args.reading, args.secs, args.volts)
    e_above = (p_busy - p_idle) * t / count     # mW x s = mJ
    e_total = p_busy * t / count
    print(f'  P_idle {p_idle:.1f} mW, P_busy {p_busy:.1f} mW -> {e_above * 1e3:.2f} uJ above idle, '
          f'{e_total * 1e3:.2f} uJ total per inference')
    if args.reading == 'mwh':
        print(f'  (1 mWh of meter resolution = +-{3600 / args.secs:.1f} mW, '
              f'+-{3600 / args.secs * t / count * 1e3:.2f} uJ per inference)')
    new = not Path(args.csv).exists()
    with open(args.csv, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        w.writerow([dt.date.today().isoformat(), build, args.model, args.secs, count, round(lat_ms, 4),
                    args.reading, idle_raw, busy_raw, round(p_idle, 2), round(p_busy, 2),
                    round(e_above, 5), round(e_total, 5), args.notes])
    print(f'  appended to {args.csv}')


if __name__ == '__main__':
    main()
