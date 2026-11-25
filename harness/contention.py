"""Do the two cores slow each other down? (notes/IDEAS.md U3, notes/FINDINGS.md F12)

Compares each core's share of the conv segment in the dual-core build with half of the single-core
conv segment, from flash and from SRAM, using the stage profiles that harness/measure.sh profile writes.
Rows of the same build are averaged (run the profile several times for repeats).

  - core 0's conv time is the sum of its conv stages;
  - core 1's conv time is estimated as core 0's time + the join wait (both start within a FIFO push of
    each other; there is no per-core timer on core 1);
  - fc1 is wall time with both cores working on half the rows each.

Usage: python harness/contention.py results/contention_stages.csv results/contention_stages_dual.csv
"""
import csv
import statistics as st
import sys

CONV = ['conv1', 'dw1', 'pool1', 'dw2', 'pool2']
PAIRS = (('flash', 'mafc_v0_prof', 'mafc_v1_dual_prof'), ('SRAM', 'mafc_v0_ram_prof', 'mafc_v1_dual_ram_prof'))


def load(path):
    out = {}
    for r in csv.DictReader(open(path, newline='')):
        out.setdefault(r['build'], []).append({k: int(v) for k, v in r.items() if k not in ('date', 'build', 'model')})
    return out


def mean(rows, key):
    return st.mean(r[key] for r in rows)


def spread(vals):
    return (max(vals) - min(vals)) / st.mean(vals) * 100 if len(vals) > 1 else 0.0


def main(single_csv, dual_csv):
    single, dual = load(single_csv), load(dual_csv)
    res = {}
    for mem, s_name, d_name in PAIRS:
        s, d = single[s_name], dual[d_name]
        half = st.mean(sum(r[k] for k in CONV) for r in s) / 2
        c0s = [sum(r[k] for k in CONV) for r in d]
        joins = [r['join'] for r in d]
        c0, join = st.mean(c0s), st.mean(joins)
        c1 = c0 + join
        fc1_s, fc1_d = mean(s, 'fc1'), mean(d, 'fc1')
        res[mem] = (half, c0, c1, fc1_s, fc1_d)
        print(f'{mem} ({len(s)} single / {len(d)} dual runs; spread of dual conv {spread(c0s):.2f}%, of join {spread(joins):.1f}%)')
        print(f'  conv, each core\'s half:  single/2 {half:,.0f}   dual core 0 {c0:,.0f} ({(c0 / half - 1) * 100:+.1f}%)'
              f'   join {join:,.0f}   dual core 1 ~{c1:,.0f} ({(c1 / half - 1) * 100:+.1f}%)')
        print('  per stage, core 0 vs single/2: ' +
              ', '.join(f'{k} {(mean(d, k) / (mean(s, k) / 2) - 1) * 100:+.0f}%' for k in CONV))
        print(f'  fc1: single {fc1_s:,.0f}, dual {fc1_d:,.0f}  -> {fc1_s / fc1_d:.2f}x of the ideal 2x')
    f, r = res['flash'], res['SRAM']
    print('flash penalty (flash vs SRAM, same build):')
    print(f'  conv:  single {f[0] / r[0] - 1:+.1%}   dual core 0 {f[1] / r[1] - 1:+.1%}   dual core 1 {f[2] / r[2] - 1:+.1%}')
    print(f'  fc1:   single {f[3] / r[3] - 1:+.1%}   dual {f[4] / r[4] - 1:+.1%}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
