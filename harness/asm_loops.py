"""List the loops (backward branches) in a ref_infer disassembly, with instruction, IT, branch-free-ReLU (bic asr #31),
branch, SMLA* and stack-access ([sp]) counts per loop body. Used to check the back end's hot loops after a change (notes/compiler.md, v1 step 3).

  OD=~/.pico-sdk/toolchain/15_2_Rel1/bin/arm-none-eabi-objdump.exe
  $OD -d --no-show-raw-insn <build>/mafc.elf | awk '/<ref_infer>:/{f=1} f{print} f&&/^$/{exit}' > ref_infer.s
  train/.venv/Scripts/python.exe harness/asm_loops.py [--max N] ref_infer.s [more.s ...]

--max: longest loop body listed (default 120; raise it to see fully unrolled loops). Stack accesses inside a hot
loop are register spills, the first thing to look at when a loop is slower than its instruction mix suggests.
"""
import re
import sys

args = sys.argv[1:]
MAX = 120
if args[:1] == ['--max']:
    MAX, args = int(args[1]), args[2:]
for path in args:
    lines = [l for l in open(path, encoding='utf-8') if re.match(r'^[0-9a-f]{8}:', l)]
    addr = [int(l[:8], 16) for l in lines]
    print(path.split('/')[-1].split('\\')[-1])
    for i, l in enumerate(lines):
        m = re.search(r'\tb(?:ne|eq|lt|ge|gt|le|cc|cs|hi|ls)?(?:\.n|\.w)?\t([0-9a-f]+)', l)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt >= addr[i]:
            continue
        j = addr.index(tgt) if tgt in addr else None
        if j is None:
            continue
        body = lines[j:i + 1]
        n = len(body)
        if n > MAX:
            continue
        it = sum(bool(re.search(r'\tit', b)) for b in body)
        bic = sum('asr #31' in b and 'bic' in b for b in body)
        br = sum(bool(re.search(r'\tb(?:ne|eq|lt|ge|gt|le|mi|pl)?(?:\.n|\.w)?\t', b)) for b in body)
        smlad = sum('smla' in b for b in body)
        sp = sum('[sp' in b for b in body)
        print(f'  {tgt:08x}-{addr[i]:08x}  instr {n:3d}  IT {it}  bic {bic}  branches {br}  smla* {smlad}  [sp] {sp}')
