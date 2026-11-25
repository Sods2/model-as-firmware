# firmware/baselines: Phase 0 reference builds

| Build | What it is | Golden model |
|---|---|---|
| [ref_c/](ref_c/) | Plain-C inference of the BitNetMCU CNN with packed weights (ternary fc1). The correctness anchor. Also builds on the PC (`host_main.c`) | `harness/golden.py:Golden` |
| [cmsis_nn/](cmsis_nn/) | The same network and integer weights through ARM CMSIS-NN int8 kernels. The speed bar to beat | `harness/golden.py:GoldenS8` |

Both implement `ref_infer()` from `ref_c/ref_infer.h` and share `ref_c/pico_main.c`, so the harness drives them the same way. Details are in [notes/cmsis-nn-baseline.md](../../notes/cmsis-nn-baseline.md) and [notes/bitnetmcu-integer-inference.md](../../notes/bitnetmcu-integer-inference.md).

## Build options (CMake)

| Option | Default | Effect |
|---|---|---|
| `MODEL` | `tern80` | Model: `tern80`, `tern80z70`, `tern80z90`, `int8_80` (cmsis_nn needs `harness/export_cmsis.py --model X` first) |
| `RUN_FROM_RAM` | OFF | `copy_to_ram` binary: code and weights are copied to SRAM at boot, so there are no flash/XIP stalls |
| `PROFILE_STAGES` | OFF | Per-stage DWT cycle counters, read with the `T` command. Costs about 3% in ref_c, so leave it off for timed runs |

The build name reported to the harness is `<build>[_ram][_prof]`.

Toolchain setup is in [docs/SETUP.md](../../docs/SETUP.md). From the repo root in bash (Git Bash on Windows):

```bash
. harness/env.sh      # the Pico toolchain on PATH
cmake -S firmware/baselines/ref_c -B firmware/baselines/ref_c/build-tern80-flash -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=tern80
ninja -C firmware/baselines/ref_c/build-tern80-flash
```

PC build of the reference (any C11 compiler; the results used 32-bit MinGW gcc 9.2 on Windows). One executable per model; the launcher and `run.py --target host` look for `build-host/ref_host_<model>` (with `.exe` on Windows):

```bash
cd firmware/baselines/ref_c && mkdir -p build-host
for m in tern80 tern80z70 tern80z90 int8_80; do
  ${HOST_CC:-gcc} -O2 -std=c11 -Wall -Wextra -I../../../train/export -DMODEL_HEADER="\"$m.h\"" ref_infer.c host_main.c -o build-host/ref_host_$m.exe
done
```

`harness/phase0_matrix.sh` builds, flashes and measures all of the configurations.
