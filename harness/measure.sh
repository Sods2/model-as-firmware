#!/usr/bin/env bash
# Build, flash and measure any set of builds on the Pico 2 (Git Bash, from the repo root).
#
#   harness/measure.sh <profile|final> <csv> <build:model:mem[:ternary[:int4[:conv]]]> ...
#     build  ref_c | cmsis_nn | mafc | mafc2 (dual-core) | mafc_s (SMLAD) | mafc2_s (dual-core + SMLAD)
#     model  tern80 | tern80z70 | tern80z90
#     mem    flash | ram          (ram = copy_to_ram: code and weights in SRAM)
#     ternary  base3 (default) | sparse | code | dense   (mafc builds: format of fc1, compile.py --ternary)
#     int4     packed (default) | code | dense           (mafc builds: format of fc2/fc3, compile.py --int4)
#     conv     data (default) | code                     (mafc SIMD builds: conv weights, compile.py --conv)
#
#   profile: 200 images, stage counters on;  final: 10k images, counters off.
#   e.g.  harness/measure.sh final results/phase1.csv mafc:tern80:flash mafc:tern80:ram
#   NOTES="..." harness/measure.sh ...   puts a label in the CSV's notes column
set -euo pipefail
MODE=$1; CSV=$2; shift 2
PORT=${PORT:-COM3}
P=~/.pico-sdk
export PATH="$P/cmake/v4.3.4/bin:$P/ninja/v1.13.2:$P/toolchain/15_2_Rel1/bin:$PATH"
PT=$P/picotool/2.3.1/picotool/picotool.exe
PY=train/.venv/Scripts/python.exe

if [ "$MODE" = profile ]; then N=200; PROF=ON; SUFFIX=-prof; else N=10000; PROF=OFF; SUFFIX=; fi

wait_port() { for _ in $(seq 1 20); do $PY -m serial.tools.list_ports | grep -q "$PORT" && { sleep 1; return 0; }; sleep 1; done; echo "board did not come back on $PORT"; return 1; }
flash() {
  # -f reboots the running firmware into BOOTSEL; Windows is sometimes too slow to see it, so retry once without -f
  if ! "$PT" load -f -x "$1" >/dev/null 2>&1; then
    local ok=0
    for _ in 1 2 3 4 5; do sleep 2; "$PT" load -x "$1" >/dev/null 2>&1 && { ok=1; break; }; done
    [ $ok = 1 ] || { echo "flashing $1 failed"; return 1; }
  fi
  wait_port
}

for spec in "$@"; do
  IFS=: read -r build model mem tern i4 cv <<< "$spec"
  tern=${tern:-base3}; i4=${i4:-packed}; cv=${cv:-data}
  extra=
  case $build in
    ref_c)    src=firmware/baselines/ref_c;    golden=ref ;;
    cmsis_nn) src=firmware/baselines/cmsis_nn; golden=s8 ;;
    mafc)     src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=OFF -DSIMD=OFF" ;;
    mafc2)    src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=ON -DSIMD=OFF" ;;
    mafc_s)   src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=OFF -DSIMD=ON" ;;
    mafc2_s)  src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=ON -DSIMD=ON" ;;
    *) echo "unknown build $build"; exit 1 ;;
  esac
  case $build in mafc*) target=mafc ;; *) target=$build ;; esac
  dir=$src/build-$build-$model-$mem$SUFFIX
  [ "$tern" != base3 ] && dir=$src/build-$build-$model-$mem-$tern$SUFFIX
  [ "$i4" != packed ] && dir=$src/build-$build-$model-$mem-$tern-i4$i4$SUFFIX
  [ "$cv" != data ] && dir=$src/build-$build-$model-$mem-$tern-i4$i4-cv$cv$SUFFIX
  ram=OFF; [ "$mem" = ram ] && ram=ON
  cmake -S $src -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=$model -DRUN_FROM_RAM=$ram -DPROFILE_STAGES=$PROF -DTERNARY=$tern -DINT4=$i4 -DCONV=$cv ${extra:-} >/dev/null
  ninja -C "$dir" >/dev/null
  flash "$dir/$target.uf2"
  $PY harness/run.py --model $model --target pico --port "$PORT" --golden $golden --n $N \
      --elf "$dir/$target.elf" --csv "$CSV" --notes "${NOTES:-}" | grep -v -E '^  [0-9]+/[0-9]+ '
done
