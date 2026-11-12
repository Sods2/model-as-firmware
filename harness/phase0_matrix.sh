#!/usr/bin/env bash
# Build, flash and measure every Phase 0 configuration on the Pico 2 (Git Bash, from the repo root).
#
#   harness/phase0_matrix.sh profile   # 200 images per build, stage profiling on  -> results/profile.csv (+ _stages.csv)
#   harness/phase0_matrix.sh final     # 10k images per build, profiling off       -> results/phase0.csv
#
# Builds land in firmware/baselines/<build>/build-<model>-<flash|ram>[-prof]/.
set -euo pipefail
MODE=${1:-final}
PORT=${PORT:-COM3}
P=~/.pico-sdk
export PATH="$P/cmake/v4.3.4/bin:$P/ninja/v1.13.2:$P/toolchain/15_2_Rel1/bin:$PATH"
PT=$P/picotool/2.3.1/picotool/picotool.exe
PY=train/.venv/Scripts/python.exe

if [ "$MODE" = profile ]; then N=200; PROF=ON; CSV=results/profile.csv; SUFFIX=-prof
else N=10000; PROF=OFF; CSV=results/phase0.csv; SUFFIX=; fi

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

# build model mem golden
run() {
  local build=$1 model=$2 mem=$3 golden=$4
  local dir=firmware/baselines/$build/build-$model-$mem$SUFFIX
  local ram=OFF; [ "$mem" = ram ] && ram=ON
  cmake -S firmware/baselines/$build -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DMODEL=$model -DRUN_FROM_RAM=$ram -DPROFILE_STAGES=$PROF >/dev/null
  ninja -C "$dir" >/dev/null
  flash "$dir/$build.uf2"
  $PY harness/run.py --model $model --target pico --port "$PORT" --golden $golden --n $N \
      --elf "$dir/$build.elf" --csv "$CSV" | grep -v -E '^  [0-9]+/[0-9]+ '
}

run ref_c    tern80    flash ref
run ref_c    tern80    ram   ref
run cmsis_nn tern80    flash s8
run cmsis_nn tern80    ram   s8
if [ "$MODE" = final ]; then
  run ref_c  tern80z70 flash ref
  run ref_c  tern80z90 flash ref
fi
