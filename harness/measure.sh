#!/usr/bin/env bash
# Build, flash and measure any set of builds on the Pico 2 (Git Bash, from the repo root).
#
#   harness/measure.sh <profile|final|energy> <csv> <build:model:mem[:ternary[:int4[:conv[:stream[:int8]]]]]> ...
#     build  ref_c | cmsis_nn | cmsis_nn_s4 (fc1 as CMSIS-NN s4) | mafc | mafc2 (dual-core) | mafc_s (SMLAD) | mafc2_s (dual-core + SMLAD)
#     model  tern80 | tern80z70 | tern80z90 | int8_80 (all linear layers int8)
#     mem    flash | ram          (ram = copy_to_ram: code and weights in SRAM)
#     ternary  base3 (default) | sparse | code | dense   (mafc builds: format of fc1, compile.py --ternary)
#     int4     packed (default) | code | dense           (mafc builds: format of fc2/fc3, compile.py --int4)
#     conv     data (default) | code                     (mafc SIMD builds: conv weights, compile.py --conv)
#     stream   nostream (default) | stream               (mafc: dense layers streamed from flash, compile.py --stream)
#     int8     plain (default) | code | dense            (mafc builds: format of int8 layers, compile.py --int8)
#
#   profile: 200 images, stage counters on;  final: 10k images, counters off.
#   energy: builds as final, then harness/energy.py (interactive: USB meter readings) instead of run.py;
#           <csv> is the energy CSV, ENERGY_ARGS="--secs 600 --reading mwh" passes options through.
#   e.g.  harness/measure.sh final results/phase1.csv mafc:tern80:flash mafc:tern80:ram
#   NOTES="..." harness/measure.sh ...   puts a label in the CSV's notes column
#   FLASH=usb harness/measure.sh ...     flashes with picotool instead of SWD (default auto: SWD, then picotool)
set -euo pipefail
MODE=$1; CSV=$2; shift 2
PORT=${PORT:-COM3}
P=~/.pico-sdk
export PATH="$P/cmake/v4.3.4/bin:$P/ninja/v1.13.2:$P/toolchain/15_2_Rel1/bin:$PATH"
PY=train/.venv/Scripts/python.exe

if [ "$MODE" = profile ]; then N=200; PROF=ON; SUFFIX=-prof; else N=10000; PROF=OFF; SUFFIX=; fi
case $MODE in profile|final|energy) ;; *) echo "unknown mode $MODE"; exit 1 ;; esac

. harness/flash.sh   # flash(): SWD via the Debug Probe, picotool fallback (FLASH=auto|swd|usb)

for spec in "$@"; do
  IFS=: read -r build model mem tern i4 cv st i8 <<< "$spec"
  tern=${tern:-base3}; i4=${i4:-packed}; cv=${cv:-data}; st=${st:-nostream}; i8=${i8:-plain}
  extra=
  case $build in
    ref_c)    src=firmware/baselines/ref_c;    golden=ref ;;
    cmsis_nn) src=firmware/baselines/cmsis_nn; golden=s8 ;;
    cmsis_nn_s4) src=firmware/baselines/cmsis_nn; golden=s8; extra="-DFC1_S4=ON" ;;
    mafc)     src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=OFF -DSIMD=OFF" ;;
    mafc2)    src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=ON -DSIMD=OFF" ;;
    mafc_s)   src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=OFF -DSIMD=ON" ;;
    mafc2_s)  src=firmware/rp2350/mafc;        golden=ref; extra="-DDUAL_CORE=ON -DSIMD=ON" ;;
    *) echo "unknown build $build"; exit 1 ;;
  esac
  case $build in mafc*) target=mafc ;; cmsis_nn*) target=cmsis_nn ;; *) target=$build ;; esac
  dir=$src/build-$build-$model-$mem$SUFFIX
  [ "$tern" != base3 ] && dir=$src/build-$build-$model-$mem-$tern$SUFFIX
  [ "$i4" != packed ] && dir=$src/build-$build-$model-$mem-$tern-i4$i4$SUFFIX
  [ "$cv" != data ] && dir=$src/build-$build-$model-$mem-$tern-i4$i4-cv$cv$SUFFIX
  [ "$st" = stream ] && dir=$src/build-$build-$model-$mem-$tern-i4$i4-cv$cv-stream$SUFFIX
  [ "$i8" != plain ] && dir=$src/build-$build-$model-$mem-$tern-i4$i4-cv$cv-$st-i8$i8$SUFFIX
  stream_flag=OFF; [ "$st" = stream ] && stream_flag=ON
  ram=OFF; [ "$mem" = ram ] && ram=ON
  cmake -S $src -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=$model -DRUN_FROM_RAM=$ram -DPROFILE_STAGES=$PROF -DTERNARY=$tern -DINT4=$i4 -DCONV=$cv -DSTREAM=$stream_flag -DINT8=$i8 ${extra:-} >/dev/null
  ninja -C "$dir" >/dev/null
  flash "$dir/$target.uf2"
  if [ "$MODE" = energy ]; then
    $PY harness/energy.py --port "$PORT" --model $model --csv "$CSV" --notes "${NOTES:-}" ${ENERGY_ARGS:-}
    continue
  fi
  $PY harness/run.py --model $model --target pico --port "$PORT" --golden $golden --n $N \
      --elf "$dir/$target.elf" --csv "$CSV" --notes "${NOTES:-}" | grep -v -E '^  [0-9]+/[0-9]+ '
done
