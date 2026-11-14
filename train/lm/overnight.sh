#!/usr/bin/env bash
# Overnight run (Git Bash, from anywhere): train both MafLM twins on Atome's token budget, then for each one check
# the integer reference against PyTorch, export it, generate the firmware data and check the PC build bit-exact.
#
#   bash train/lm/overnight.sh                 # 61,000 steps x 32K tokens = 2.0B tokens each, ~6.5 h for both
#   STEPS=30500 bash train/lm/overnight.sh     # half the budget, ~3.3 h
#   STEPS=40 EVAL=20 TAG=dry bash train/lm/overnight.sh   # a dry run of the whole pipeline in a couple of minutes
#
# Everything goes to train/runs/: lm_<twin>{,_<TAG>}.pt/.json/.npz/.log, and overnight{,_<TAG>}.txt as the summary.
# The firmware data goes to firmware/rp2350/maflm/gen/maflm_lm_<twin>{_<TAG>}.c/.h. A failure in one twin doesn't stop
# the other. Tomorrow: flash and measure with harness/lm_run.py (notes/lm.md).
set -u
cd "$(dirname "$0")/../.."
PY=train/.venv/Scripts/python.exe
STEPS=${STEPS:-61000}
EVAL=${EVAL:-2000}
TAG=${TAG:-}
SUF=${TAG:+_$TAG}
RUNS=train/runs
SUM=$RUNS/overnight$SUF.txt
mkdir -p $RUNS
export PYTHONIOENCODING=utf-8

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUM"; }

say "overnight run: STEPS=$STEPS EVAL=$EVAL TAG=${TAG:-none}, commit $(git rev-parse --short HEAD)"
for twin in dense tree; do
  name=lm_$twin$SUF
  log=$RUNS/$name.log
  say "$twin: training -> $RUNS/$name.pt (log $log)"
  if ! $PY -u train/lm/train.py --ffn $twin --steps "$STEPS" --eval-every "$EVAL" --out $RUNS/$name.pt >"$log" 2>&1; then
    say "$twin: TRAINING FAILED (last lines of the log follow)"; tail -5 "$log" | tee -a "$SUM"; continue
  fi
  grep -E "^best val" "$log" | sed "s/^/  $twin: /" | tee -a "$SUM"

  say "$twin: integer reference, export, samples"
  if ! $PY train/lm/intref.py --pt $RUNS/$name.pt --npz $RUNS/$name.npz --samples 150 >>"$log" 2>&1; then
    say "$twin: INTEGER REFERENCE FAILED"; tail -5 "$log" | tee -a "$SUM"; continue
  fi
  sed -n '/float (fp32, fake-quant)/,$p' "$log" | sed "s/^/  /" | tee -a "$SUM"

  say "$twin: compile firmware data and check the PC build"
  if $PY compiler/compile_lm.py --npz $RUNS/$name.npz --name $name >>"$log" 2>&1 &&
     $PY harness/lm_host_check.py --name $name --npz $RUNS/$name.npz --n 2 >>"$log" 2>&1 &&
     $PY harness/lm_host_check.py --name $name --npz $RUNS/$name.npz --n 2 --dual >>"$log" 2>&1; then
    say "$twin: PC build bit-exact (one and two cores); firmware data in firmware/rp2350/maflm/gen/maflm_$name.c"
  else
    say "$twin: COMPILE OR PC CHECK FAILED"; tail -5 "$log" | tee -a "$SUM"
  fi
done
say "done"
