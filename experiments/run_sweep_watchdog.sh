#!/usr/bin/env bash
# Run a resumable uz_sweep.py preset; if its log stops growing for STALL seconds (the UnrealCV
# client occasionally blocks forever on a reply), kill the sweep and the engine and restart it.
# usage: bash run_sweep_watchdog.sh <seeds> <preset> [stall_s=600]
SEEDS=$1; PRESET=$2; STALL=${3:-600}
cd /d/Aero2711
PY="/c/Users/AUFB/miniconda3/envs/unrealzoo/python.exe"
LOG=results/logs/uz_sweep_${PRESET}.log; ERR=results/logs/uz_sweep_${PRESET}.err
kill_all() {
  powershell -NoProfile -ExecutionPolicy Bypass -File experiments/kill_sweep.ps1
}
for attempt in 1 2 3 4 5 6 7 8; do
  kill_all >/dev/null; echo "[watchdog] start attempt $attempt $(date +%T)"
  "$PY" experiments/uz_sweep.py "$SEEDS" "$PRESET" >> "$LOG" 2>> "$ERR" &
  PID=$!; T0=$(date +%s)
  while kill -0 $PID 2>/dev/null; do
    sleep 30
    last=$(stat -c %Y "$LOG"); [ $last -lt $T0 ] && last=$T0     # count from this attempt's start
    age=$(( $(date +%s) - last ))
    if [ $age -gt $STALL ]; then
      echo "[watchdog] log stalled ${age}s -> restart"; kill_all; break
    fi
  done
  wait $PID 2>/dev/null
  if grep -q "SWEEP DONE" "$LOG"; then echo "[watchdog] SWEEP DONE"; exit 0; fi
  kill_all; sleep 10
done
echo "[watchdog] gave up"
