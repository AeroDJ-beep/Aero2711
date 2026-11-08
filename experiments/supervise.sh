#!/usr/bin/env bash
# Outer supervisor: keep relaunching the watchdog for a preset until its log says SWEEP DONE (the watchdog itself
# gives up after 8 attempts; native crashes of the UnrealCV client can exhaust that on a long run).
# usage: bash supervise.sh <seeds> <preset> [max_rounds=6]
SEEDS=$1; PRESET=$2; MAXR=${3:-6}
cd /d/Aero2711
for round in $(seq 1 $MAXR); do
  while ps -ef | grep -v grep | grep -q "run_sweep_watchdog.sh $SEEDS $PRESET"; do sleep 60; done
  if grep -q "SWEEP DONE" results/logs/uz_sweep_${PRESET}.log 2>/dev/null; then echo "[supervise] $PRESET done $(date +%T)"; exit 0; fi
  echo "[supervise] relaunching watchdog for $PRESET (round $round) $(date +%T)"
  bash experiments/run_sweep_watchdog.sh "$SEEDS" "$PRESET" >> results/logs/uz_watchdog_${PRESET}.log 2>&1
done
echo "[supervise] gave up on $PRESET"
