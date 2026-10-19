#!/usr/bin/env bash
# Wait until no OVServo (ovservo env) python process has been seen for QUIET seconds, then run a
# uz_sweep preset under the watchdog.  One project on the machine at a time (shared GPU / engines).
# usage: bash wait_then_run.sh <seeds> <preset> [quiet_s=600]
SEEDS=$1; PRESET=$2; QUIET=${3:-600}
last=$(date +%s)
echo "[wait] waiting for OVServo jobs to finish $(date +%T)"
while true; do
  if ps -ef | grep -v grep | grep -qi "ovservo"; then last=$(date +%s); fi
  idle=$(( $(date +%s) - last ))
  if [ $idle -ge $QUIET ]; then break; fi
  sleep 60
done
echo "[wait] machine quiet for ${QUIET}s -> starting $PRESET $(date +%T)"
cd /d/Aero2711 && bash experiments/run_sweep_watchdog.sh "$SEEDS" "$PRESET"
