# -*- coding: utf-8 -*-
"""
Fully random target motion ('rand' profile: random route, speed re-drawn in
[40, 99] cm/s every 2-4 s, ramped at up to 80 cm/s^2; nothing tied to the
corner). 30 cm margin, seeds 2000-2039.

Configs: baseline CV (no mechanisms), full system with none / CV / CA.
Each episode also logs the target's speed change during the blind phase,
so results can be split by whether the target actually changed speed while
hidden. Does not modify run_metrics_test.py.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics_test as m     # noqa: E402

FULL = dict(VCAP=True, EDGEMEAS=True, EDGEAIM=True, BLINDSPEED=True, MAPCLAMP=True)
NONE = dict(VCAP=False, EDGEMEAS=False, EDGEAIM=False, BLINDSPEED=False, MAPCLAMP=False)
SEEDS = list(range(2000, 2040))
JOBS = [('baseline_cv', 'cv', NONE), ('full_none', 'none', FULL),
        ('full_cv', 'cv', FULL), ('full_ca', 'ca', FULL)]


def blind_stats(steplog):
    """Longest blind run: length, target speed at its start and end, and the
    largest speed change inside it (cm/s)."""
    P = np.array([(r[3], r[4]) for r in steplog])
    vis = np.array([r[5] for r in steplog])
    v = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1) / m.DT]
    best, cur, start = (0, 0), 0, 0
    for k, ok in enumerate(vis):
        if not ok:
            if cur == 0:
                start = k
            cur += 1
            if cur > best[1]:
                best = (start, cur)
        else:
            cur = 0
    s, n = best
    if n < 5:
        return None
    seg = v[max(s - 1, 0):s + n + 1]
    return dict(n_blind=int(n), v_start=float(seg[0]), v_end=float(seg[-1]),
                dv=float(seg[-1] - seg[0]), dv_max=float(seg.max() - seg.min()))


def main():
    runner, ucv, z_trk = m.boot()
    out = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                        'metrics_rand_margin30_2026-09-29.json')
    try:
        for tag, mode, flags in JOBS:
            for k, val in flags.items():
                setattr(m, k, val)
            for seed in SEEDS:
                log = []
                t0 = time.time()
                res = m.run_episode(ucv, z_trk, mode, seed, 'rand', steplog=log)
                res.update(cfg=tag, wall_s=round(time.time() - t0, 1), blind=blind_stats(log))
                out.append(res)
                print('[ep]', json.dumps(res))
                sys.stdout.flush()
    finally:
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('saved:', path)
        runner.close()
    for tag, _, _ in JOBS:
        rs = [r for r in out if r['cfg'] == tag]
        rq = [r['reacq'] for r in rs if r['reacq'] is not None]
        print(f"{tag:12s} AR={np.mean([r['AR'] for r in rs]):6.1f} SR={np.mean([r['SR'] for r in rs]):.2f} "
              f"hits={np.mean([r['hits'] for r in rs]):5.1f} reacq={np.mean(rq):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
