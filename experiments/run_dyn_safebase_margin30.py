# -*- coding: utf-8 -*-
"""
Baseline WITH the safety layer (collision avoidance is a prerequisite for
every configuration). Acceleration-limited camera, random motion, 40 seeds.
Reuses the patched modules of run_dyn_all_margin30.py.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_dyn_all_margin30 as base   # noqa: E402  (applies the REPAIR/SAFETY patches)

m = base.m
SAFE_BASE = base.cfg(base.BASE, SAFETY=True)
JOBS = [('bs_cv', 'cv'), ('bs_ca', 'ca'), ('bs_none', 'none')]


def main():
    runner, ucv, z_trk = m.boot()
    out = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                        'metrics_dyn_safebase_margin30_2026-09-30.json')
    try:
        for k, v in SAFE_BASE.items():
            setattr(m, k, v)
        m.KalmanFilterCA = base.ORIG_CA
        for tag, mode in JOBS:
            for seed in base.SEEDS:
                t0 = time.time()
                res = m.run_episode(ucv, z_trk, mode, seed, 'rand')
                res.update(cfg=tag, wall_s=round(time.time() - t0, 1))
                out.append(res)
                print('[ep]', json.dumps(res)); sys.stdout.flush()
    finally:
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('saved:', path)
        runner.close()
    for tag, _ in JOBS:
        rs = [r for r in out if r['cfg'] == tag]
        rq = [r['reacq'] for r in rs if r['reacq'] is not None]
        print(f"{tag:8s} AR={np.mean([r['AR'] for r in rs]):6.1f} SR={np.mean([r['SR'] for r in rs]):.2f} "
              f"hits={np.mean([r['hits'] for r in rs]):5.1f} reacq={np.mean(rq):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
