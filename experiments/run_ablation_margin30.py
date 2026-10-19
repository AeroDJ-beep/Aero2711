# -*- coding: utf-8 -*-
"""
Remaining controls for the progress report (30 cm margin, 2026-09-29).
Imports run_metrics_test.py without modifying it.

  1+2. Leave-one-out on the full architecture (CV): drop edgemeas / edgeaim /
       blindspeed / vcap, P2 + P2a, seeds 2000-2019.  'no_edgeaim_blindspeed'
       = repaired measurements with belief-driven steering (decoupling test).
  3.   Normal targets (profile 'rand'): baseline CV vs full architecture CV.
  4.   Extra P2a seeds 2020-2039: cv, ca, ca_blind0, cv_noclamp, ca_noclamp.
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
S0 = list(range(2000, 2020))
S1 = list(range(2020, 2040))
ORIG_CA = m.KalmanFilterCA


class CABlind0(ORIG_CA):
    def __init__(self):
        super().__init__(coast_decay=0.0)


def without(*keys):
    d = dict(FULL)
    for k in keys:
        d[k] = False
    return d


# (tag, mode, flags, profiles, seeds, blind0)
JOBS = [
    ('full_cv', 'cv', FULL, ['p2', 'p2a'], S0, False),
    ('no_edgemeas', 'cv', without('EDGEMEAS'), ['p2', 'p2a'], S0, False),
    ('no_edgeaim', 'cv', without('EDGEAIM'), ['p2', 'p2a'], S0, False),
    ('no_blindspeed', 'cv', without('BLINDSPEED'), ['p2', 'p2a'], S0, False),
    ('no_vcap', 'cv', without('VCAP'), ['p2', 'p2a'], S0, False),
    ('no_edgeaim_blindspeed', 'cv', without('EDGEAIM', 'BLINDSPEED'), ['p2', 'p2a'], S0, False),
    ('rand_baseline_cv', 'cv', NONE, ['rand'], S0, False),
    ('rand_full_cv', 'cv', FULL, ['rand'], S0, False),
    ('xs_cv', 'cv', FULL, ['p2a'], S1, False),
    ('xs_ca', 'ca', FULL, ['p2a'], S1, False),
    ('xs_ca_blind0', 'ca', FULL, ['p2a'], S1, True),
    ('xs_cv_noclamp', 'cv', without('MAPCLAMP'), ['p2a'], S1, False),
    ('xs_ca_noclamp', 'ca', without('MAPCLAMP'), ['p2a'], S1, False),
]


def main():
    runner, ucv, z_trk = m.boot()
    out = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                        'metrics_ablation_margin30_2026-09-29.json')
    try:
        for tag, mode, flags, profiles, seeds, blind0 in JOBS:
            for k, v in flags.items():
                setattr(m, k, v)
            m.KalmanFilterCA = CABlind0 if blind0 else ORIG_CA
            for prof in profiles:
                for seed in seeds:
                    t0 = time.time()
                    res = m.run_episode(ucv, z_trk, mode, seed, prof)
                    res.update(cfg=tag, wall_s=round(time.time() - t0, 1))
                    out.append(res)
                    print('[ep]', json.dumps(res))
                    sys.stdout.flush()
    finally:
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('saved:', path)
        runner.close()
    for tag, _, _, profiles, _, _ in JOBS:
        for prof in profiles:
            rs = [r for r in out if r['cfg'] == tag and r['profile'] == prof]
            rq = [r['reacq'] for r in rs if r['reacq'] is not None]
            print(f"{tag:22s} {prof:4s} AR={np.mean([r['AR'] for r in rs]):6.1f} "
                  f"SR={np.mean([r['SR'] for r in rs]):.2f} hits={np.mean([r['hits'] for r in rs]):5.1f} "
                  f"reacq={np.mean(rq) if rq else float('nan'):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
