# -*- coding: utf-8 -*-
"""
All remaining L-corner experiments under fully random target motion
('rand' profile), 30 cm margin, seeds 2000-2039 (2026-09-29).
Complements metrics_rand_margin30_2026-09-29.json (baseline_cv, full_none,
full_cv, full_ca). run_metrics_test.py is not modified; run_episode is
recompiled with a REPAIR switch (as in run_ablation2_margin30.py).
"""
import os
import sys
import json
import time
import inspect
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics_test as m     # noqa: E402

src = inspect.getsource(m.run_episode)
a = 'if EDGEMEAS and mean is not None and w_good is not None:\n                    u = edge_u'
b = 'if EDGEMEAS and REPAIR and mean is not None and w_good is not None:\n                    u = edge_u'
c = '                else:\n                    meas = None                       # plain rejection'
d = '                elif REPAIR:\n                    meas = None                       # plain rejection'
assert src.count(a) == 1 and src.count(c) == 1, 'run_episode changed; patch no longer applies'
exec(compile(src.replace(a, b).replace(c, d), m.__file__, 'exec'), m.__dict__)

ORIG_CA = m.KalmanFilterCA


class CABlind0(ORIG_CA):
    def __init__(self):
        super().__init__(coast_decay=0.0)


BASE = dict(VCAP=False, EDGEMEAS=False, EDGEAIM=False, BLINDSPEED=False, MAPCLAMP=False, REPAIR=True)
FULL = dict(BASE, VCAP=True, EDGEMEAS=True, EDGEAIM=True, BLINDSPEED=True, MAPCLAMP=True)


def cfg(base, **kw):
    x = dict(base)
    x.update(kw)
    return x


# (tag, mode, flags, ca_blind0)
JOBS = [
    # motion-model test on the baseline (baseline_cv already run)
    ('base_none', 'none', BASE, False),
    ('base_ca', 'ca', BASE, False),
    ('base_cv_r01', 'cv_r01', BASE, False),
    ('base_ca_r01', 'ca_r01', BASE, False),
    # development stages 3-5 (calibrated noise)
    ('st3_cv_r01', 'cv_r01', cfg(BASE, VCAP=True), False),
    ('st3_ca_r01', 'ca_r01', cfg(BASE, VCAP=True), False),
    ('st4_cv_r01', 'cv_r01', cfg(BASE, VCAP=True, EDGEMEAS=True, EDGEAIM=True), False),
    ('st4_ca_r01', 'ca_r01', cfg(BASE, VCAP=True, EDGEMEAS=True, EDGEAIM=True), False),
    ('st5_cv_r01', 'cv_r01', FULL, False),
    ('st5_ca_r01', 'ca_r01', FULL, False),
    # leave-one-out on the full system (CV)
    ('lo_norepair', 'cv', cfg(FULL, REPAIR=False), False),
    ('lo_noedgeaim', 'cv', cfg(FULL, EDGEAIM=False), False),
    ('lo_noblindspeed', 'cv', cfg(FULL, BLINDSPEED=False), False),
    ('lo_novcap', 'cv', cfg(FULL, VCAP=False), False),
    ('lo_noclamp_cv', 'cv', cfg(FULL, MAPCLAMP=False), False),
    # CA mechanism
    ('ca_blind0', 'ca', FULL, True),
    ('ca_noclamp', 'ca', cfg(FULL, MAPCLAMP=False), False),
    # steering while hidden: follow the prediction
    ('follow_em_clamp', 'cv', cfg(FULL, EDGEAIM=False, BLINDSPEED=False), False),
    ('follow_em_noclamp', 'cv', cfg(FULL, EDGEAIM=False, BLINDSPEED=False, MAPCLAMP=False), False),
    ('follow_noem_noclamp', 'cv', cfg(FULL, EDGEAIM=False, BLINDSPEED=False, EDGEMEAS=False, MAPCLAMP=False), False),
]
SEEDS = list(range(2000, 2040))


def main():
    runner, ucv, z_trk = m.boot()
    out = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                        'metrics_rand_all_margin30_2026-09-29.json')
    try:
        for tag, mode, flags, blind0 in JOBS:
            for k, v in flags.items():
                setattr(m, k, v)
            m.KalmanFilterCA = CABlind0 if blind0 else ORIG_CA
            for seed in SEEDS:
                t0 = time.time()
                res = m.run_episode(ucv, z_trk, mode, seed, 'rand')
                res.update(cfg=tag, wall_s=round(time.time() - t0, 1))
                out.append(res)
                print('[ep]', json.dumps(res))
                sys.stdout.flush()
            with open(path, 'w') as f:          # checkpoint after each config
                json.dump(out, f, indent=1)
    finally:
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('saved:', path)
        runner.close()
    for tag, *_ in JOBS:
        rs = [r for r in out if r['cfg'] == tag]
        rq = [r['reacq'] for r in rs if r['reacq'] is not None]
        print(f"{tag:20s} AR={np.mean([r['AR'] for r in rs]):6.1f} SR={np.mean([r['SR'] for r in rs]):.2f} "
              f"hits={np.mean([r['hits'] for r in rs]):5.1f} reacq={np.mean(rq) if rq else float('nan'):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
