# -*- coding: utf-8 -*-
"""
Two follow-up controls (30 cm margin, 2026-09-29). run_metrics_test.py is not
modified; a patched copy of run_episode is compiled in its namespace.

A. Decoupling test (C3): belief-driven steering (no edgeaim, no blindspeed),
   with / without measurement repair, with / without map clamp.
   ('bd_em_clamp' = no_edgeaim_blindspeed from the first ablation run.)
B. Confound check: fade-frame flag kept (edgeaim/blindspeed still switch on
   during fade-out) but the measurement repair itself disabled -> the raw
   clipped box is fed to the KF.
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
m.REPAIR = True

S0 = list(range(2000, 2020))
BASE = dict(VCAP=True, MAPCLAMP=True, EDGEMEAS=True, EDGEAIM=True, BLINDSPEED=True, REPAIR=True)


def cfg(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


JOBS = [
    ('bd_noem_clamp', cfg(EDGEAIM=False, BLINDSPEED=False, EDGEMEAS=False)),
    ('bd_em_noclamp', cfg(EDGEAIM=False, BLINDSPEED=False, MAPCLAMP=False)),
    ('bd_noem_noclamp', cfg(EDGEAIM=False, BLINDSPEED=False, EDGEMEAS=False, MAPCLAMP=False)),
    ('fadeflag_norepair', cfg(REPAIR=False)),
]


def main():
    runner, ucv, z_trk = m.boot()
    out = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                        'metrics_ablation2_margin30_2026-09-29.json')
    try:
        for tag, flags in JOBS:
            for k, v in flags.items():
                setattr(m, k, v)
            for prof in ('p2', 'p2a'):
                for seed in S0:
                    t0 = time.time()
                    res = m.run_episode(ucv, z_trk, 'cv', seed, prof)
                    res.update(cfg=tag, wall_s=round(time.time() - t0, 1))
                    out.append(res)
                    print('[ep]', json.dumps(res))
                    sys.stdout.flush()
    finally:
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('saved:', path)
        runner.close()
    for tag, _ in JOBS:
        for prof in ('p2', 'p2a'):
            rs = [r for r in out if r['cfg'] == tag and r['profile'] == prof]
            rq = [r['reacq'] for r in rs if r['reacq'] is not None]
            print(f"{tag:18s} {prof:4s} AR={np.mean([r['AR'] for r in rs]):6.1f} "
                  f"SR={np.mean([r['SR'] for r in rs]):.2f} reacq={np.mean(rq) if rq else float('nan'):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
