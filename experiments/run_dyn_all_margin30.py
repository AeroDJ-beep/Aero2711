# -*- coding: utf-8 -*-
"""
All L-corner experiments re-run with the acceleration-limited camera (--dyn),
random target motion, 30 cm margin, seeds 2000-2039 (2026-09-30).

Every configuration uses the same camera dynamics (velocity is a state,
|dv| <= A_CAM*DT per step). The safety layer is the braking-distance law
inside dyn_step; SAFETY=False removes only that law (inertia stays), which
is used for the baseline and for the safety ablation.
run_metrics_test.py is not modified; run_episode and dyn_step are
recompiled with REPAIR / SAFETY switches.
"""
import os
import sys
import json
import time
import inspect
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics_test as m     # noqa: E402

# --- REPAIR switch (measurement repair on/off, fade flag kept) ---
src = inspect.getsource(m.run_episode)
a = 'if EDGEMEAS and mean is not None and w_good is not None:\n                    u = edge_u'
b = 'if EDGEMEAS and REPAIR and mean is not None and w_good is not None:\n                    u = edge_u'
c = '                else:\n                    meas = None                       # plain rejection'
d = '                elif REPAIR:\n                    meas = None                       # plain rejection'
assert src.count(a) == 1 and src.count(c) == 1, 'run_episode changed; patch no longer applies'
exec(compile(src.replace(a, b).replace(c, d), m.__file__, 'exec'), m.__dict__)

# --- SAFETY switch (braking-distance law on/off, inertia kept) ---
dsrc = inspect.getsource(m.dyn_step)
e = '    u_cap = math.sqrt(lim * lim + 2.0 * A_CAM * s_) - lim\n'
f = e + '    if not SAFETY:\n        u_cap = 1e9\n'
assert dsrc.count(e) == 1, 'dyn_step changed; patch no longer applies'
exec(compile(dsrc.replace(e, f), m.__file__, 'exec'), m.__dict__)

ORIG_CA = m.KalmanFilterCA


class CABlind0(ORIG_CA):
    def __init__(self):
        super().__init__(coast_decay=0.0)


BASE = dict(DYN=True, VCAP=False, SAFETY=False, EDGEMEAS=False, EDGEAIM=False,
            BLINDSPEED=False, MAPCLAMP=False, REPAIR=True)
FULL = dict(BASE, SAFETY=True, EDGEMEAS=True, EDGEAIM=True, BLINDSPEED=True, MAPCLAMP=True)


def cfg(base, **kw):
    x = dict(base)
    x.update(kw)
    return x


# (tag, mode, flags, ca_blind0, keep_steplog)
JOBS = [
    ('baseline_cv', 'cv', BASE, False, False),
    ('full_none', 'none', FULL, False, False),
    ('full_cv', 'cv', FULL, False, True),
    ('full_ca', 'ca', FULL, False, False),
    ('base_none', 'none', BASE, False, False),
    ('base_ca', 'ca', BASE, False, False),
    ('base_cv_r01', 'cv_r01', BASE, False, False),
    ('base_ca_r01', 'ca_r01', BASE, False, False),
    ('st3_cv_r01', 'cv_r01', cfg(BASE, SAFETY=True), False, False),
    ('st3_ca_r01', 'ca_r01', cfg(BASE, SAFETY=True), False, False),
    ('st4_cv_r01', 'cv_r01', cfg(BASE, SAFETY=True, EDGEMEAS=True, EDGEAIM=True), False, False),
    ('st4_ca_r01', 'ca_r01', cfg(BASE, SAFETY=True, EDGEMEAS=True, EDGEAIM=True), False, False),
    ('st5_cv_r01', 'cv_r01', FULL, False, False),
    ('st5_ca_r01', 'ca_r01', FULL, False, False),
    ('lo_norepair', 'cv', cfg(FULL, REPAIR=False), False, False),
    ('lo_noedgeaim', 'cv', cfg(FULL, EDGEAIM=False), False, False),
    ('lo_noblindspeed', 'cv', cfg(FULL, BLINDSPEED=False), False, False),
    ('lo_nosafety', 'cv', cfg(FULL, SAFETY=False), False, False),
    ('lo_noclamp_cv', 'cv', cfg(FULL, MAPCLAMP=False), False, False),
    ('ca_blind0', 'ca', FULL, True, False),
    ('ca_noclamp', 'ca', cfg(FULL, MAPCLAMP=False), False, False),
    ('follow_em_clamp', 'cv', cfg(FULL, EDGEAIM=False, BLINDSPEED=False), False, False),
    ('follow_em_noclamp', 'cv', cfg(FULL, EDGEAIM=False, BLINDSPEED=False, MAPCLAMP=False), False, False),
    ('follow_noem_noclamp', 'cv', cfg(FULL, EDGEAIM=False, BLINDSPEED=False, EDGEMEAS=False, MAPCLAMP=False), False, False),
]
SEEDS = list(range(2000, 2040))


def blind_stats(steplog):
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
    return dict(n_blind=int(n), dv_max=float(seg.max() - seg.min()))


def main():
    runner, ucv, z_trk = m.boot()
    out = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                        'metrics_dyn_all_margin30_2026-09-30.json')
    try:
        for tag, mode, flags, blind0, keep in JOBS:
            for k, v in flags.items():
                setattr(m, k, v)
            m.KalmanFilterCA = CABlind0 if blind0 else ORIG_CA
            for seed in SEEDS:
                log = [] if keep else None
                t0 = time.time()
                res = m.run_episode(ucv, z_trk, mode, seed, 'rand', steplog=log)
                res.update(cfg=tag, wall_s=round(time.time() - t0, 1))
                if keep:
                    res['blind'] = blind_stats(log)
                out.append(res)
                print('[ep]', json.dumps(res))
                sys.stdout.flush()
            with open(path, 'w') as fh:
                json.dump(out, fh, indent=1)
    finally:
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('saved:', path)
        runner.close()
    for tag, *_ in JOBS:
        rs = [r for r in out if r['cfg'] == tag]
        rq = [r['reacq'] for r in rs if r['reacq'] is not None]
        print(f"{tag:20s} AR={np.mean([r['AR'] for r in rs]):6.1f} SR={np.mean([r['SR'] for r in rs]):.2f} "
              f"hits={np.mean([r['hits'] for r in rs]):5.1f} reacq={np.mean(rq) if rq else float('nan'):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
