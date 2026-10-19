# -*- coding: utf-8 -*-
"""
Why is CA worse than CV inside the full architecture (30 cm margin runs,
2026-09-28)?  Hypothesis under test: the CA filter carries an acceleration
state into the blind phase and extrapolates it (0.5*a*t^2), and this term is
wrong.  Configurations (all on the full architecture unless noted):

  cv            baseline CV filter
  ca            CA filter (as in the report)
  ca_blind0     CA, acceleration zeroed on every blind frame (coast_decay=0)
  cv_noclamp    CV, full architecture without --mapclamp
  ca_noclamp    CA, full architecture without --mapclamp

The KF classes are wrapped to log the state every step, so for each episode
we get the filter state at blind onset and the signed re-acquisition error.
Does not modify run_metrics_test.py.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics_test as m     # noqa: E402

LOG = []


def logged(base, **kw):
    class K(base):
        def __init__(self):
            super().__init__(**kw)

        def predict(self, mean, cov, coasting=False):
            if isinstance(self, m.KalmanFilterCA):
                mean, cov = super().predict(mean, cov, coasting=coasting)
            else:
                mean, cov = super().predict(mean, cov)
            LOG.append(('p', mean.copy()))
            return mean, cov

        def update(self, mean, cov, meas):
            LOG.append(('u', mean.copy(), np.array(meas, float).copy()))
            mean, cov = super().update(mean, cov, meas)
            return mean, cov
    return K


def analyse(log):
    """Find the first update after >=5 consecutive predict-only steps.
    Return state at blind onset, signed error and the blind-phase
    displacement split into velocity and acceleration parts."""
    steps = []                      # (predicted mean, update record or None)
    for i, rec in enumerate(log):
        if rec[0] == 'p':
            nxt = log[i + 1] if i + 1 < len(log) and log[i + 1][0] == 'u' else None
            steps.append((rec[1], nxt))
    run, onset = 0, None
    for pred, upd in steps:
        if upd is None:
            if run == 0:
                onset = pred                # state at the first blind step
            run += 1
            continue
        if run >= 5:
            n = run
            ax0 = onset[8] if len(onset) >= 10 else 0.0
            return dict(err=float(upd[1][0] - upd[2][0]), n_blind=n,
                        vx0=float(onset[4]), ax0=float(ax0),
                        acc_part=float(0.5 * ax0 * n * n),
                        vel_part=float(onset[4] * n))
        run = 0
    return None


def main():
    runner, ucv, z_trk = m.boot()
    m.VCAP = m.EDGEMEAS = m.EDGEAIM = m.BLINDSPEED = True
    configs = [
        ('cv', 'cv', m.KalmanFilter, {}, True),
        ('ca', 'ca', m.KalmanFilterCA, {}, True),
        ('ca_blind0', 'ca', m.KalmanFilterCA, {'coast_decay': 0.0}, True),
        ('cv_noclamp', 'cv', m.KalmanFilter, {}, False),
        ('ca_noclamp', 'ca', m.KalmanFilterCA, {'coast_decay': 1.0}, False),
    ]
    orig_cv, orig_ca = m.KalmanFilter, m.KalmanFilterCA
    out = []
    try:
        for tag, mode, base, kw, clamp in configs:
            m.MAPCLAMP = clamp
            m.KalmanFilter = logged(orig_cv) if base is orig_cv else orig_cv
            m.KalmanFilterCA = logged(orig_ca, **kw) if base is orig_ca else orig_ca
            for prof in ('p2', 'p2a'):
                for seed in range(2000, 2020):
                    LOG.clear()
                    t0 = time.time()
                    res = m.run_episode(ucv, z_trk, mode, seed, prof)
                    res.update(cfg=tag, wall_s=round(time.time() - t0, 1))
                    res['mech'] = analyse(LOG)
                    out.append(res)
                    print('[ep]', json.dumps(res))
                    sys.stdout.flush()
    finally:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics',
                            'metrics_ca_mechanism_margin30_2026-09-28.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print('saved:', path)
        runner.close()
    for tag, *_ in configs:
        for prof in ('p2', 'p2a'):
            rs = [r for r in out if r['cfg'] == tag and r['profile'] == prof and r['reacq'] is not None]
            print(f'{prof:4s} {tag:11s} reacq={np.mean([r["reacq"] for r in rs]):6.1f} (n={len(rs)})')


if __name__ == '__main__':
    main()
