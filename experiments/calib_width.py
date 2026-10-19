# -*- coding: utf-8 -*-
"""Calibrate the target's bbox width as a function of its speed and heading
relative to the line of sight, from un-clipped (healthy) frames.
Output: results/data/width_calib.csv + figure."""
import os, sys, math, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_metrics_test as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
M.DYN = M.EDGEMEAS = M.EDGEAIM = M.BLINDSPEED = M.MAPCLAMP = True
SEEDS = [(s, p) for p in ('p2', 'p2a') for s in range(2000, 2006)]
OUT_CSV = os.path.join(HERE, '..', 'env_config', 'data', 'width_calib.csv')
rows = []
runner, ucv, z_trk = M.boot()
try:
    for seed, prof in SEEDS:
        M.DEBUG_EDGE = []; log = []
        M.run_episode(ucv, z_trk, 'ca_r01', seed, prof, steplog=log)
        P = np.array([(r[3], r[4]) for r in log]); C = np.array([(r[0], r[1]) for r in log])
        V = np.gradient(P, axis=0) / M.DT
        det = np.array([r[5] for r in log], bool)
        since_blind = np.full(len(log), 9999)
        last = -9999
        for i in range(len(log)):
            if not det[i]: last = i
            since_blind[i] = i - last
        for (k, bl, br, u, wg, cxc, npx, rh, g) in M.DEBUG_EDGE:
            if rh or g or k < 5 or k >= len(log) - 1: continue
            if since_blind[k] < 30: continue          # drop un-flagged half-clipped reappearance frames
            v = V[k]; sp = float(np.hypot(*v))
            los = P[k] - C[k]; rng = float(np.hypot(*los))
            phi = math.degrees(math.atan2(v[1], v[0]) - math.atan2(los[1], los[0]))
            phi = (phi + 180) % 360 - 180
            # box height from the KF state is smoothed; use geometric h = fx*168/range
            h_geo = M.FX * M.H_PERSON / rng
            rows.append((seed, prof, k, br - bl, h_geo, (br - bl) / h_geo, npx, sp, abs(math.sin(math.radians(phi))), rng))
            # also drop frames whose box touches the occluder edge (pre-clip band the ratio trigger misses)
            ue = M.edge_u(C[k][0], C[k][1], log[k][2])
            if ue is not None and bl - 6 <= ue <= br + 6: rows.pop()
        print(seed, prof, 'healthy rows', len(rows)); sys.stdout.flush()
finally:
    try: runner.close()
    except Exception: pass
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['seed', 'prof', 'k', 'w_px', 'h_geo', 'aspect', 'npx', 'speed', 'sin_phi', 'range'])
    w.writerows(rows)
R = np.array([r[2:] for r in rows], float)
k, w, h, a, npx, sp, sphi, rng = R.T
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
sc = ax[0].scatter(sp, a, c=sphi, s=6, cmap='viridis'); ax[0].set_xlabel('target speed (cm/s)'); ax[0].set_ylabel('aspect w/h'); plt.colorbar(sc, ax=ax[0], label='|sin(heading vs LOS)|')
sc = ax[1].scatter(sphi, a, c=sp, s=6, cmap='plasma'); ax[1].set_xlabel('|sin phi| (0 = along LOS, 1 = across)'); ax[1].set_ylabel('aspect w/h'); plt.colorbar(sc, ax=ax[1], label='speed')
# fit a = c0 + c1*speed + c2*sin_phi + c3*speed*sin_phi
X = np.column_stack([np.ones_like(sp), sp, sphi, sp * sphi]); coef, *_ = np.linalg.lstsq(X, a, rcond=None)
pred = X @ coef; rms = np.sqrt(np.mean((a - pred) ** 2))
ax[2].scatter(pred, a, s=6, alpha=.5); lo, hi = a.min(), a.max(); ax[2].plot([lo, hi], [lo, hi], 'k--'); ax[2].set_xlabel('fitted aspect'); ax[2].set_ylabel('measured aspect')
ax[2].set_title(f'a = {coef[0]:.3f} + {coef[1]:.5f}*v + {coef[2]:.3f}*s + {coef[3]:.5f}*v*s   rms {rms:.3f}', fontsize=9)
for x in ax: x.grid(alpha=.3)
fig.suptitle(f'bbox aspect vs speed & heading, {len(rows)} healthy frames, 12 episodes'); fig.tight_layout()
fig.savefig(os.path.join(HERE, '..', 'results', 'figures', 'width_calib.png'), dpi=120)
print('coef', coef, 'rms', rms)
import json; json.dump({'c0': float(coef[0]), 'c1': float(coef[1]), 'c2': float(coef[2]), 'c3': float(coef[3]), 'rms': float(rms), 'n': int(len(rows))},
                       open(os.path.join(HERE, '..', 'env_config', 'data', 'width_model.json'), 'w'), indent=1)
print('aspect by speed bin:')
for lo_, hi_ in [(0, 30), (30, 50), (50, 70), (70, 90), (90, 120)]:
    m = (sp >= lo_) & (sp < hi_)
    if m.any(): print(f'  v {lo_:3d}-{hi_:3d}: n={m.sum():4d}  aspect mean {a[m].mean():.3f} sd {a[m].std():.3f}   |sinphi| mean {sphi[m].mean():.2f}')
print('aspect by |sin phi| bin (speed>60):')
for lo_, hi_ in [(0, .3), (.3, .6), (.6, .8), (.8, 1.01)]:
    m = (sphi >= lo_) & (sphi < hi_) & (sp > 60)
    if m.any(): print(f'  s {lo_:.1f}-{hi_:.1f}: n={m.sum():4d}  aspect mean {a[m].mean():.3f} sd {a[m].std():.3f}')
