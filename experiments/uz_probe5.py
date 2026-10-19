# -*- coding: utf-8 -*-
"""Probe 5: reachable-point cache semantics (spacing / count args), nav status,
and a free-space grid built from the navmesh points."""
import os, sys, json, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
env = U.UZEnv(MAP, sleep_time=25)
def pts_of(r):
    try:
        j = json.loads(r); p = j.get('points', j) if isinstance(j, dict) else j
        return np.array([[q[0], q[1], q[2]] for q in p], float), (list(j.keys()) if isinstance(j, dict) else 'list')
    except Exception:
        return np.zeros((0, 3)), str(r)[:120]
try:
    env.spawn()
    c = env.c
    print('status ->', c.request('vget /reachablepoints/status')[:300].replace('\n', ' '))
    for cmd in ['vget /reachablepoints', 'vget /reachablepoints 50', 'vget /reachablepoints 25', 'vget /reachablepoints 50 100000',
                'vget /reachablepoints/count', 'vget /reachablepoints/count 25']:
        t0 = time.time(); r = c.request(cmd, timeout=120); P, keys = pts_of(r)
        if len(P):
            d = np.sort(np.hypot(P[1:, 0] - P[:-1, 0], P[1:, 1] - P[:-1, 1]))
            print(f'{cmd:40s} -> {len(P)} pts in {time.time() - t0:.1f}s, keys {keys}, x[{P[:,0].min():.0f},{P[:,0].max():.0f}] y[{P[:,1].min():.0f},{P[:,1].max():.0f}] z[{P[:,2].min():.0f},{P[:,2].max():.0f}], median consecutive gap {np.median(d):.0f}')
        else:
            print(f'{cmd:40s} -> {str(r)[:160]}')
    print('nav status ->', c.request('vget /agent/tgt/nav/status')[:300].replace('\n', ' '))
    gx, gy, _ = env.target_pose(); goal = (gx + 20, gy + 1500)
    print('nav goto ->', c.request(f'vset /agent/tgt/nav/goto {goal[0]} {goal[1]} {env.z_floor}')[:200].replace('\n', ' '))
    for i in range(4):
        time.sleep(1.0); print('  ', env.target_pose(), c.request('vget /agent/tgt/nav/status')[:160].replace('\n', ' '))
    # free grid from the densest point set
    r = c.request('vget /reachablepoints 25', timeout=120); P, _ = pts_of(r)
    if len(P) == 0:
        r = c.request('vget /reachablepoints', timeout=120); P, _ = pts_of(r)
    np.save(os.path.join(HERE, '..', 'env_config', 'data', f'navpts_{MAP}.npy'), P)
    fig, ax = plt.subplots(figsize=(8, 12)); ax.plot(P[:, 0], P[:, 1], '.', ms=1.2, color='#1668C7')
    S = np.array(env.starts); ax.plot(S[:, 0], S[:, 1], 'r^', ms=6); ax.set_aspect('equal'); ax.set_title(f'{MAP}: navmesh reachable points ({len(P)})')
    p = os.path.join(HERE, '..', 'results', 'figures', f'uz_navpts_{MAP}.png'); fig.savefig(p, dpi=110); print('saved', p)
finally:
    env.close()
