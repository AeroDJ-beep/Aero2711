# -*- coding: utf-8 -*-
"""Offline route bank for a UnrealZoo scene: (start, goal) pairs whose shortest
path on the navmesh grid has an OCCLUDING corner 5-9 m after the start (line of
sight between 3 m before and 3 m after the corner is blocked) -> guaranteed
corner occlusion like the L map.  Saved to env_config/data/routes_<MAP>.json.
usage: python uz_routes.py <MAP> [n_routes]"""
import os, sys, json, math, heapq
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U

DATA = os.path.join(HERE, '..', 'env_config', 'data')


def grid_path(nav, a, b):
    """Dijkstra on the 8-connected free grid, returns path in cm (subsampled ~100 cm)."""
    h, w = nav.free.shape
    sa, sb = nav.cell_of(*a), nav.cell_of(*b)
    if not nav.free[sa] or not nav.free[sb]:
        return None
    dist = {sa: 0.0}; prev = {}; pq = [(0.0, sa)]
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while pq:
        d, c = heapq.heappop(pq)
        if c == sb:
            break
        if d > dist.get(c, 1e18):
            continue
        for dy, dx, wgt in nbrs:
            n = (c[0] + dy, c[1] + dx)
            if 0 <= n[0] < h and 0 <= n[1] < w and nav.free[n] and nav.dist[n] > 30.0:
                nd = d + wgt
                if nd < dist.get(n, 1e18):
                    dist[n] = nd; prev[n] = c; heapq.heappush(pq, (nd, n))
    if sb not in prev and sa != sb:
        return None
    cells = [sb]
    while cells[-1] != sa:
        cells.append(prev[cells[-1]])
    cells = cells[::-1]
    pts = np.array([[nav.x0 + c[1] * nav.cell, nav.y0 + c[0] * nav.cell] for c in cells], float)
    return pts[::4] if len(pts) > 8 else pts


def build_bank(map_name, n_routes=40, seed=0, dmin=350.0, dmax=1200.0):
    import uz_run as R
    pts = np.load(os.path.join(DATA, f'navpts_{map_name}.npy'))
    nav = U.NavGrid(pts)
    P = nav.pts
    clear = np.array([nav.clearance(x, y) for x, y in P])
    ok = np.where(clear > 70.0)[0]
    rng = np.random.default_rng(seed)
    bank = []; tries = 0
    while len(bank) < n_routes and tries < 4000:
        tries += 1
        i, j = rng.choice(ok, 2, replace=False)
        a, b = P[i], P[j]
        if np.hypot(*(a - b)) < 1200.0:
            continue
        path = grid_path(nav, a, b)
        if path is None or len(path) < 4:
            continue
        oc = R.occluding_corner(nav, path, dmin=dmin, dmax=dmax, look=250.0)
        if oc is None:
            continue
        # tracker must fit 250 cm behind the start along the initial heading with line of sight
        h0 = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        bx, by = a[0] - 250.0 * math.cos(h0), a[1] - 250.0 * math.sin(h0)
        if nav.clearance(bx, by) < 40.0 or R.los_blocked(nav, (bx, by), tuple(a)):
            continue
        bank.append(dict(start=[float(a[0]), float(a[1])], goal=[float(b[0]), float(b[1])], corner=[float(oc[0][0]), float(oc[0][1])],
                         corner_dist=float(oc[1]), heading=math.degrees(h0), path=path.round(1).tolist()))
    print(f'{map_name}: {len(bank)} routes with an occluding corner at {dmin:.0f}-{dmax:.0f} cm (tries {tries})')
    out = os.path.join(DATA, f'routes_{map_name}.json'); json.dump(bank, open(out, 'w')); print('saved', out)
    return bank, nav


def load_bank(map_name):
    p = os.path.join(DATA, f'routes_{map_name}.json')
    return json.load(open(p)) if os.path.isfile(p) else []


if __name__ == '__main__':
    MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    bank, nav = build_bank(MAP, n)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(nav.free, cmap='gray', origin='lower', extent=[nav.x0, nav.x0 + nav.free.shape[1] * nav.cell, nav.y0, nav.y0 + nav.free.shape[0] * nav.cell])
    for r in bank[:12]:
        p = np.array(r['path']); ax.plot(p[:, 0], p[:, 1], '-', lw=1, alpha=.7)
        ax.plot(*r['start'], 'go', ms=4); ax.plot(*r['corner'], 'r^', ms=6); ax.plot(*r['goal'], 'bs', ms=4)
    ax.set_aspect('equal'); ax.set_title(f'{MAP}: {len(bank)} corner routes (first 12 drawn; green start, red corner, blue goal)')
    fp = os.path.join(HERE, '..', 'results', 'figures', f'uz_routes_{MAP}.png'); fig.savefig(fp, dpi=110); print('saved', fp)
