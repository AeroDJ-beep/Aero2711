# -*- coding: utf-8 -*-
"""Side-by-side: wall margin 15 cm (render-limit) vs 30 cm (body + localisation
slack), seed 2005 p2a, ca_r01, full architecture. Draws the margin band
around every wall (dashed) and a clearance circle around the UAV so that
'scraping' can be judged by eye."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_metrics_test as M
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp

SEED = 2005
MARGINS = (15.0, 30.0)
OUT = os.path.join(HERE, '..', 'results', 'videos', f'traj_margin_compare_seed{SEED}.mp4')


def record(ucv, z_trk, margin):
    M.VCAP = M.EDGEMEAS = M.EDGEAIM = M.BLINDSPEED = M.MAPCLAMP = True
    M.MARGIN_CAM = margin
    log = []
    res = M.run_episode(ucv, z_trk, 'ca_r01', SEED, 'p2a', steplog=log)
    d = [M.dist_to_walls(r[0], r[1]) for r in log]
    print(f'margin {margin:.0f}: {res}  min clearance {min(d):.1f} cm at step {int(np.argmin(d))}')
    sys.stdout.flush()
    return log, d


def phantom_xy(row):
    tx, ty, yaw, _, _, _, kfcx, kfh, have = row
    if not have:
        return None
    ang = math.degrees(math.atan2(kfcx - M.CX0, M.FX))
    bear = math.radians(yaw + ang)
    rho = min(M.FX * M.H_PERSON / max(kfh, 1.0), 900.0)
    return tx + rho * math.cos(bear), ty + rho * math.sin(bear)


def draw_panel(ax, log, dist, k, margin, label):
    ax.clear()
    ax.add_patch(mp.Rectangle((-585, -700), 150, 1095, facecolor='#FBF3DC', zorder=1))
    ax.add_patch(mp.Rectangle((-585, 245), 700, 150, facecolor='#FBF3DC', zorder=1))
    for (x0, x1, y0, y1) in M.WALLS:
        ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                       boxstyle=f'round,pad={margin}',
                                       facecolor='#F2C6C6', edgecolor='#C81E1E',
                                       ls='--', lw=0.8, alpha=0.55, zorder=2))
    for (x0, x1, y0, y1) in M.WALLS:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor='#8f8f8f', edgecolor='k', zorder=3))
    cam = np.array([(r[0], r[1]) for r in log[:k + 1]])
    tgt = np.array([(r[3], r[4]) for r in log[:k + 1]])
    ax.plot(tgt[:, 0], tgt[:, 1], '-', color='#E07B00', lw=1.5, alpha=0.85, zorder=5)
    ax.plot(cam[:, 0], cam[:, 1], '-', color='#1668C7', lw=1.5, alpha=0.85, zorder=5)
    det = log[k][5]
    col = '#2E9E44' if det else '#C81E1E'
    ax.plot([cam[-1, 0], tgt[-1, 0]], [cam[-1, 1], tgt[-1, 1]], '--', color=col, lw=1.2, zorder=6)
    ph = [phantom_xy(log[j]) for j in range(max(0, k - 60), k + 1)]
    ph = [p for p in ph if p is not None]
    if ph:
        pa = np.array(ph)
        ax.plot(pa[:, 0], pa[:, 1], '-', color='#8B2FC9', lw=1.6, alpha=0.6, zorder=7)
        ax.plot(pa[-1, 0], pa[-1, 1], '*', ms=14, color='#8B2FC9', mec='k', zorder=8)
    ax.plot(tgt[-1, 0], tgt[-1, 1], 'o', ms=10, color='#E07B00', mec='k', zorder=9)
    # clearance circle (radius = margin) + UAV as a small dot, not a big marker
    ax.add_patch(mp.Circle((cam[-1, 0], cam[-1, 1]), margin, facecolor='none',
                           edgecolor='#1668C7', lw=1.4, zorder=9))
    ax.plot(cam[-1, 0], cam[-1, 1], '.', ms=6, color='#1668C7', zorder=10)
    hx = math.cos(math.radians(log[k][2])); hy = math.sin(math.radians(log[k][2]))
    ax.plot([cam[-1, 0], cam[-1, 0] + 25 * hx], [cam[-1, 1], cam[-1, 1] + 25 * hy],
            '-', color='#1668C7', lw=1.8, zorder=10)
    status = 'VISIBLE' if det else 'OCCLUDED'
    dmin = min(dist[:k + 1])
    ax.set_title(f'{label}\n{status}   clearance now {dist[k]:5.1f} cm   min so far {dmin:5.1f} cm',
                 fontsize=11, fontweight='bold',
                 color='#C81E1E' if not det else '#222222')
    ax.set_xlim(-780, 200)
    ax.set_ylim(-760, 480)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)


if __name__ == '__main__':
    runner, ucv, z_trk = M.boot()
    try:
        logs = [record(ucv, z_trk, m) for m in MARGINS]
    finally:
        try:
            runner.close()
        except Exception:
            pass
    n = max(len(L) for L, _ in logs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=100)
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1600, 720))
    for k in range(n):
        for ax, (L, d), m in zip(axes, logs, MARGINS):
            draw_panel(ax, L, d, min(k, len(L) - 1), m,
                       f'margin {m:.0f} cm' + (' (render limit)' if m == 15 else ' (body + localisation)'))
        fig.suptitle(f'seed {SEED} p2a - red band = safety margin around walls, blue circle = margin around UAV '
                     f'- step {k + 1}/{n}  t={(k + 1) * M.DT:4.1f}s',
                     fontsize=12, fontweight='bold')
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        vw.write(cv2.cvtColor(cv2.resize(buf, (1600, 720)), cv2.COLOR_RGB2BGR))
        if (k + 1) % 125 == 0:
            print(f'rendered {k + 1}/{n}'); sys.stdout.flush()
    vw.release()
    print('video saved:', os.path.abspath(OUT))
