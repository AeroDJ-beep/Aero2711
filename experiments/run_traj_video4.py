# -*- coding: utf-8 -*-
"""Side-by-side: decoupled combo v2 vs +mapclamp, seed 2005 sprint, ca_r01.
Uses the real run_episode via its steplog hook - no duplicated logic.
Extra visual: purple phantom trail (last 60 steps of the KF belief's
back-projected world point) - mapclamp keeps it out of the walls."""
import os, sys, math
import numpy as np
SCRATCH_OLD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH_OLD)
import run_metrics_test as M
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp

SEED = 2005
OUT = r'D:\Aero2711\results\videos\traj_mapclamp_compare_seed2005.mp4'


def record(ucv, z_trk, use_clamp):
    M.VCAP, M.EDGEMEAS, M.EDGEAIM = True, True, True
    M.BLINDSPEED = True
    M.MAPCLAMP = use_clamp
    log = []
    res = M.run_episode(ucv, z_trk, 'ca_r01', SEED, 'p2a', steplog=log)
    print(('clamp ON ' if use_clamp else 'clamp OFF'), res)
    return log


def phantom_xy(row):
    tx, ty, yaw, _, _, _, kfcx, kfh, have = row
    if not have:
        return None
    ang = math.degrees(math.atan2(kfcx - M.CX0, M.FX))
    bear = math.radians(yaw + ang)
    rho = min(M.FX * M.H_PERSON / max(kfh, 1.0), 900.0)
    return tx + rho * math.cos(bear), ty + rho * math.sin(bear)


def draw_panel(ax, log, k, label):
    ax.clear()
    for (x0, x1, y0, y1) in M.WALLS:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  facecolor='#8f8f8f', edgecolor='k', zorder=3))
    ax.add_patch(mp.Rectangle((-585, -700), 150, 1095, facecolor='#FBF3DC', zorder=1))
    ax.add_patch(mp.Rectangle((-585, 245), 700, 150, facecolor='#FBF3DC', zorder=1))
    cam = np.array([(r[0], r[1]) for r in log[:k + 1]])
    tgt = np.array([(r[3], r[4]) for r in log[:k + 1]])
    ax.plot(tgt[:, 0], tgt[:, 1], '-', color='#E07B00', lw=1.5, alpha=0.85, zorder=5)
    ax.plot(cam[:, 0], cam[:, 1], '-', color='#1668C7', lw=1.5, alpha=0.85, zorder=5)
    det = log[k][5]
    col = '#2E9E44' if det else '#C81E1E'
    ax.plot([cam[-1, 0], tgt[-1, 0]], [cam[-1, 1], tgt[-1, 1]], '--', color=col, lw=1.2, zorder=6)
    # phantom trail (last 60 steps) + current star
    ph = [phantom_xy(log[j]) for j in range(max(0, k - 60), k + 1)]
    ph = [p for p in ph if p is not None]
    if ph:
        pa = np.array(ph)
        ax.plot(pa[:, 0], pa[:, 1], '-', color='#8B2FC9', lw=1.6, alpha=0.6, zorder=7)
        ax.plot(pa[-1, 0], pa[-1, 1], '*', ms=14, color='#8B2FC9', mec='k', zorder=8)
    ax.plot(tgt[-1, 0], tgt[-1, 1], 'o', ms=10, color='#E07B00', mec='k', zorder=9)
    ax.plot(cam[-1, 0], cam[-1, 1], marker=(3, 0, log[k][2] - 90), ms=13,
            color='#1668C7', mec='k', zorder=9)
    status = 'VISIBLE' if det else 'OCCLUDED'
    ax.set_title(f'{label}\n{status}', fontsize=11.5, fontweight='bold',
                 color='#C81E1E' if not det else '#222222')
    ax.set_xlim(-780, 200)
    ax.set_ylim(-760, 480)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)


if __name__ == '__main__':
    runner, ucv, z_trk = M.boot()
    try:
        L_off = record(ucv, z_trk, use_clamp=False)
        L_on = record(ucv, z_trk, use_clamp=True)
    finally:
        try:
            runner.close()
        except Exception:
            pass
    n = len(L_off)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=100)
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1600, 720))
    for k in range(n):
        draw_panel(axL, L_off, k, 'combo v2 (edgemeas+edgeaim)')
        draw_panel(axR, L_on, min(k, len(L_on) - 1), '+ mapclamp v3 (project into corridor free space)')
        fig.suptitle(f'seed {SEED} sprint - purple = KF belief back-projected to world '
                     f'(trail = last 3 s) - step {k + 1}/{n}  t={(k + 1) * M.DT:4.1f}s',
                     fontsize=12.5, fontweight='bold')
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        vw.write(cv2.cvtColor(cv2.resize(buf, (1600, 720)), cv2.COLOR_RGB2BGR))
        if (k + 1) % 125 == 0:
            print(f'rendered {k + 1}/{n}')
            sys.stdout.flush()
    vw.release()
    print('video saved:', OUT)
