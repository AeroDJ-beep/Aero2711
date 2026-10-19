# -*- coding: utf-8 -*-
"""Analyze one logged maze episode: overall tracking quality and every blind
segment (cause, duration, belief error, re-acquisition).
usage: python uz_analyze_episode.py <log.npy> <maze_name>"""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_maze as Z, occl_core as C

L = np.load(sys.argv[1]); rects, info = Z.load_layout(sys.argv[2]); geo = Z.MazeGeo(rects, info)
n = len(L); vis = L[:, 5] > 0.5
tx, ty, yaw, gx, gy = L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4]
rng, bear, vcmd, vabs = L[:, 11], L[:, 12], L[:, 13], L[:, 14]
fade, vclip = L[:, 19] > 0.5, L[:, 20] > 0.5
dt = L[:, 10]
tgt_v = np.r_[0, np.hypot(np.diff(gx), np.diff(gy))] / np.maximum(np.r_[0.11, dt[1:]], 1e-3)
clr = np.array([geo.clearance(x, y) for x, y in zip(tx, ty)])

print(f'== overall: {n} steps, {n * dt.mean():.0f} s, loop {1 / dt.mean():.1f} Hz')
print(f'   visible {vis.mean() * 100:.0f}%  | in protocol window (range<=750, |bearing|<=45): {np.mean((rng <= 750) & (np.abs(bear) <= 45)) * 100:.0f}%')
print(f'   range: mean {rng.mean():.0f}  p10 {np.percentile(rng, 10):.0f}  p90 {np.percentile(rng, 90):.0f} cm (set point 250)')
print(f'   |bearing|: mean {np.abs(bear).mean():.1f}  p90 {np.percentile(np.abs(bear), 90):.1f} deg')
print(f'   tracker speed mean {vabs.mean():.0f} cm/s, commanded at the 99 cm/s cap {np.mean(vcmd >= 98.5) * 100:.0f}% of steps')
print(f'   wall clearance of the camera: min {clr.min():.0f} cm, p10 {np.percentile(clr, 10):.0f} cm')
print(f'   frames: fade/edge-rebuilt {fade.sum()}, head-row rejected {vclip.sum()}' +
      (f', DP planner in control {int((L[:, 21] > 0.5).sum())}' if L.shape[1] > 21 else ''))

# belief error on visible frames (pose k-1 = pose of the frame)
E = []
for k in range(1, n):
    px, py = C.back_project(L[k, 6], L[k, 7], tx[k - 1], ty[k - 1], yaw[k - 1])
    E.append(math.hypot(px - gx[k], py - gy[k]))
E = np.r_[np.nan, E]
print(f'   belief error on visible frames: p50 {np.nanmedian(E[vis]):.0f}  p90 {np.nanpercentile(E[vis], 90):.0f} cm')

blind = np.where(~vis)[0]
segs = np.split(blind, np.where(np.diff(blind) > 1)[0] + 1) if len(blind) else []
print(f'\n== blind segments: {len(segs)}')
for s in segs:
    a, b = s[0], s[-1]
    k0 = max(a - 1, 0)
    # cause: is the line of sight blocked by a wall, or is the target outside the field of view / too far?
    los = [Z.los_blocked(geo, (tx[k - 1] if k else tx[k], ty[k - 1] if k else ty[k]), (gx[k], gy[k])) for k in s]
    outfov = [abs(bear[k]) > 41.3 for k in s]
    cause = 'wall' if np.mean(los) > 0.5 else ('outside FOV' if np.mean(outfov) > 0.5 else 'other')
    # did the target turn during this segment?
    h_in = math.degrees(math.atan2(gy[a] - gy[k0 - 3 if k0 >= 3 else 0], gx[a] - gx[k0 - 3 if k0 >= 3 else 0]))
    h_out = math.degrees(math.atan2(gy[min(b + 3, n - 1)] - gy[b], gx[min(b + 3, n - 1)] - gx[b]))
    turn = (h_out - h_in + 180) % 360 - 180
    eb = [E[k] for k in s if not np.isnan(E[k])]
    print(f'  steps {a:3d}-{b:3d} ({len(s):2d} frames, {dt[s].sum():.1f} s): cause {cause:11s} target turn {turn:+4.0f} deg '
          f'| range at onset {rng[k0]:3.0f} | belief err {eb[0]:4.0f} -> {eb[-1]:4.0f} cm (max {max(eb):4.0f}; the last value is the error at re-acquisition) '
          f'| tracker v {vabs[s].mean():3.0f} | min wall clearance {clr[s].min():3.0f}')
