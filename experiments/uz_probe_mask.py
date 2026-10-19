# -*- coding: utf-8 -*-
"""Probe: replay logged tracker/target poses of an episode and inspect the raw object mask
(why does the target vanish from the mask while clearly in view?).
usage: python uz_probe_mask.py <episode npy> k0 k1"""
import os, sys, time, math
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R

L = np.load(sys.argv[1]); k0, k1 = int(sys.argv[2]), int(sys.argv[3])
OUT = os.path.join(HERE, '..', 'results', 'figures', 'probe_mask'); os.makedirs(OUT, exist_ok=True)
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    geo = R.setup_maze(env, 'fr10x10l35t24g_s0')
    print('tgt colour', env.tgt_rgb)
    for k in range(k0, k1 + 1):
        gx, gy = L[k, 3:5]; h = math.degrees(math.atan2(L[k, 4] - L[k - 2, 4], L[k, 3] - L[k - 2, 3]))
        R._teleport_target(env, gx, gy, h)
        tx, ty, yaw = L[k - 1, 0:3]; env.set_tracker(tx, ty, yaw); time.sleep(0.15)
        for rep in range(2):
            box, npx, m, img = env.frame(lit=True)
            msk = env.api.get_image(env.cam, 'object_mask')
            cols, cnt = np.unique(msk.reshape(-1, 3), axis=0, return_counts=True)
            top = sorted(zip(cnt, map(tuple, cols)), reverse=True)[:6]
            print(f'k{k} rep{rep} npx {npx} box {box} logged {int(L[k,5])} | top colours (BGR) {top}')
        if k in (k0, (k0 + k1) // 2, k1):
            cv2.imwrite(os.path.join(OUT, f'k{k}_mask.png'), msk); cv2.imwrite(os.path.join(OUT, f'k{k}_lit.png'), img)
finally:
    env.close()
