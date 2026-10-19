# -*- coding: utf-8 -*-
"""Probe 2 for set_move on the maze target (physics off): persistence of one command,
yaw-rate mapping, speed step response.  Pose is sampled every 0.1 s."""
import os, sys, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R

env = U.UZEnv('FlexibleRoom'); env.verbose = False
def track(cmd_fn, n, dt=0.1):
    P = []; t0 = time.time()
    for i in range(n):
        cmd_fn(i)
        time.sleep(dt); x, y, yaw = env.target_pose(); P.append((time.time() - t0, x, y, yaw))
    return np.array(P)
def speed(P):
    d = np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2])); return d / np.diff(P[:, 0])
try:
    geo = R.setup_maze(env, 'fr10x10l35t24g_s0'); inf = geo.info
    x0 = inf['x0'] - 400.0; y0 = inf['y0'] + inf['ny'] * inf['cell'] + 300.0
    mv = lambda a, v: env.c.request(f'vbp {env.tgt} set_move {a} {v}')
    def reset(yaw=0.0):
        mv(0, 0); time.sleep(0.6); R._teleport_target(env, x0, y0, yaw); time.sleep(0.6)
    # 1) one command, then silence
    reset(); P = track(lambda i: mv(0, 80) if i == 0 else None, 20)
    print('one command then wait: speed per 0.1 s =', np.round(speed(P)).astype(int).tolist())
    # 2) speed step response (command every step)
    for v in (50, 99, 150, 250):
        reset(); P = track(lambda i: mv(0, v), 25)
        s = speed(P); print(f'asked {v:3d}: first 1 s {np.round(s[:10]).astype(int).tolist()}  steady {s[12:].mean():.0f} cm/s')
    # 3) yaw-rate mapping
    for a in (30, 60, 90, 180):
        reset(); P = track(lambda i: mv(a, 60), 10)
        dy = (P[-1, 3] - P[0, 3] + 180) % 360 - 180
        print(f'yaw arg {a:3d}: yaw change {dy:+.0f} deg over {P[-1,0]-P[0,0]:.1f} s -> {dy / (P[-1,0]-P[0,0]):.0f} deg/s')
    # 4) stop response
    reset(); P = track(lambda i: mv(0, 99) if i < 15 else mv(0, 0), 25)
    print('99 then 0 at 1.5 s:', np.round(speed(P)).astype(int).tolist())
    mv(0, 0)
finally:
    env.close()
