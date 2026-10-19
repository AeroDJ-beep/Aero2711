# -*- coding: utf-8 -*-
"""Probe: can the maze target be driven by its movement component (vbp set_move
<yaw rate deg/s> <speed cm/s>) so that the walk/run animation plays, instead of being
teleported along the route?  Saves side-view frames for teleport vs set_move and
prints the measured speed / turn rate.
outputs: results/figures/probe_move/*.png"""
import os, sys, time, math
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R, uz_maze as Z

OUT = os.path.join(HERE, '..', 'results', 'figures', 'probe_move'); os.makedirs(OUT, exist_ok=True)
env = U.UZEnv('FlexibleRoom')
try:
    geo = R.setup_maze(env, 'fr10x10l35t24g_s0')
    inf = geo.info
    # open floor outside the top-left entrance: room to walk 4 m in +x
    x0 = inf['x0'] - 200.0; y0 = inf['y0'] + inf['ny'] * inf['cell'] + 250.0
    def view(tag, n=4, dt=0.25, cam=(0.0, -330.0)):
        for i in range(n):
            gx, gy, gyaw = env.target_pose()
            env.set_tracker(gx + cam[0], gy + cam[1], 90.0)              # camera to the side, looking at the target
            cv2.imwrite(os.path.join(OUT, f'{tag}_{i}.png'), env.api.get_image(env.cam, 'lit'))
            time.sleep(dt)
    # A) teleport (current): 4 frames while stepping 8 cm / 0.1 s
    R._teleport_target(env, x0, y0, 0.0); time.sleep(0.5)
    for i in range(4):
        for _ in range(3):
            x0 += 8.0; R._teleport_target(env, x0, y0, 0.0); time.sleep(0.1)
        gx, gy, _ = env.target_pose(); env.set_tracker(gx, gy - 330.0, 90.0)
        cv2.imwrite(os.path.join(OUT, f'teleport_{i}.png'), env.api.get_image(env.cam, 'lit'))
    # B) set_move with physics as spawned (set_phy 0)
    for phy in (0, 1):
        R._teleport_target(env, x0 - 100.0, y0, 0.0); time.sleep(0.5)
        env.c.request(f'vbp {env.tgt} set_phy {phy}'); time.sleep(0.3)
        p0 = env.target_pose(); t0 = time.time()
        for _ in range(20):
            env.c.request(f'vbp {env.tgt} set_move 0 80'); time.sleep(0.1)
        p1 = env.target_pose(); el = time.time() - t0
        print(f'set_phy {phy}: moved {math.hypot(p1[0]-p0[0], p1[1]-p0[1]):.0f} cm in {el:.1f} s '
              f'-> {math.hypot(p1[0]-p0[0], p1[1]-p0[1]) / el:.0f} cm/s, yaw {p0[2]:.0f} -> {p1[2]:.0f}', flush=True)
        view(f'setmove_phy{phy}', n=4, dt=0.0)
        env.c.request(f'vbp {env.tgt} set_move 0 0'); time.sleep(0.5)
    # C) turning: yaw rate 90 deg/s for 1 s while walking
    p0 = env.target_pose()
    for _ in range(10):
        env.c.request(f'vbp {env.tgt} set_move 90 60'); time.sleep(0.1)
    p1 = env.target_pose()
    print(f'turn: yaw {p0[2]:.0f} -> {p1[2]:.0f} in ~1 s (asked 90 deg/s)', flush=True)
    env.c.request(f'vbp {env.tgt} set_move 0 0')
    # D) speed response: 40 -> 99 cm/s step
    for v in (40, 99):
        p0 = env.target_pose(); t0 = time.time()
        for _ in range(15):
            env.c.request(f'vbp {env.tgt} set_move 0 {v}'); time.sleep(0.1)
        p1 = env.target_pose(); el = time.time() - t0
        print(f'asked {v} cm/s -> measured {math.hypot(p1[0]-p0[0], p1[1]-p0[1]) / el:.0f} cm/s', flush=True)
    env.c.request(f'vbp {env.tgt} set_move 0 0')
finally:
    env.close()
print('saved frames to', OUT)
