# -*- coding: utf-8 -*-
"""Probe 3: animated target (uz_maze.AnimatedDriver) along the gated maze route,
entrance -> exit, no tracker.  Reports completion, deviation from the route centre line,
minimum wall clearance, speed tracking; saves a top-view plot and a few chase frames.
outputs: results/figures/probe_move/anim_route.png, anim_chase_*.png"""
import os, sys, time, math
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R, uz_maze as Z
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = os.path.join(HERE, '..', 'results', 'figures', 'probe_move'); os.makedirs(OUT, exist_ok=True)
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    geo = R.setup_maze(env, 'fr10x10l35t24g_s0')
    mt = Z.MazeTarget(geo, np.random.default_rng(0), 'p2'); mt.plan_gates()
    x0, y0, h0 = mt.point_at(0.0)
    R._teleport_target(env, x0, y0, h0); time.sleep(0.8)
    drv = Z.AnimatedDriver(mt, env)
    T = []; t_prev = time.time(); k = 0; done = 0
    while k < 1500 and done < 10:
        t0 = time.time(); dt = float(np.clip(t0 - t_prev, 0.05, 0.3)) if k else 0.1; t_prev = t0
        pose = env.target_pose()
        drv.step(dt, pose)
        T.append((t0, pose[0], pose[1], pose[2], mt.s, mt.v, geo.clearance(pose[0], pose[1])))
        if k % 60 == 0 and k > 0:                                       # chase view from 2.5 m behind
            c, s_ = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
            env.set_tracker(pose[0] - 250 * c, pose[1] - 250 * s_, pose[2])
            cv2.imwrite(os.path.join(OUT, f'anim_chase_{k:04d}.png'), env.api.get_image(env.cam, 'lit'))
        done = done + 1 if mt.done() else 0
        k += 1
        el = time.time() - t0
        if el < 0.1:
            time.sleep(0.1 - el)
    env.c.request(f'vbp {env.tgt} set_move 0 0')
    T = np.array(T)
    v_real = np.hypot(np.diff(T[:, 1]), np.diff(T[:, 2])) / np.diff(T[:, 0])
    inside = T[:, 6] < 1e6
    print(f'steps {k}, finished {int(mt.done())}, route {mt.cum[-1]:.0f} cm, time {T[-1,0]-T[0,0]:.1f} s')
    print(f'deviation from centre line: mean {np.mean(drv.dev):.1f}  p95 {np.percentile(drv.dev, 95):.1f}  max {np.max(drv.dev):.1f} cm')
    print(f'min wall clearance of the actor origin: {T[:, 6].min():.0f} cm')
    print(f'speed: commanded mean {T[:-1, 5].mean():.0f}, real mean {v_real.mean():.0f} cm/s, |real - cmd| mean {np.mean(np.abs(v_real - T[:-1, 5])):.0f}')
    fig, ax = plt.subplots(figsize=(7, 7))
    for (a, b, c_, d) in geo.rects:
        ax.add_patch(plt.Rectangle((a, c_), b - a, d - c_, color='#888780'))
    ax.plot(mt.P[:, 0], mt.P[:, 1], '--', color='#7F77DD', lw=1, label='scripted route')
    ax.plot(T[:, 1], T[:, 2], color='#1D9E75', lw=1.5, label='animated target')
    ax.set_aspect('equal'); ax.legend(loc='upper right'); ax.set_title('animated target vs. route')
    fig.savefig(os.path.join(OUT, 'anim_route.png'), dpi=120, bbox_inches='tight')
finally:
    env.close()
