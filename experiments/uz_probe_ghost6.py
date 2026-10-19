# -*- coding: utf-8 -*-
"""Probe 6: camera height with the centred camera + hidden tracker, free space vs near walls, over time."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    geo = R.setup_maze(env, 'fr10x10l35t24g_s0')
    print('env.z', env.z, 'cam_z', env.cam_z, 'cam_dz', env.cam_dz)
    env.c.request(f'vset /object/{env.trk}/hide'); time.sleep(0.3)
    free = [(x, y) for x in range(-1300, 500, 50) for y in range(-1000, 1000, 50) if geo.clearance(x, y) > 90][:3]
    near = [(0, 160), (7, 160)]
    for (x, y) in free + near:
        env.set_tracker(x, y, 0); zs = []
        for _ in range(5):
            time.sleep(0.1); zs.append(round(U._floats(env.c.request(f'vget /camera/{env.cam}/location'))[2], 1))
        print(f'({x},{y}) clearance {geo.clearance(x, y):.0f}  cam z over 0.5 s {zs}  actor {env.c.request(f"vget /object/{env.trk}/location")}')
finally:
    env.close()
