# -*- coding: utf-8 -*-
"""Probe 2: a nearby wall swallowing the object mask.  Target fixed in view; tracker placed at
increasing distance from wall mz15 (facing away); count target / wall pixels in the mask."""
import os, sys, time, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
env = U.UZEnv('FlexibleRoom'); env.verbose = False
WALL = np.array([191, 191, 63], np.int16)                    # BGR of mz15
def wallpx(msk): return int((np.abs(msk.astype(np.int16) - WALL).sum(axis=2) < 9).sum())
try:
    R.setup_maze(env, 'fr10x10l35t24g_s0')
    R._teleport_target(env, 200, 40, 180); time.sleep(0.5)
    for x in (0, 10, 20, 40, 60, 80):
        env.set_tracker(x, 160, -23); time.sleep(0.15)
        print(f'tracker x={x:3d}  cam loc {env.c.request(f"vget /camera/{env.cam}/location")}  rot {env.c.request(f"vget /camera/{env.cam}/rotation")}')
        r = []
        for _ in range(6):
            box, npx, m, _ = env.frame(); r.append((npx, wallpx(env.api.get_image(env.cam, 'object_mask'))))
        print('   (target px, wall px):', r)
    print('--- facing the other way at x=0 (yaw 157)')
    env.set_tracker(0, 160, 157); time.sleep(0.15)
    print('   wall px', [wallpx(env.api.get_image(env.cam, 'object_mask')) for _ in range(4)])
    print('--- tracker x=0, wall mz15 moved away')
    env.c.request('vset /object/mz15/location -50 250 -2000'); time.sleep(0.2)
    env.set_tracker(0, 160, -23); time.sleep(0.15)
    print('   target px', [env.frame()[1] for _ in range(6)])
finally:
    env.close()
