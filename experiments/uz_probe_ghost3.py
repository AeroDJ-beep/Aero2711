# -*- coding: utf-8 -*-
"""Probe 3: camera offset vs actor body near wall mz15 (x -62..-38): commanded camera x,
actual actor / camera location, for yaw -23 and 0 and 90."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    R.setup_maze(env, 'fr10x10l35t24g_s0')
    print('cam_off', env.cam_off, 'trk', env.trk, 'z', env.z)
    for yaw in (-23, 0, 90, 180):
        for x in (-10, 0, 5, 10, 20, 30, 40, 60):
            env.set_tracker(400, 160, yaw); time.sleep(0.1)            # reset in open space first
            env.set_tracker(x, 160, yaw); time.sleep(0.12)
            a = env.c.request(f'vget /object/{env.trk}/location'); c = env.c.request(f'vget /camera/{env.cam}/location')
            print(f'yaw {yaw:4d} cmd cam x {x:4d} -> actor {a} | cam {c}')
finally:
    env.close()
