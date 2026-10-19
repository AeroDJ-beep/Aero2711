# -*- coding: utf-8 -*-
"""Probe 5: camera at the capsule centre (CAM_OFF x=-19) + tracker mesh hidden.
Checks the mask sees the target (no self-occlusion) and the camera is still kept out of walls."""
import os, sys, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
OUT = os.path.join(HERE, '..', 'results', 'figures', 'probe_mask')
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    R.setup_maze(env, 'fr10x10l35t24g_s0')
    print('cam_off', env.cam_off)
    R._teleport_target(env, 200, 40, 180); time.sleep(0.5)
    for hide in (0, 1):
        if hide:
            env.c.request(f'vset /object/{env.trk}/hide'); time.sleep(0.3)
        for (x, y, yaw) in [(-50, 40, 0), (0, 160, -23), (-10, 160, 0), (20, 160, 0)]:
            env.set_tracker(400, 160, yaw); time.sleep(0.1); env.set_tracker(x, y, yaw); time.sleep(0.15)
            env.frame(); box, npx, m, img = env.frame(lit=True)
            cl = env.c.request(f'vget /camera/{env.cam}/location')
            print(f'hide {hide} cmd ({x},{y},{yaw}) cam {cl} target px {npx} box {box} lit mean {img.mean():.0f}')
        cv2.imwrite(os.path.join(OUT, f'hide{hide}_lit.png'), img)
        cv2.imwrite(os.path.join(OUT, f'hide{hide}_mask.png'), env.api.get_image(env.cam, 'object_mask'))
finally:
    env.close()
