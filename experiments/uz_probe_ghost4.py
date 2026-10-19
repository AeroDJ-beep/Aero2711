# -*- coding: utf-8 -*-
"""Probe 4: camera mounted at the tracker actor centre (no 59 cm forward offset).
Checks (a) self-occlusion in the object mask, (b) camera position near wall mz15."""
import os, sys, time, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    R.setup_maze(env, 'fr10x10l35t24g_s0')
    trk_rgb = [int(v) for v in re.findall(r'[RGB]=(\d+)', env.c.request(f'vget /object/{env.trk}/color'))]
    selfc = np.array(trk_rgb[::-1], np.int16)
    print('old cam_off', env.cam_off, 'trk colour', trk_rgb)
    env.c.request(f'vbp {env.trk} set_cam 0 0 {U.CAM_OFF[2]} 0 {U.PITCH} 0'); time.sleep(0.5)
    env.cam_off = (0.0, 0.0)
    R._teleport_target(env, 200, 40, 180); time.sleep(0.5)
    for (x, y, yaw) in [(400, 160, 180), (0, 160, -23), (-10, 160, 0), (5, 160, 0), (20, 160, 0), (-20, 160, 180), (30, 160, -23)]:
        env.set_tracker(400, 160, yaw); time.sleep(0.1); env.set_tracker(x, y, yaw); time.sleep(0.15)
        cl = env.c.request(f'vget /camera/{env.cam}/location')
        r = []
        for _ in range(3):
            msk = env.api.get_image(env.cam, 'object_mask')
            r.append((int((np.abs(msk.astype(np.int16) - np.array(env.tgt_rgb[::-1], np.int16)).sum(axis=2) < 9).sum()),
                      int((np.abs(msk.astype(np.int16) - selfc).sum(axis=2) < 9).sum())))
        print(f'cmd ({x},{y},{yaw}) cam {cl}  (target px, self px) {r}')
finally:
    env.close()
