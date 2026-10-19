# -*- coding: utf-8 -*-
"""Build a saved maze layout in FlexibleRoom and capture a top view and a
first-person view down a corridor.  usage: python uz_maze_view.py <maze_name>"""
import os, sys, time, math
import cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R, uz_maze as Z

NAME = sys.argv[1] if len(sys.argv) > 1 else 'fr10x10l35t24_s0'
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_room'); os.makedirs(OUT, exist_ok=True)
env = U.UZEnv('FlexibleRoom')
try:
    geo = R.setup_maze(env, NAME)
    info = geo.info; W = info['nx'] * info['cell']; c = env.c
    c.request(f'vset /camera/0/location {Z.ORIGIN[0]} {Z.ORIGIN[1]} {env.z_floor + W * 0.62}'); c.request('vset /camera/0/rotation -90 0 0')
    time.sleep(0.8); cv2.imwrite(os.path.join(OUT, f'{NAME}_top.png'), env.api.get_image(0, 'lit'))
    import numpy as np
    mt = Z.MazeTarget(geo, np.random.default_rng(0), 'p2'); Z.MazeTarget.MIN_AFTER = 600.0; mt.plan()
    x0, y0, h0 = mt.point_at(0.0); R._teleport_target(env, x0, y0, h0)
    tx, ty = mt.tracker_start; env.set_tracker(tx, ty, math.degrees(math.atan2(y0 - ty, x0 - tx))); time.sleep(0.6)
    cv2.imwrite(os.path.join(OUT, f'{NAME}_fp.png'), env.api.get_image(env.cam, 'lit'))
    print('saved views for', NAME)
finally:
    env.close()
