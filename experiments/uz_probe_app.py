# -*- coding: utf-8 -*-
"""Spawn the original 'target' blueprint and cycle its appearance ids; tile the
results next to an old L-map frame for comparison."""
import os, sys, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U

TGT = '/Game/HumanCharacter/ThirdPersonBP/Blueprints/target.target'
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_room')
env = U.UZEnv('FlexibleRoom', sleep_time=30); c = env.c
try:
    base = (-450.0, -1500.0); z = 178.0
    c.request(f'vset /objects/spawn_from_path {TGT} t0')
    c.request([f'vset /object/t0/location {base[0]} {base[1]} {z}', 'vset /object/t0/rotation 0 -90 0', 'vbp t0 set_phy 0',
               'vset /object/t0/scale 0.85 0.85 0.85'], -1)
    c.request(f'vset /camera/0/location {base[0]} {base[1] - 250} {z + 20}'); c.request('vset /camera/0/rotation -5 90 0')
    time.sleep(1.0)
    tiles = []
    APPS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(16))
    for app in APPS:
        r = c.request(f'vbp t0 set_app {app}')
        time.sleep(0.5)
        im = env.api.get_image(0, 'lit')[60:420, 220:420].copy()
        cv2.putText(im, str(app), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(im)
        print('app', app, '->', str(r)[:40].replace('\n', ' '))
    old = os.path.join(OUT, '..', 'frames2005', 'frame_214.png')
    ref = cv2.imread(old)
    if ref is not None:
        ref = cv2.resize(ref[:, :640][60:420, 220:420], (200, 360)); cv2.putText(ref, 'OLD', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles = [ref] + tiles
    while len(tiles) % 9:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)]
    cv2.imwrite(os.path.join(OUT, 'char_apps.png'), np.vstack(rows)); print('saved char_apps.png')
finally:
    env.close()
