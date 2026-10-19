# -*- coding: utf-8 -*-
"""Is the original L-map target (targetbp2) available in the UnrealZoo FlexibleRoom?
Lists target-like actors / assets, spawns candidates in front of a camera and saves
a comparison image to results/logs/uz_room/char_*.png"""
import os, sys, re, json, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U

OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_room')
env = U.UZEnv('FlexibleRoom', sleep_time=30); c = env.c
try:
    objs = env.api.get_objects()
    tg = [o for o in objs if re.search(r'target|person|human|girl|woman|char', o, re.I)]
    print('target-like actors in the map:', tg)
    for o in tg:
        print('  ', o, 'loc', c.request(f'vget /object/{o}/location'), '| meta', str(c.request(f'vget /object/{o}/class_metadata'))[:200].replace('\n', ' '))
    lst = open(os.path.join(OUT, 'FlexibleRoom_assets.txt'), encoding='utf-8').read().split('\n') if os.path.isfile(os.path.join(OUT, 'FlexibleRoom_assets.txt')) else []
    cand = [a for a in lst if re.search(r'target|Target', a)]
    print('target-like assets:', cand[:30])
    # put the camera 0 in an empty spot and bring each target-like actor in front of it
    base = (-450.0, -1500.0); z = 178.0
    c.request(f'vset /camera/0/location {base[0]} {base[1] - 300} {z}'); c.request('vset /camera/0/rotation 0 90 0')
    shots = []
    for i, o in enumerate(tg[:4]):
        c.request(f'vset /object/{o}/location {base[0] - 150 + i * 100} {base[1]} {z}'); c.request(f'vset /object/{o}/rotation 0 -90 0')
    time.sleep(1.0)
    cv2.imwrite(os.path.join(OUT, 'char_existing.png'), env.api.get_image(0, 'lit'))
    # spawn candidate assets
    for i, a in enumerate(cand[:6]):
        n = f'cand{i}'
        r = c.request(f'vset /objects/spawn_from_path {a} {n}')
        ok = n in env.api.get_objects()
        print('spawn', a, '->', str(r)[:60], 'ok' if ok else 'FAILED')
        if ok:
            c.request([f'vset /object/{n}/location {base[0] - 250 + i * 100} {base[1] + 200} {z}', f'vset /object/{n}/rotation 0 -90 0'], -1)
    time.sleep(1.0)
    cv2.imwrite(os.path.join(OUT, 'char_spawned.png'), env.api.get_image(0, 'lit'))
finally:
    env.close()
