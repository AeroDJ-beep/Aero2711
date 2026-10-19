# -*- coding: utf-8 -*-
"""Probe 7: does any scene object share the target's annotation colour?  Can we
assign a unique colour?  Also check mask freshness (two frames 0.3 s apart)."""
import os, sys, time, re, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U
env = U.UZEnv('Hotel_Corridor', sleep_time=25)
try:
    env.spawn(); nav = env.build_navgrid(); c = env.c
    print('tgt colour', env.tgt_rgb)
    cols = {}
    for o in env.api.get_objects():
        r = c.request(f'vget /object/{o}/color')
        v = tuple(int(x) for x in re.findall(r'[RGB]=(\d+)', str(r))[:3])
        if len(v) == 3:
            cols.setdefault(v, []).append(o)
    same = [(k, v) for k, v in cols.items() if all(abs(a - b) < 4 for a, b in zip(k, env.tgt_rgb))]
    print('objects sharing (within 3) the target colour:', same)
    print('distinct colours', len(cols), 'of', sum(len(v) for v in cols.values()), 'objects')
    # try to assign a unique colour
    used = set(cols.keys())
    for cand in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (250, 10, 120)]:
        if not any(all(abs(a - b) < 12 for a, b in zip(cand, u)) for u in used):
            break
    print('candidate unique colour', cand, 'set ->', c.request(f'vset /object/tgt/color {cand[0]} {cand[1]} {cand[2]}'),
          'readback', c.request('vget /object/tgt/color'))
    time.sleep(0.5)
    env.tgt_rgb = list(cand)
    x, y, yaw = env.place_tracker_behind(); print('placed', round(x), round(y), round(yaw))
    b1, n1, m1, _ = env.frame(); time.sleep(0.3); b2, n2, m2, _ = env.frame()
    print('frame1 box', b1, 'npx', n1, '| frame2 box', b2, 'npx', n2)
    # also check with the ORIGINAL colour how big the box was (collision evidence)
    env.tgt_rgb = [223, 63, 0]; b3, n3, _, _ = env.frame(); print('with old colour: box', b3, 'npx', n3)
    msk = env.api.get_image(env.cam, 'object_mask'); cv2.imwrite(os.path.join(HERE, '..', 'results', 'logs', 'uz_smoke', 'probe7_mask.png'), msk)
finally:
    env.close()
