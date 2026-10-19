# -*- coding: utf-8 -*-
"""Measure effective height (cm) and box aspect (w/h) of candidate target looks at a
known range, to match the L-map target (H 168 cm at scale 0.85, aspect ~0.32
walking away from the camera)."""
import os, sys, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U

TGT = '/Game/HumanCharacter/ThirdPersonBP/Blueprints/target.target'
env = U.UZEnv('FlexibleRoom', sleep_time=30)
try:
    spot = (-450.0, -1500.0)
    env.spawn(tgt_xy=spot, trk_xy=(spot[0], spot[1] - 450), yaw=90.0, tgt_asset=TGT, tgt_app=5, tgt_scale=0.85)
    tiles = []
    for app in [3, 4, 5]:
        env.c.request(f'vbp {env.tgt} set_app {app}'); time.sleep(0.6)
        rows = []
        for d in (300.0, 400.0, 500.0, 600.0):
            env.c.request([f'vset /object/{env.tgt}/location {spot[0]} {spot[1]} {env.z}', f'vset /object/{env.tgt}/rotation 0 90 0'], -1)
            env.set_tracker(spot[0], spot[1] - d, 90.0); time.sleep(1.0)
            env.frame(); time.sleep(0.2)
            box, npx, m, _ = env.frame()
            cam = U._floats(env.c.request(f'vget /camera/{env.cam}/location'))
            tl = U._floats(env.c.request(f'vget /object/{env.tgt}/location'))
            dc = float(np.hypot(tl[0] - cam[0], tl[1] - cam[1]))
            if box is None or box[1] <= 1 or box[1] + box[3] >= U.H - 1:
                print(f'   app {app} d {d:.0f}: box {box} clipped/none, skip'); continue
            rows.append((dc, box[3] * dc / U.FX, box[2] / box[3], box[3], cam[2] - (env.z_floor - 98.0)))
        if not rows:
            continue
        print('   rows (cam dist, H, aspect, box_h, cam height above floor):', [tuple(round(v, 2) for v in r) for r in rows])
        H = np.mean([r[1] for r in rows]); A = np.mean([r[2] for r in rows])
        print(f'app {app:2d}: H_eff {H:6.1f} cm at scale 0.85 (per range {[round(r[1], 1) for r in rows]}), aspect {A:.3f}  -> scale for 168 cm: {0.85 * 168.0 / H:.3f}')
        env.set_tracker(spot[0], spot[1] - 350, 90.0); time.sleep(0.4)
        im = env.api.get_image(env.cam, 'lit')[40:460, 200:440].copy()
        cv2.putText(im, f'{app}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2); tiles.append(im)
    cv2.imwrite(os.path.join(HERE, '..', 'results', 'logs', 'uz_room', 'char_size.png'), np.hstack(tiles))
finally:
    env.close()
