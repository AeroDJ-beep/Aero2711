# -*- coding: utf-8 -*-
"""UnrealZoo smoke test on Hotel_Corridor: launch, spawn a target human and a
tracker human, read RGB + object mask, navigate the target, spawn a wall cube,
probe scene-perception/occupancy.  Outputs -> results/logs/uz_smoke/"""
import os, sys, json, time
import numpy as np, cv2
from unrealcv.launcher import RunUnreal
from unrealcv.api import UnrealCv_API

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = r'E:\UnrealZoo\UnrealZoo_UE5_7_Win64_v3.1.0\Windows\UnrealZoo_UE5_7\Binaries\Win64\UnrealZoo_UE5_7.exe'
MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_smoke'); os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(os.path.join(HERE, '..', 'unrealzoo-gym', 'gym_unrealcv', 'envs', 'setting', 'Track', MAP + '.json')))
starts = cfg['safe_start']; print('safe starts:', len(starts), starts[:3])
CHAR = cfg['agents']['player']['asset_path'][0]      # spawn_from_path needs the asset path

def save(name, img):
    p = os.path.join(OUT, f'{MAP}_{name}.png'); cv2.imwrite(p, img); print('saved', p, img.shape)

runner = RunUnreal(ENV_BIN=EXE, ENV_MAP=MAP)
if not os.path.isfile(runner.path2unrealcv):       # packaged build ships without the launcher ini
    with open(runner.path2unrealcv, 'w', encoding='utf-8') as f:
        f.write('[UnrealCV.Core]\nPort=9000\nWidth=640\nHeight=480\n')
ip, port = runner.start(resolution=(640, 480), sleep_time=30)
try:
    api = UnrealCv_API(port=port, ip=ip, resolution=(640, 480))
    c = api.client
    print('status:', c.request('vget /unrealcv/status')[:200].replace('\n', ' | '))
    objs = api.get_objects(); print('objects:', len(objs))
    # spawn target + tracker humans at two safe starts
    s0, s1 = starts[0], starts[1 % len(starts)]
    for name, s in [('tgt', s0), ('trk', s1)]:
        print(name, c.request(f'vset /objects/spawn_from_path {CHAR} {name}'), c.request(f'vset /object/{name}/location {s[0]} {s[1]} {s[2]}'),
              c.request(f'vset /object/{name}/rotation 0 0 0'), c.request(f'vbp {name} set_phy 0'))
    time.sleep(1.5)
    print('cameras:', c.request('vget /cameras'))
    print('tgt loc', api.get_obj_location('tgt'), 'trk loc', api.get_obj_location('trk'))
    api.set_obj_color('tgt', [255, 0, 0])
    # top view (cam 0) and tracker view (cam 1 / 2)
    for cam in [0, 1, 2]:
        try:
            lit = api.get_image(cam, 'lit'); save(f'cam{cam}_lit', lit)
            msk = api.get_image(cam, 'object_mask'); save(f'cam{cam}_mask', msk)
            m, box = api.get_bbox(msk, 'tgt', normalize=False)
            print(f'cam{cam}: tgt mask px={int((m > 0).sum())} box={box}')
        except Exception as e:
            print(f'cam{cam} failed: {e}')
    # navigation: send target to another safe start, watch it move
    goal = starts[2 % len(starts)]
    print('nav ->', goal, c.request(f'vbp tgt nav_to_goal_bypath {goal[0]} {goal[1]} {goal[2]}'))
    for i in range(6):
        time.sleep(1.0); print('  t=%d loc' % (i + 1), api.get_obj_location('tgt'))
    lit = api.get_image(1, 'lit'); save('cam1_after_nav', lit)
    # spawn a wall cube near the tracker
    tl = api.get_obj_location('trk')
    r = c.request(f'vset /objects/spawn_from_path /Engine/BasicShapes/Cube.Cube wall_test')
    print('spawn wall:', r, c.request(f'vset /object/wall_test/location {tl[0] + 300} {tl[1]} {tl[2]}'),
          c.request('vset /object/wall_test/scale 0.3 3 2.5'), c.request('vset /object/wall_test/color 0 255 0'))
    time.sleep(1.0)
    print('wall loc', c.request('vget /object/wall_test/location'), 'in objects:', 'wall_test' in api.get_objects())
    lit = api.get_image(1, 'lit'); save('cam1_wall_lit', lit)
    msk = api.get_image(1, 'object_mask'); save('cam1_wall_mask', msk)
    # scene perception / occupancy probe
    for cmd in ['vget /scene/perception', 'vget /reachablepoints/inradius %s %s %s 500' % tuple(tl)]:
        r = c.request(cmd); print(cmd, '->', (r[:300] + '...') if isinstance(r, str) else type(r))
finally:
    try: runner.close()
    except Exception: pass
