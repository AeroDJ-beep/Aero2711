# -*- coding: utf-8 -*-
"""UnrealZoo probe 4: (a) can existing scene cubes be moved/rescaled as walls,
(b) does the new spawn_cube [x y z] render, (c) perception radius/max params
and how many objects carry world bboxes (map prior coverage), (d) reflection
help.  Saves results/logs/uz_smoke/<MAP>_p4_*.png"""
import os, sys, json, time, re
import numpy as np, cv2
from unrealcv.launcher import RunUnreal
from unrealcv.api import UnrealCv_API

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = r'E:\UnrealZoo\UnrealZoo_UE5_7_Win64_v3.1.0\Windows\UnrealZoo_UE5_7\Binaries\Win64\UnrealZoo_UE5_7.exe'
MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_smoke')
cfg = json.load(open(os.path.join(HERE, '..', 'unrealzoo-gym', 'gym_unrealcv', 'envs', 'setting', 'Track', MAP + '.json')))
s0 = cfg['safe_start'][0]; CHAR = cfg['agents']['player']['asset_path'][0]
def save(name, img): cv2.imwrite(os.path.join(OUT, f'{MAP}_p4_{name}.png'), img); print('saved', name)

runner = RunUnreal(ENV_BIN=EXE, ENV_MAP=MAP)
if not os.path.isfile(runner.path2unrealcv):
    open(runner.path2unrealcv, 'w').write('[UnrealCV.Core]\nPort=9000\nWidth=640\nHeight=480\n')
ip, port = runner.start(resolution=(640, 480), sleep_time=30)
try:
    api = UnrealCv_API(port=port, ip=ip, resolution=(640, 480)); c = api.client
    # tracker camera looking +y from 250 cm behind s0
    c.request([f'vset /objects/spawn_from_path {CHAR} trk', f'vset /object/trk/location {s0[0]} {s0[1] - 250} {s0[2]}',
               f'vset /object/trk/rotation 0 90 0', 'vbp trk set_phy 0'], -1)
    time.sleep(1.0); c.request('vbp trk set_cam 40 0 65 0 0 0'); time.sleep(0.5)
    cams = c.request('vget /cameras').split(); cam = next(i for i, n in enumerate(cams) if 'trk' in n)
    save('base_lit', api.get_image(cam, 'lit'))
    # (a) existing scene cubes
    for o in ['Cube_2', 'Cube2_2']:
        print(o, 'mobility', c.request(f'vget /object/{o}/mobility'), 'loc', c.request(f'vget /object/{o}/location'),
              'scale', c.request(f'vget /object/{o}/scale'))
    print('move Cube_2 ->', c.request(f'vset /object/Cube_2/mobility Movable'), c.request(f'vset /object/Cube_2/location {s0[0] + 40} {s0[1] - 100} {s0[2] + 80}'),
          c.request('vset /object/Cube_2/scale 0.6 0.2 2.0'), 'now', c.request('vget /object/Cube_2/location'))
    time.sleep(0.8); save('cube2_lit', api.get_image(cam, 'lit')); save('cube2_mask', api.get_image(cam, 'object_mask'))
    c.request('vset /object/Cube_2/location 100000 100000 -5000')   # park it away again
    # (b) new spawn_cube at coordinates
    r = c.request(f'vset /objects/spawn_cube wall_s {s0[0] - 40} {s0[1] - 100} {s0[2] + 80}'); print('spawn_cube xyz ->', r)
    print('  ', c.request('vset /object/wall_s/scale 0.6 0.2 2.0'), c.request('vget /object/wall_s/location'), c.request('vget /object/wall_s/color'))
    time.sleep(0.8); save('spawncube_lit', api.get_image(cam, 'lit')); save('spawncube_mask', api.get_image(cam, 'object_mask'))
    # (c) perception coverage
    for args in ['', ' 6000 300']:
        r = c.request('vget /scene/perception' + args)
        try:
            j = json.loads(r); objs = j['objects']
            nb = sum(1 for o in objs if o.get('bbox')); cls = {}
            for o in objs: cls[o.get('class')] = cls.get(o.get('class'), 0) + 1
            print(f'perception{args or " (default)"}: found {j.get("objects_found")} returned {len(objs)} with bbox {nb} truncated {j.get("objects_truncated")} classes {cls}')
        except Exception as e:
            print('perception', args, 'failed:', str(r)[:120])
    # (d) reflection / property commands available?
    h = c.request('vget /unrealcv/help')
    for line in h.split('\n'):
        if re.search(r'property|reflect|function|invoke|json|occup|voxel|bound|bbox', line, re.I):
            print('  help:', line.strip()[:120])
finally:
    try: runner.close()
    except Exception: pass
