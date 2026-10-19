# -*- coding: utf-8 -*-
"""UnrealZoo smoke test 2: tracker first-person camera, target mask colour +
bbox, spawned wall visibility in RGB and mask, top-down view, reachable points.
Outputs -> results/logs/uz_smoke/<MAP>_*"""
import os, sys, json, time, math
import numpy as np, cv2
from unrealcv.launcher import RunUnreal
from unrealcv.api import UnrealCv_API

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = r'E:\UnrealZoo\UnrealZoo_UE5_7_Win64_v3.1.0\Windows\UnrealZoo_UE5_7\Binaries\Win64\UnrealZoo_UE5_7.exe'
MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_smoke'); os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(os.path.join(HERE, '..', 'unrealzoo-gym', 'gym_unrealcv', 'envs', 'setting', 'Track', MAP + '.json')))
starts = cfg['safe_start']; CHAR = cfg['agents']['player']['asset_path'][0]
TOP_H = cfg['third_cam']['height_top_view']

def save(name, img):
    p = os.path.join(OUT, f'{MAP}_{name}.png'); cv2.imwrite(p, img); print('saved', os.path.basename(p))

runner = RunUnreal(ENV_BIN=EXE, ENV_MAP=MAP)
if not os.path.isfile(runner.path2unrealcv):
    open(runner.path2unrealcv, 'w').write('[UnrealCV.Core]\nPort=9000\nWidth=640\nHeight=480\n')
ip, port = runner.start(resolution=(640, 480), sleep_time=30)
try:
    api = UnrealCv_API(port=port, ip=ip, resolution=(640, 480)); c = api.client
    s0, s1 = starts[0], starts[1]
    # tracker 250 cm "behind" the target along -y, both facing +y
    tgt_loc = [s0[0], s0[1], s0[2]]; trk_loc = [s0[0], s0[1] - 250.0, s0[2]]
    for name, s in [('tgt', tgt_loc), ('trk', trk_loc)]:
        c.request([f'vset /objects/spawn_from_path {CHAR} {name}', f'vset /object/{name}/location {s[0]} {s[1]} {s[2]}',
                   f'vset /object/{name}/rotation 0 90 0', f'vbp {name} set_phy 0'], -1)
    time.sleep(1.5)
    cams = c.request('vget /cameras').split(); print('cameras:', cams)
    cam_trk = next(i for i, n in enumerate(cams) if 'trk' in n); cam_top = 0
    print('trk cam id', cam_trk)
    # first-person camera on the tracker: at head height, no offset backwards
    print('set_cam ->', c.request('vbp trk set_cam 40 0 65 0 0 0'))   # in front of the head, no chase offset
    time.sleep(0.5)
    import re
    raw = c.request('vget /object/tgt/color'); print('tgt mask colour raw', raw)
    rgb = [int(v) for v in re.findall(r'[RGB]=(\d+)', raw)]
    def tgt_mask(img):                       # object_mask image is BGR
        d = np.abs(img.astype(int) - np.array(rgb[::-1])[None, None, :]).sum(axis=2)
        return (d < 9).astype(np.uint8)
    def bbox(m):
        ys, xs = np.where(m > 0)
        return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())] if len(xs) else None
    lit = api.get_image(cam_trk, 'lit'); save('fp_lit', lit)
    msk = api.get_image(cam_trk, 'object_mask'); save('fp_mask', msk)
    m = tgt_mask(msk); print('tgt px', int(m.sum()), 'box', bbox(m))
    print('cam pose', c.request(f'vget /camera/{cam_trk}/location'), c.request(f'vget /camera/{cam_trk}/rotation'), 'fov', c.request(f'vget /camera/{cam_trk}/fov'))
    # spawn a wall between them, offset to the right so half the body is clipped
    r = c.request(['vset /objects/spawn_from_path /Engine/BasicShapes/Cube.Cube wall_test',
                   f'vset /object/wall_test/location {s0[0] + 45} {s0[1] - 120} {s0[2] + 60}',
                   'vset /object/wall_test/scale 1.0 0.3 2.4'], -1)
    time.sleep(0.8)
    print('wall', c.request('vget /object/wall_test/location'), 'colour', c.request('vget /object/wall_test/color'))
    lit = api.get_image(cam_trk, 'lit'); save('fp_wall_lit', lit)
    msk = api.get_image(cam_trk, 'object_mask'); save('fp_wall_mask', msk)
    m = tgt_mask(msk); print('tgt px with wall', int(m.sum()), 'box', bbox(m))
    # top view from the free camera 0
    api.set_cam_location(cam_top, [s0[0], s0[1] - 100, s0[2] + TOP_H]); api.set_cam_rotation(cam_top, [-90, 90, 0])
    time.sleep(0.5); save('top_lit', api.get_image(cam_top, 'lit')); save('top_mask', api.get_image(cam_top, 'object_mask'))
    # reachable points around the target
    r = c.request(f'vget /reachablepoints/inradius {s0[0]} {s0[1]} {s0[2]} 800')
    try:
        j = json.loads(r); pts = j.get('points', j); print('reachable points:', len(pts), 'sample', pts[:2])
    except Exception:
        print('reachablepoints raw:', r[:200])
    # occupancy / perception summary keys
    j = json.loads(c.request('vget /scene/perception')); print('perception keys:', list(j.keys())[:12], 'objects:', len(j.get('objects', [])))
finally:
    try: runner.close()
    except Exception: pass
