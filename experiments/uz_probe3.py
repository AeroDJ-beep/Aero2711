# -*- coding: utf-8 -*-
"""UnrealZoo probe 3: what is a spawned cube actually (class metadata), which
scene assets could serve as walls, and what the perception snapshot exposes
(bounds?) for map priors."""
import os, sys, json, time, re
from unrealcv.launcher import RunUnreal
from unrealcv.api import UnrealCv_API

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = r'E:\UnrealZoo\UnrealZoo_UE5_7_Win64_v3.1.0\Windows\UnrealZoo_UE5_7\Binaries\Win64\UnrealZoo_UE5_7.exe'
MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
cfg = json.load(open(os.path.join(HERE, '..', 'unrealzoo-gym', 'gym_unrealcv', 'envs', 'setting', 'Track', MAP + '.json')))
s0 = cfg['safe_start'][0]
runner = RunUnreal(ENV_BIN=EXE, ENV_MAP=MAP)
if not os.path.isfile(runner.path2unrealcv):
    open(runner.path2unrealcv, 'w').write('[UnrealCV.Core]\nPort=9000\nWidth=640\nHeight=480\n')
ip, port = runner.start(resolution=(640, 480), sleep_time=30)
try:
    api = UnrealCv_API(port=port, ip=ip, resolution=(640, 480)); c = api.client
    objs = api.get_objects()
    kw = [o for o in objs if re.search(r'wall|Wall|door|Door|SM_|Cube|cube|Corridor|corridor', o)]
    print('scene objects with wall/door/mesh-like names (%d):' % len(kw), kw[:40])
    # class metadata of one scene actor and of a spawned cube
    for o in kw[:2]:
        print(o, 'meta ->', c.request(f'vget /object/{o}/class_metadata')[:400].replace('\n', ' '))
    print('spawn cube ->', c.request('vset /objects/spawn_from_path /Engine/BasicShapes/Cube.Cube wall_a'))
    print('wall_a meta ->', c.request('vget /object/wall_a/class_metadata')[:400].replace('\n', ' '))
    print('wall_a verts ->', c.request('vget /object/wall_a/mesh_vertices')[:200].replace('\n', ' '))
    print('spawn_cube (legacy) ->', c.request('vset /objects/spawn_cube wall_b'), c.request('vget /object/wall_b/class_metadata')[:300].replace('\n', ' '))
    # try duplicating a scene static mesh by its asset path if metadata exposes it
    for o in kw[:1]:
        meta = c.request(f'vget /object/{o}/class_metadata')
        m = re.search(r'(/Game/[^"\s]+)', meta)
        if m:
            path = m.group(1); print('try spawn scene asset', path, '->', c.request(f'vset /objects/spawn_from_path {path} wall_c'))
    # perception snapshot: what per-object fields exist?
    j = json.loads(c.request('vget /scene/perception'))
    ob = j['objects'][0] if j['objects'] else {}
    print('perception object fields:', list(ob.keys()))
    print('first object:', json.dumps(ob)[:600])
    print('agent_nav:', json.dumps(j.get('agent_nav'))[:400])
    # occupancy / other perception commands in the help
    h = c.request('vget /unrealcv/help')
    for line in h.split('\n'):
        if re.search(r'occup|percep|reachable|spawn|mesh|bound|bbox|nav', line, re.I):
            print('  help:', line.strip()[:110])
finally:
    try: runner.close()
    except Exception: pass
