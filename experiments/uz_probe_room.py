# -*- coding: utf-8 -*-
"""Probe: can we build a maze inside UnrealZoo?
 1. launch the FlexibleRoom map of the UnrealZoo package, list movable blocks
 2. scan spawnable assets for cube / wall meshes or blueprints
 3. spawn candidates, scale them into a wall, check RGB + object mask visibility
 4. try the reflection bridge to give a spawned StaticMeshActor a real mesh
Images -> results/logs/uz_room/"""
import os, sys, json, time, re
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U

MAP = sys.argv[1] if len(sys.argv) > 1 else 'FlexibleRoom'
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_room'); os.makedirs(OUT, exist_ok=True)
env = U.UZEnv(MAP, sleep_time=30)
c = env.c


def req(cmd, n=300):
    r = c.request(cmd, timeout=60)
    return str(r)[:n].replace('\n', ' ')


def mask_px(img, rgb):
    d = np.abs(img.astype(np.int16) - np.array(rgb[::-1], np.int16)[None, None, :]).sum(axis=2)
    return int((d < 9).sum())


try:
    objs = env.api.get_objects()
    print('objects', len(objs), ':', objs[:80])
    # mobility query crashes the engine on actors without a root component: only ask named blocks
    blocks = [o for o in objs if re.match(r'(cube|cylinder|cone|wall|sphere|block|box)', o, re.I)]
    for o in blocks[:25]:
        print(f'  {o:14s} mob {req(f"vget /object/{o}/mobility", 10):8s} loc {req(f"vget /object/{o}/location", 40)} scale {req(f"vget /object/{o}/scale", 30)}')
    # asset scan
    r = c.request('vget /objects/scan_assets', timeout=120)
    try:
        assets = json.loads(r)
        lst = assets.get('assets', assets) if isinstance(assets, dict) else assets
    except Exception:
        lst = re.findall(r'/Game/[^\s",\]]+', str(r))
    lst = [a if isinstance(a, str) else json.dumps(a) for a in lst]
    open(os.path.join(OUT, f'{MAP}_assets.txt'), 'w', encoding='utf-8').write('\n'.join(lst))
    cand = [a for a in lst if re.search(r'cube|wall|block|box|shape|plane', a, re.I)]
    print('scanned assets', len(lst), '| cube/wall-like:', cand[:30])
    # camera: free cam 0 looking at a spot in the room
    s0 = env.starts[0]
    cx, cy, cz = s0[0], s0[1], s0[2] + 60
    c.request(f'vset /camera/0/location {cx - 400} {cy} {cz}'); c.request('vset /camera/0/rotation 0 0 0')
    time.sleep(0.5)
    cv2.imwrite(os.path.join(OUT, f'{MAP}_view0.png'), env.api.get_image(0, 'lit'))
    tried = ['/Engine/BasicShapes/Cube.Cube', '/Engine/BasicShapes/Cube', 'StaticMeshActor'] + cand[:8]
    for k, path in enumerate(tried):
        name = f'wt{k}'
        r = req(f'vset /objects/spawn_from_path {path} {name}') if path != 'StaticMeshActor' else req(f'vset /objects/spawn StaticMeshActor {name}')
        ok = name in env.api.get_objects()
        if not ok:
            print(f'[{k}] {path}: spawn -> {r[:80]}'); continue
        c.request(f'vset /object/{name}/location {cx} {cy + (k - 3) * 60} {s0[2] + 80}')
        c.request(f'vset /object/{name}/scale 0.3 0.3 1.5')
        col = [(37 * k + 50) % 256, (91 * k + 20) % 256, 200]
        c.request(f'vset /object/{name}/color {col[0]} {col[1]} {col[2]}')
        time.sleep(0.4)
        msk = env.api.get_image(0, 'object_mask')
        print(f'[{k}] {path}: spawned, meta {req(f"vget /object/{name}/class_metadata", 160)} | mask px {mask_px(msk, col)}')
    cv2.imwrite(os.path.join(OUT, f'{MAP}_spawned_lit.png'), env.api.get_image(0, 'lit'))
    cv2.imwrite(os.path.join(OUT, f'{MAP}_spawned_mask.png'), env.api.get_image(0, 'object_mask'))
    # reflection bridge: what does it offer on a StaticMeshActor?
    print('vreflect help ->', req('vreflect help', 600))
    for args in ['wt0 list_properties', 'wt0 list_functions', 'wt0 components', 'wt0.StaticMeshComponent0 list_properties']:
        print(f'vreflect {args} ->', req(f'vreflect {args}', 500))
    # try moving an existing movable block into view as a comparison
    if blocks:
        b = blocks[0]
        c.request(f'vset /object/{b}/location {cx} {cy + 250} {s0[2]}'); time.sleep(0.4)
        cv2.imwrite(os.path.join(OUT, f'{MAP}_block_lit.png'), env.api.get_image(0, 'lit'))
        print('moved', b, '->', req(f'vget /object/{b}/location', 60))
finally:
    env.close()
