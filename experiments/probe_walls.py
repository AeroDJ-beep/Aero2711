# -*- coding: utf-8 -*-
"""Maze feasibility probe: can wall1-wall4 / cube_11 be rescaled and moved
into the arena as extra wall segments, and does a spawned CubeActor render?
Writes results/logs/probe3_lit.png + probe3_mask.png."""
import os, sys, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_metrics_test as M

runner, ucv, z_trk = M.boot(); c = ucv.client
try:
    c.request('vset /objects/spawn_cube')
    new = [o for o in c.request('vget /objects').split() if o.startswith('CubeActor')]
    n = new[0]
    print('spawned', n, 'show ->', c.request(f'vset /object/{n}/show'), 'mobility ->', c.request(f'vget /object/{n}/mobility'))
    for o in ['wall1', 'cube_11', 'cube1']:
        print(o, 'mobility', c.request(f'vget /object/{o}/mobility'))
    # repurpose wall1 and cube_11 as corridor walls ahead of the camera
    print('wall1', c.request('vset /object/wall1/scale 1 3 3.5'), c.request('vset /object/wall1/location -500 -100 100'), c.request('vset /object/wall1/rotation 0 0 0'))
    print('cube_11', c.request('vset /object/cube_11/scale 1 3 3.5'), c.request('vset /object/cube_11/location -560 -250 100'))
    print('spawned', c.request(f'vset /object/{n}/location -440 -250 100'), c.request(f'vset /object/{n}/scale 1 3 3.5'))
    ucv.set_location(1, [-500, -450, z_trk]); ucv.set_rotation(1, [0, 90, M.PITCH])
    out = os.path.join(HERE, '..', 'results', 'logs')
    cv2.imwrite(os.path.join(out, 'probe3_lit.png'), ucv.read_image(1, 'lit', 'fast'))
    cv2.imwrite(os.path.join(out, 'probe3_mask.png'), ucv.read_image(1, 'object_mask', 'fast'))
    print('wall1 now', c.request('vget /object/wall1/location'), c.request('vget /object/wall1/scale'))
    print('images saved to', out)
finally:
    try:
        runner.close()
    except Exception:
        pass
