# -*- coding: utf-8 -*-
"""Probe: find the scene object whose mask colour matches a given RGB (invisible in lit,
present in object_mask) and report its location / bounds / visibility."""
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
want = [int(v) for v in sys.argv[1].split(',')]
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    R.setup_maze(env, 'fr10x10l35t24g_s0')
    objs = env.c.request('vget /objects').split()
    print('n objects', len(objs))
    for o in objs:
        col = [int(v) for v in re.findall(r'[RGB]=(\d+)', env.c.request(f'vget /object/{o}/color'))]
        if sum(abs(a - b) for a, b in zip(col, want)) < 9:
            print('MATCH', o, col, 'loc', env.c.request(f'vget /object/{o}/location'),
                  'bounds', env.c.request(f'vget /object/{o}/bounds'), 'scale', env.c.request(f'vget /object/{o}/scale'))
finally:
    env.close()
