# -*- coding: utf-8 -*-
"""Dump the UnrealCV server command help to results/logs/uz_help.txt and print
lines about scene/occupancy/shared/reachable/lidar/nav."""
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U
env = U.UZEnv('Hotel_Corridor', sleep_time=25)
try:
    h = env.c.request('vget /unrealcv/help')
    p = os.path.join(HERE, '..', 'results', 'logs', 'uz_help.txt')
    open(p, 'w', encoding='utf-8').write(h); print('saved', p, len(h.splitlines()), 'lines')
    for line in h.splitlines():
        if re.search(r'scene|occup|shared|reachable|lidar|/nav|safe|spawn_from', line, re.I):
            print('  ', line.strip()[:140])
finally:
    env.close()
