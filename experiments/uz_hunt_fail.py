# -*- coding: utf-8 -*-
"""Run episodes until a loss happens and keep the full log of every lost episode.
usage: python uz_hunt_fail.py <max_episodes> [--T=1] [--L=60] [--dp]"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R

MAZE = 'fr10x10l35t24g_s0'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for f in R.FLAGS:
    R.FLAGS[f] = f in ('edgemeas', 'edgeaim', 'blindspeed', 'mapclamp', 'egocomp', 'astar')
R.FLAGS['worldpred'] = '--worldpred' in sys.argv
if '--noastar' in sys.argv:
    R.FLAGS['astar'] = False
for x in sys.argv:
    if x.startswith('--T='): R.DP_TRIGGER = int(x[4:])
    if x.startswith('--L='): R.ASTAR_LOOK = float(x[4:])
OUT = os.path.join(HERE, '..', 'results', 'logs', 'hunt'); os.makedirs(OUT, exist_ok=True)
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    R.setup_maze(env, MAZE)
    for i in range(N):
        seed = i % 3
        log = []; R.DP_LOG = []
        res = R.run_episode(env, 'ca', seed, 'p2', steps=1200, log=log)
        tag = f'ep{i}_s{seed}_T{R.DP_TRIGGER}_L{int(R.ASTAR_LOOK)}' + ('_wp' if R.FLAGS['worldpred'] else '') + ('' if R.FLAGS['astar'] else '_noastar')
        lost = not res.get('finished')
        print(f'[{tag}] finished {res.get("finished", 0)} progress {res.get("progress")} EL {res["EL"]} occl {res["occl"]} '
              f'reacq_all {res.get("reacq_all_cm")}', flush=True)
        with open(os.path.join(OUT, 'hunt_results.jsonl'), 'a') as fo:
            fo.write(json.dumps(dict(res, tag=tag)) + '\n')
        if lost:
            np.save(os.path.join(OUT, f'{tag}_LOST.npy'), np.array(log, float))
            json.dump(R.DP_LOG, open(os.path.join(OUT, f'{tag}_LOST_plans.json'), 'w'))
            print('   saved lost episode log', flush=True)
finally:
    env.close()
print('HUNT DONE', flush=True)
