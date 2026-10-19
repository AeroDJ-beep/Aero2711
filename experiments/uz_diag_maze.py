# -*- coding: utf-8 -*-
"""One logged maze episode: top view with walls + route + corner, range /
bearing / speed / detection timelines, first-person snapshots every 10 steps.
usage: python uz_diag_maze.py <maze> <mode> <seed> [profile] [--edgemeas ...]"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.patches as mp

a = [x for x in sys.argv[1:] if not x.startswith('--')]
MAZE, MODE, SEED = a[0], a[1], int(a[2]); PROF = a[3] if len(a) > 3 else 'p2'
for f in R.FLAGS:
    R.FLAGS[f] = ('--' + f) in sys.argv
R.C.MAZE_CTRL = R.FLAGS['mazectrl']
for x in sys.argv:                                   # planner parameters (same names as uz_sweep)
    if x.startswith('--T='): R.DP_TRIGGER = int(x[4:])
    if x.startswith('--E='): R.ASTAR_EVERY = int(x[4:])
    if x.startswith('--D='): R.ASTAR_DSET = float(x[4:])
    if x.startswith('--L='): R.ASTAR_LOOK = float(x[4:])
DP_LOG_ON = R.FLAGS['astar'] or R.FLAGS['dp']
if DP_LOG_ON:
    R.DP_LOG = []
if '--single' in sys.argv:
    R.uz_maze.MazeTarget.MIN_AFTER = 600.0
tag = f'{MAZE}_{MODE}_s{SEED}_{PROF}' + ''.join('_' + x[2:].replace('=', '') for x in sys.argv if x.startswith(('--T=', '--E=', '--D=', '--L=', '--astar', '--egocomp')))
snap = os.path.join(HERE, '..', 'results', 'logs', 'uz_maze', tag); os.makedirs(snap, exist_ok=True)
R.SNAP = (snap, 10)
env = U.UZEnv('FlexibleRoom')
try:
    geo = R.setup_maze(env, MAZE)
    log = []
    STEPS = int(next((x.split('=')[1] for x in sys.argv if x.startswith('--steps=')), 1200 if geo.info.get('gates') else 400))
    res = R.run_episode(env, MODE, SEED, PROF, steps=STEPS, log=log)
    print('[ep]', json.dumps(res))
finally:
    env.close()
L = np.array(log, float)
np.save(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}.npy'), L)
if R.DP_LOG is not None:
    json.dump(R.DP_LOG, open(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}_plans.json'), 'w'))
t = np.arange(len(L))
fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.3])
ax = fig.add_subplot(gs[:, 0])
for (x0, x1, y0, y1) in geo.rects:
    ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, color='#333333'))
ax.plot(L[:, 3], L[:, 4], '-', color='#E07B00', lw=1.6, label='target')
ax.plot(L[:, 0], L[:, 1], '-', color='#1668C7', lw=1.6, label='tracker')
ax.plot(L[0, 3], L[0, 4], 'o', color='#E07B00'); ax.plot(L[0, 0], L[0, 1], 'o', color='#1668C7')
ax.plot(L[-1, 3], L[-1, 4], 's', color='#E07B00'); ax.plot(L[-1, 0], L[-1, 1], 's', color='#1668C7')
blind = L[:, 5] < 0.5
ax.plot(L[blind, 3], L[blind, 4], '.', color='#C81E1E', ms=3, label='target while undetected')
ax.set_aspect('equal'); ax.autoscale(); ax.legend(loc='upper right', fontsize=8)
ax.set_title(f'{tag}: AR {res["AR"]} EL {res["EL"]} SR {res["SR"]} corner@{res["c"]}cm')
ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(t, L[:, 11], 'k'); ax2.axhline(750, color='r', ls='--'); ax2.axhline(250, color='g', ls=':'); ax2.set_ylabel('range (cm)')
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2); ax3.plot(t, L[:, 12], 'k'); [ax3.axhline(v, color='r', ls='--') for v in (45, -45)]; ax3.set_ylabel('bearing (deg)')
ax4 = fig.add_subplot(gs[2, 1], sharex=ax2); ax4.plot(t, L[:, 13], 'b', label='v_cmd'); ax4.plot(t, L[:, 14], 'g', label='|v|'); ax4.set_ylabel('cm/s'); ax4.legend(fontsize=8)
ax5 = fig.add_subplot(gs[3, 1], sharex=ax2); ax5.plot(t, L[:, 9], 'c'); ax5.set_ylabel('target px'); ax5.set_xlabel('step')
for a_ in (ax2, ax3, ax4, ax5): a_.grid(alpha=.3)
fig.tight_layout(); p = os.path.join(HERE, '..', 'results', 'figures', f'uz_ep_{tag}.png'); fig.savefig(p, dpi=110); print('saved', p)
