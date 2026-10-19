# -*- coding: utf-8 -*-
"""Run one logged episode on a UnrealZoo scene and plot: top view on the
navgrid, range / bearing / speed / loop-period timelines.
usage: python uz_diag_episode.py <MAP> <mode> <seed> [profile] [--edgemeas ...]"""
import os, sys, time, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

a = [x for x in sys.argv[1:] if not x.startswith('--')]
MAP, MODE, SEED = a[0], a[1], int(a[2]); PROF = a[3] if len(a) > 3 else 'p2'
for f in R.FLAGS:
    R.FLAGS[f] = ('--' + f) in sys.argv
env = U.UZEnv(MAP)
try:
    env.spawn(); nav = env.build_navgrid()
    log = []
    res = R.run_episode(env, MODE, SEED, PROF, steps=400, log=log)
    print('[ep]', json.dumps(res))
finally:
    env.close()
L = np.array(log, float)
tag = f'{MAP}_{MODE}_s{SEED}_{PROF}'
np.save(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}.npy'), L)
t = np.arange(len(L))
fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.3])
ax = fig.add_subplot(gs[:, 0])
ax.imshow(nav.free, cmap='gray', origin='lower', extent=[nav.x0, nav.x0 + nav.free.shape[1] * nav.cell, nav.y0, nav.y0 + nav.free.shape[0] * nav.cell])
ax.plot(L[:, 3], L[:, 4], '-', color='#E07B00', lw=1.5, label='target'); ax.plot(L[:, 0], L[:, 1], '-', color='#1668C7', lw=1.5, label='tracker')
ax.plot(L[0, 3], L[0, 4], 'o', color='#E07B00'); ax.plot(L[-1, 0], L[-1, 1], 's', color='#1668C7'); ax.plot(L[-1, 3], L[-1, 4], 's', color='#E07B00')
ph = [C for C in [None]]
ax.set_aspect('equal'); ax.legend(loc='upper right'); ax.set_title(f'{tag}: AR {res["AR"]} EL {res["EL"]} SR {res["SR"]}')
ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(t, L[:, 11], 'k'); ax2.axhline(750, color='r', ls='--'); ax2.axhline(250, color='g', ls=':'); ax2.set_ylabel('range (cm)')
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2); ax3.plot(t, L[:, 12], 'k'); ax3.axhline(45, color='r', ls='--'); ax3.axhline(-45, color='r', ls='--'); ax3.set_ylabel('bearing (deg)')
ax4 = fig.add_subplot(gs[2, 1], sharex=ax2); ax4.plot(t, L[:, 13], 'b', label='v_cmd'); ax4.plot(t, L[:, 14], 'g', label='|v| realised'); ax4.set_ylabel('cm/s'); ax4.legend(fontsize=8)
ax5 = fig.add_subplot(gs[3, 1], sharex=ax2); ax5.plot(t, L[:, 10] * 1000, 'm'); ax5.set_ylabel('loop ms'); ax5.set_xlabel('step')
ax5b = ax5.twinx(); ax5b.plot(t, L[:, 5], 'c', alpha=.5); ax5b.set_ylabel('detected')
for a_ in (ax2, ax3, ax4, ax5): a_.grid(alpha=.3)
fig.tight_layout(); p = os.path.join(HERE, '..', 'results', 'figures', f'uz_ep_{tag}.png'); fig.savefig(p, dpi=110); print('saved', p)
print('loop period: mean %.0f ms, p95 %.0f ms; realised speed mean %.0f cm/s' % (L[:, 10].mean() * 1000, np.percentile(L[:, 10], 95) * 1000, L[:, 14].mean()))
