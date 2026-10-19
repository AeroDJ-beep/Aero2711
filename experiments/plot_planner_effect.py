# -*- coding: utf-8 -*-
"""Effect of the A* planner (astar_T1) vs. no planner (ego_wp), same seeds.
(a) re-acquisition error per seed (20 paired episodes, uz_sweep_final.jsonl)
(b) error split along / across the target's motion (per event, uz_sweep_diag.jsonl)
(c) error vs. how long the target was hidden (per event)
-> results/figures/planner_effect.png"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(HERE, '..', 'results', 'metrics')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COL = {'ego_wp': '#eb6834', 'astar_T1': '#2a78d6'}
NAME = {'ego_wp': '无规划器 no planner', 'astar_T1': 'A* 规划器 A* planner'}

def load(f):
    return [json.loads(l) for l in open(os.path.join(M, f))]

fin = {}
for r in load('uz_sweep_final.jsonl'):
    if r['setting'] in COL:
        fin.setdefault(r['setting'], {})[r['seed']] = r
ev = {}
for r in load('uz_sweep_diag.jsonl'):
    ev.setdefault(r['setting'], []).extend(r.get('reacq_ev') or [])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1, 1.2]))
for a in ax:
    a.set_facecolor('#fcfcfb'); a.grid(axis='y', color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

# (a) paired per seed, sorted by the no-planner error
seeds = sorted(fin['ego_wp'], key=lambda s: fin['ego_wp'][s]['reacq_all_cm'])
e0 = np.array([fin['ego_wp'][s]['reacq_all_cm'] for s in seeds]); e1 = np.array([fin['astar_T1'][s]['reacq_all_cm'] for s in seeds])
x = np.arange(len(seeds))
ax[0].vlines(x, np.minimum(e0, e1), np.maximum(e0, e1), color='#c3c2b7', lw=2, zorder=1)
for g, e in (('ego_wp', e0), ('astar_T1', e1)):
    ax[0].scatter(x, e, s=46, color=COL[g], edgecolor='#fcfcfb', lw=1.5, zorder=3, label=f'{NAME[g]}  均值 {e.mean():.0f} cm')
ax[0].set_xticks(x, [str(s) for s in seeds], fontsize=8)
ax[0].set_xlabel('seed（按无规划器误差排序）', color=INK2); ax[0].set_ylabel('重获误差 re-acq error (cm)', color=INK2)
ax[0].set_title(f'(a) 每局配对：A* 更好 {int((e1 < e0).sum())}/20 局', loc='left', color=INK, fontsize=11)
ax[0].legend(frameon=False, fontsize=9, loc='upper left'); ax[0].set_ylim(0, None)

# (b) along / across
cats = ['前后 along', '左右 across', '总 total']
w = 0.36
for i, g in enumerate(('ego_wp', 'astar_T1')):
    E = ev[g]
    v = [np.nanmean([abs(e['along']) for e in E]), np.nanmean([abs(e['cross']) for e in E]), np.mean([e['e'] for e in E])]
    b = ax[1].bar(np.arange(3) + (i - 0.5) * w, v, w - 0.03, color=COL[g], label=NAME[g])
    for xi, vi in zip(np.arange(3) + (i - 0.5) * w, v):
        ax[1].text(xi, vi + 1.2, f'{vi:.0f}', ha='center', va='bottom', fontsize=9, color=INK)
ax[1].set_xticks(np.arange(3), cats); ax[1].set_ylabel('平均误差 mean |error| (cm)', color=INK2)
ax[1].set_title(f'(b) 误差分解（{len(ev["ego_wp"])} 次重获/组）', loc='left', color=INK, fontsize=11)
ax[1].legend(frameon=False, fontsize=9, loc='upper left'); ax[1].set_ylim(0, 85)

# (c) error vs. hidden time
for g in ('ego_wp', 'astar_T1'):
    E = ev[g]
    ax[2].scatter([e['blind'] * 0.1 for e in E], [e['e'] for e in E], s=40, color=COL[g], alpha=0.85, edgecolor='#fcfcfb', lw=1.2,
                  label=f'{NAME[g]}  平均遮挡 {np.mean([e["blind"] for e in E]) * 0.1:.1f} s')
ax[2].set_xlabel('遮挡时长 hidden time (s)', color=INK2); ax[2].set_ylabel('重获误差 re-acq error (cm)', color=INK2)
ax[2].set_title('(c) 误差与遮挡时长', loc='left', color=INK, fontsize=11)
ax[2].legend(frameon=False, fontsize=9, loc='upper left'); ax[2].set_ylim(0, 160)

fig.tight_layout()
out = os.path.join(HERE, '..', 'results', 'figures', 'planner_effect.png')
fig.savefig(out, dpi=150, facecolor='#fcfcfb'); print('saved', out)
