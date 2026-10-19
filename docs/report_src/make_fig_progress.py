# Figure 5: re-acquisition error at each development stage
# (acceleration-limited camera, random target motion, 40 seeds;
#  data: results/metrics/metrics_dyn_all_margin30_2026-09-30.json).
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(HERE, 'figs', 'fig_progress.png'),
        os.path.join(HERE, '..', 'latex', 'figs', 'fig_progress.png')]
R = (json.load(open(r'D:\Aero2711\results\metrics\metrics_dyn_all_margin30_2026-09-30.json'))
     + json.load(open(r'D:\Aero2711\results\metrics\metrics_dyn_safebase_margin30_2026-09-30.json')))


def stat(cfg):
    rs = [r for r in R if r['cfg'] == cfg]
    rq = [r['reacq'] for r in rs if r['reacq'] is not None]
    return float(np.mean(rq)), float(np.mean([r['SR'] for r in rs]))


stages = [('bs_cv', 'bs_ca'), ('st3_cv_r01', 'st3_ca_r01'), ('st4_cv_r01', 'st4_ca_r01'), ('st5_cv_r01', 'st5_ca_r01')]
labels = ['Baseline\n(original noise)', 'Calibrated\nnoise',
          '+ measurement repair\n+ edge aiming', '+ blind speed\n+ map projection']
cv = [stat(a) for a, _ in stages]
ca = [stat(b) for _, b in stages]
none_full = stat('full_none')[0]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, ax = plt.subplots(figsize=(8.0, 3.8))
x = np.arange(len(stages)); w = 0.36
b1 = ax.bar(x - w / 2, [v for v, _ in cv], w, color='#2e75b6', label='CV')
b2 = ax.bar(x + w / 2, [v for v, _ in ca], w, color='#c0392b', label='CA')
for bars, data in ((b1, cv), (b2, ca)):
    for r, (v, sr) in zip(bars, data):
        ax.text(r.get_x() + r.get_width() / 2, v + 2, f'{v:.0f}', ha='center', fontsize=8)
for i, ((_, s1), (_, s2)) in enumerate(zip(cv, ca)):
    ax.text(i, 146, f'SR {s1:.2f} / {s2:.2f}', ha='center', fontsize=7.5, color='#374151')
ax.axhline(none_full, color='#6b7280', ls='--', lw=1)
ax.text(len(stages) - 0.55, none_full + 2, f'full system, no predictor: {none_full:.0f} px',
        ha='right', fontsize=7.5, color='#4b5563')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('mean re-acquisition error (px)')
ax.set_ylim(0, 152)
ax.legend(fontsize=8, loc='center right', frameon=False)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o, dpi=220)
print('cv', cv); print('ca', ca)
