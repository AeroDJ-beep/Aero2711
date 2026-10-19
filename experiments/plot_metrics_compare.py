"""Visualize mode differences (none / cv / ca) in a metrics JSON, per profile, per seed.
usage: python plot_metrics_compare.py [path/to/metrics_*.json]
"""
import json, sys, os, statistics as st
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ['Microsoft YaHei', 'SimHei']:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
MET = next((a.split('=')[1] for a in sys.argv if a.startswith('--metric=')), 'reacq')
UNIT = {'reacq': 'px', 'reacq_cm': 'cm', 'reacq_dr': '% range'}.get(MET, '')
argv = [a for a in sys.argv if not a.startswith('--')]
SRC = argv[1] if len(argv) > 1 else os.path.join(
    HERE, '..', 'results', 'metrics', 'metrics_fullarch_none_cv_ca_2026-09-19.json')
OUT = os.path.join(HERE, '..', 'results', 'figures',
                   'metrics_compare_' + os.path.basename(SRC).replace('metrics_', '').replace('.json', '') + ('' if MET == 'reacq' else '_' + MET) + '.png')
d = json.load(open(SRC))
modes = [m for m in ['none', 'cv', 'ca', 'cv_r01', 'ca_r01'] if any(r['mode'] == m for r in d)]
profs = [p for p in ['p2', 'p2a', 'rand', 'randa'] if any(r['profile'] == p for r in d)]
col = {'none': '#9e9e9e', 'cv': '#1f77b4', 'ca': '#d62728', 'cv_r01': '#6baed6', 'ca_r01': '#fb6a4a'}
lab = {'p2': 'p2 · 拐角冲刺 sprint (80cm/s²→99)', 'p2a': 'p2a · 拐角随机加/减速 [−80,80]'}
ok = lambda v: [abs(x) if MET == 'reacq_dr' else x for x in v if x is not None]


def get(p, m, k):
    return {r['seed']: r[k] for r in d if r['profile'] == p and r['mode'] == m}


fig, axs = plt.subplots(len(profs), 3, figsize=(16, 4.6 * len(profs)))
axs = np.atleast_2d(axs)
for i, p in enumerate(profs):
    seeds = sorted(get(p, modes[0], 'seed'))
    xs = np.arange(len(modes))

    # (1) reacq per seed, paired lines
    ax = axs[i, 0]
    per = {m: [get(p, m, MET)[s] for s in seeds] for m in modes}
    top = max(max(ok(v)) for v in per.values())
    for s_i, s in enumerate(seeds):
        ys = [per[m][s_i] for m in modes]
        if None not in ys:
            ax.plot(xs, ys, '-', color='0.75', lw=0.8, zorder=1)
    for j, m in enumerate(modes):
        v = ok(per[m]); nmiss = len(per[m]) - len(v)
        ax.scatter([j] * len(v), v, color=col[m], s=28, zorder=2)
        mu = st.mean(v); md = st.median(v)
        ax.hlines(mu, j - .28, j + .28, color='k', lw=2.2, zorder=3)
        txt = 'mean %.1f\nmed %.1f' % (mu, md) + ('\n(n/a x%d)' % nmiss if nmiss else '')
        ax.text(j, top * 1.02, txt, ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(modes)
    ax.set_ylabel(f're-acquisition error  {MET} ({UNIT})')
    ax.set_title(lab.get(p, p) + '\nreacq 逐 seed（灰线 = 同一 seed，黑横线 = 均值）', fontsize=10)
    ax.set_ylim(0, top * 1.3)
    ax.grid(axis='y', alpha=.3)

    # (2) AR bars + SR
    ax = axs[i, 1]
    for j, m in enumerate(modes):
        ar = [get(p, m, 'AR')[s] for s in seeds]
        sr = st.mean([get(p, m, 'SR')[s] for s in seeds])
        ax.bar(j, st.mean(ar), yerr=st.pstdev(ar), color=col[m], capsize=4, width=.6)
        ax.text(j, st.mean(ar) + 3, '%.1f\nSR=%.2f' % (st.mean(ar), sr), ha='center', va='bottom', fontsize=9)
    lo = min(min(get(p, m, 'AR').values()) for m in modes)
    ax.set_ylim(lo - 30, 500); ax.set_xticks(xs); ax.set_xticklabels(modes)
    ax.set_ylabel('AR (accumulated reward, max 500)')
    ax.set_title('AR / SR（误差棒 = 20 seed 标准差）', fontsize=10); ax.grid(axis='y', alpha=.3)

    # (3) cv vs ca scatter per seed
    ax = axs[i, 2]
    if 'cv' in modes and 'ca' in modes:
        cv = get(p, 'cv', MET); ca = get(p, 'ca', MET)
        seeds2 = [s for s in seeds if cv[s] is not None and ca[s] is not None]
        x = [cv[s] for s in seeds2]; y = [ca[s] for s in seeds2]
        t = max(max(x), max(y)) * 1.1
        ax.plot([0, t], [0, t], 'k--', lw=1, label='cv = ca')
        ax.scatter(x, y, color='#7b2cbf', s=32)
        for s, xi, yi in zip(seeds2, x, y):
            ax.annotate(str(s), (xi, yi), fontsize=7, xytext=(3, 3), textcoords='offset points')
        win = sum(yi < xi for xi, yi in zip(x, y))
        ax.set_xlim(0, t); ax.set_ylim(0, t)
        ax.set_xlabel(f'cv {MET} ({UNIT})'); ax.set_ylabel(f'ca {MET} ({UNIT})')
        ax.set_title('逐 seed: ca 优于 cv 的 seed 数 = %d/%d\n（点在虚线下方 = ca 更准）' % (win, len(seeds2)), fontsize=10)
        ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=.3)

fig.suptitle(os.path.basename(SRC), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140); print('saved', OUT)

# text table of secondary counters
hdr = ('prof', 'mode', 'AR', 'reacq', 'occl', 'fade', 'clamp', 'caps', 'hits', 'wall_s')
print('%-5s%-7s%7s%8s%6s%6s%7s%6s%5s%7s' % hdr)
for p in profs:
    for m in modes:
        rs = [r for r in d if r['profile'] == p and r['mode'] == m]
        f = lambda k: st.mean([r[k] for r in rs if r[k] is not None])
        print('%-5s%-7s%7.1f%8.1f%6.1f%6.1f%7.1f%6.1f%5.0f%7.1f' % (
            p, m, f('AR'), f('reacq'), f('occl'), f('fade'), f('clamp'), f('caps'), f('hits'), f('wall_s')))
