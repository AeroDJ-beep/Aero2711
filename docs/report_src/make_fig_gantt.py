# Figure 6: revised Gantt chart (updated 2026-10-01; final report due 8 Nov 2026).
import os
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(HERE, 'figs', 'fig_gantt.png'),
        os.path.join(HERE, '..', 'latex', 'figs', 'fig_gantt.png')]
D = dt.date.fromisoformat
TODAY, DUE = D('2026-10-01'), D('2026-11-08')

tasks = [  # (label, start, end, status)
    ('Reproduce OA-VAT', '2026-07-20', '2026-08-01', 'done'),
    ('Pilot tests (accelerating / corner targets)', '2026-08-05', '2026-08-12', 'done'),
    ('L-corner test environment', '2026-08-15', '2026-08-25', 'done'),
    ('Closed-loop test program; CV vs CA', '2026-08-28', '2026-09-06', 'done'),
    ('Safety layer', '2026-09-04', '2026-09-10', 'done'),
    ('MC-KF: measurement repair + edge aiming', '2026-09-10', '2026-09-16', 'done'),
    ('MC-KF: map projection', '2026-09-14', '2026-09-20', 'done'),
    ('Re-runs: random motion, camera inertia', '2026-09-28', '2026-09-30', 'done'),
    ('Literature survey', '2026-08-01', '2026-10-09', 'ongoing'),
    ('Progress report', '2026-09-24', '2026-10-02', 'ongoing'),
    ('Self-built mazes + general map module', '2026-09-28', '2026-10-12', 'ongoing'),
    ('Planner for camera motion while hidden', '2026-10-06', '2026-10-24', 'planned'),
    ('Real detector in the loop', '2026-10-13', '2026-10-26', 'planned'),
    ('Junctions: several predictions', '2026-10-17', '2026-10-28', 'planned'),
    ('IMM baseline + repeated seed sets', '2026-10-24', '2026-10-31', 'planned'),
    ('Final report', '2026-10-27', '2026-11-08', 'planned'),
]
col = {'done': '#2e7d32', 'ongoing': '#f39c12', 'planned': '#2e75b6'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for i, (lab, s, e, st) in enumerate(tasks):
    ax.barh(i, (D(e) - D(s)).days, left=D(s), height=0.6, color=col[st], edgecolor='none')
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in tasks])
ax.invert_yaxis()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
for x, txt, c in [(TODAY, 'today (1 Oct)', '#dc2626'), (DUE, 'due (8 Nov)', '#111827')]:
    ax.axvline(x, color=c, ls='--', lw=1.1)
    ax.text(x - dt.timedelta(days=1), -1.0, txt, color=c, ha='right', fontsize=8)
ax.set_xlim(D('2026-07-15'), D('2026-11-15'))
ax.set_ylim(len(tasks) - 0.4, -1.5)
ax.grid(axis='x', alpha=0.25)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[Patch(color=col[k], label=k) for k in col], loc='lower left', frameon=False, ncol=3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o, dpi=220)
print('saved')
