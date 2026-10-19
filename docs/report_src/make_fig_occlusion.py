# Figure 2: target pixels during one corner passage (engine sweep, d = 250 cm,
# c = 40 cm, camera on the target's path). Simplified layout (2026-09-30).
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r'D:\Aero2711\OA-VAT\data\maps\L_occlusion_sweep_w150_hug250.csv'
OUTS = [os.path.join(HERE, 'figs', 'fig_occlusion.png'),
        os.path.join(HERE, '..', 'latex', 'figs', 'fig_occlusion.png')]
V = 80.0            # target speed, cm/s
S_TURN = 735.0      # path position of the corner (cm)
THR = 20            # occlusion threshold (pixels)

rows = [l.strip().split(',') for l in open(SRC) if l.strip() and not l.startswith('#')]
s, px = np.array([[float(a), float(b)] for a, b in rows[1:] if a.replace('.', '').isdigit()]).T
s, idx = np.unique(s, return_index=True)
px = px[idx]
t = (s - S_TURN) / V          # time since the target turned the corner
base = np.median(px[t < 0])


def cross(level, falling=True):
    for i in range(len(px) - 1):
        a, b = px[i], px[i + 1]
        if (falling and a >= level > b) or (not falling and a < level <= b):
            return t[i] + (a - level) / (a - b) * (t[i + 1] - t[i])


t_fade = cross(0.95 * base)             # clipping starts
t_blind = cross(THR)                    # fully hidden
t_back = cross(THR, falling=False)      # reappears
print(f'fade {t_blind - t_fade:.2f} s, blind {t_back - t_blind:.2f} s')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, ax = plt.subplots(figsize=(8.0, 3.2))
ax.axvspan(t_fade, t_blind, color='#fde7c7', lw=0)
ax.axvspan(t_blind, t_back, color='#f8c9c9', lw=0)
ax.plot(t, px / 1000, '-o', color='#1f4e79', ms=3, lw=1.6)
ax.axvline(0, color='#6b7280', ls='--', lw=1)
ax.text(-0.1, 5.0, 'target turns\nthe corner', ha='right', va='center', fontsize=8, color='#374151')
ymax = base / 1000 * 1.25
ax.text((t_fade + t_blind) / 2, ymax * 0.93, f'fade-out\n{t_blind - t_fade:.1f} s', ha='center', va='top', fontsize=8)
ax.text((t_blind + t_back) / 2, ymax * 0.93, f'fully hidden\n{t_back - t_blind:.1f} s', ha='center', va='top', fontsize=8)
ax.set_xlim(-2, 5)
ax.set_ylim(-0.3, ymax)
ax.set_xlabel('time since the target turned the corner (s), walking at 80 cm/s')
ax.set_ylabel('visible target pixels (×1000)')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
for out in OUTS:
    fig.savefig(out, dpi=220)
print('saved')
