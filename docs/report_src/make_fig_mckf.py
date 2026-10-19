# Two illustrations of the MC-KF steps, drawn from a real episode
# (full system with CV, acceleration-limited camera, random motion, seed 2004; data from
# experiments/collect_mckf_figs.py).
#   fig_repair.png : first-person view of a fade-out frame (measurement repair)
#   fig_clamp.png  : top view of the first blind frame (map projection)
import os
import json
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r'D:\Aero2711\results\logs\mckf_figs_dyn'
SEED, K_FADE, K_BLIND = 2004, 199, 208
OUTDIRS = [os.path.join(HERE, 'figs'), os.path.join(HERE, '..', 'latex', 'figs')]
FX, CX0 = 364.3, 320.0
WALLS = [(-685, -585, -530, 70), (-685, -585, 70, 395), (-435, -335, -530, 145),
         (-435, 40, 145, 245), (-710, -35, 395, 495)]
FREE = [(-565, -455, -680, 375), (-565, 880, 265, 375)]   # reachable strips
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

d = json.load(open(os.path.join(DATA, f's{SEED}.json')))
L = d['steplog']


def save(fig, name):
    for o in OUTDIRS:
        fig.savefig(os.path.join(o, name), dpi=220, bbox_inches='tight')


# ---------------- first-person view: measurement repair ----------------
e = [x for x in d['edge'] if int(x[0]) == K_FADE][0]
_, bl, br, u, wg, cxc, npx, _, _ = e
tx, ty, yaw = L[K_FADE - 1][:3]
Px, Py = L[K_FADE][3:5]
rel = (math.degrees(math.atan2(Py - ty, Px - tx)) - yaw + 180) % 360 - 180
u_true = CX0 + FX * math.tan(math.radians(rel))
lit = cv2.cvtColor(cv2.imread(os.path.join(DATA, f's{SEED}_k{K_FADE:03d}_lit.png')), cv2.COLOR_BGR2RGB)
mask = cv2.imread(os.path.join(DATA, f's{SEED}_k{K_FADE:03d}_mask.png'))
red = (mask[:, :, 2] > 200) & (mask[:, :, 1] < 60) & (mask[:, :, 0] < 60)
ys = np.where(red.any(axis=1))[0]
top, bot = int(ys.min()), int(ys.max())
cen = (bl + br) / 2

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.imshow(lit)
ax.add_patch(Rectangle((bl, top), br - bl, bot - top, fill=False, ec='#dc2626', lw=1.8))
ax.add_patch(Rectangle((br - wg, top), wg, bot - top, fill=False, ec='#16a34a', lw=1.6, ls='--'))
ax.axvline(u, color='#d97706', ls=':', lw=1.8)
for x, c in [(cen, '#dc2626'), (cxc, '#16a34a'), (u_true, '#2563eb')]:
    ax.plot([x, x], [top - 18, bot + 18], color=c, lw=2)
ax.set_xlim(150, 430); ax.set_ylim(bot + 45, top - 60)
ax.axis('off')
handles = [plt.Line2D([], [], color='#dc2626', lw=2, label=f'detected box (cut off) and its centre: {cen:.0f} px'),
           plt.Line2D([], [], color='#d97706', lw=2, ls=':', label=f'occluding edge from the map: {u:.0f} px'),
           plt.Line2D([], [], color='#16a34a', lw=2, ls='--', label=f'remembered width from the visible edge; repaired centre: {cxc:.0f} px'),
           plt.Line2D([], [], color='#2563eb', lw=2, label=f'true centre: {u_true:.0f} px')]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=False)
save(fig, 'fig_repair.png')
plt.close(fig)
print(f'repair frame k={K_FADE}: raw {cen:.0f}, repaired {cxc:.0f}, true {u_true:.0f}')

# ---------------- top view: map projection ----------------
c = [x for x in d['clamp'] if x['k'] == K_BLIND and x['clamped']][0]
cx_, cy_, cyaw = c['cam']
tru = L[K_BLIND][3:5]
fig, ax = plt.subplots(figsize=(6.0, 4.6))
for (x0, x1, y0, y1) in WALLS:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='#9ca3af', ec='#4b5563', lw=0.8))
for i, (x0, x1, y0, y1) in enumerate(FREE):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='#dcfce7', ec='#16a34a', lw=0.8, ls='--', alpha=0.7,
                           label='free space (where the target can be)' if i == 0 else None))
cam_tr = np.array([r[:2] for r in L[K_BLIND - 60:K_BLIND]])
tgt_tr = np.array([r[3:5] for r in L[K_BLIND - 60:K_BLIND + 1]])
ax.plot(tgt_tr[:, 0], tgt_tr[:, 1], color='#2563eb', lw=1, alpha=0.5)
ax.plot(cam_tr[:, 0], cam_tr[:, 1], color='#374151', lw=1, alpha=0.5)
ax.plot(cx_, cy_, 's', ms=7, color='#374151', label='camera (arrow: heading)')
hx, hy = math.cos(math.radians(cyaw)), math.sin(math.radians(cyaw))
ax.add_patch(FancyArrowPatch((cx_, cy_), (cx_ + 70 * hx, cy_ + 70 * hy), arrowstyle='-|>',
                             mutation_scale=12, color='#374151', lw=1.5))
ax.plot([cx_, c['pre'][0]], [cy_, c['pre'][1]], ls='--', color='#dc2626', lw=1, alpha=0.7,
        label='predicted bearing and range')
ax.plot(*c['pre'], 'x', color='#dc2626', ms=10, mew=2.5, label='filter prediction (inside the wall)')
ax.plot(*c['post'], 'o', color='#16a34a', ms=8, label='after map projection')
ax.plot(*tru, 'o', color='#2563eb', ms=8, label='true target position')
ax.add_patch(FancyArrowPatch(c['pre'], c['post'], arrowstyle='-|>', mutation_scale=12, color='#16a34a', lw=1.5))
ax.set_xlim(-720, -100); ax.set_ylim(-60, 520); ax.set_aspect('equal')
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
ax.legend(loc='lower right', fontsize=7.5, framealpha=0.95)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
save(fig, 'fig_clamp.png')
print(f"clamp frame k={K_BLIND}: moved {math.dist(c['pre'], c['post']):.0f} cm; "
      f"error to truth {math.dist(c['pre'], tru):.0f} -> {math.dist(c['post'], tru):.0f} cm")
