# Figure 1: closed-loop system, simplified layout (2026-09-30).
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(HERE, 'figs', 'fig_arch.png'),
        os.path.join(HERE, '..', 'latex', 'figs', 'fig_arch.png')]

GREY_FC, GREY_EC = '#f4f5f7', '#6b7280'
NEW_FC, NEW_EC = '#fff4e5', '#d97706'          # parts added in this project
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(10.6, 4.0))
ax.set_xlim(-0.5, 101.5); ax.set_ylim(0, 46); ax.axis('off')
W, H = 17.0, 11


def box(x, y, title, sub, new=False, w=W, h=H):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.2',
                                fc=NEW_FC if new else GREY_FC, ec=NEW_EC if new else GREY_EC,
                                lw=1.8 if new else 1.1))
    ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=8.2, fontweight='bold')
    ax.text(x + w / 2, y + h * 0.30, sub, ha='center', va='center', fontsize=7.6, color='#374151')


def arrow(p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, lw=1.2, color='#374151', **kw))


top, bot = 32, 5
xs = [1, 21.5, 42, 62.5, 83]
box(xs[0], top, 'Simulator', 'image + object mask')
box(xs[1], top, 'Detection', 'box of the target')
box(xs[2], top, 'Measurement repair', 'fix a cut-off box', new=True)
box(xs[3], top, 'Kalman filter', 'CV or CA')
box(xs[4], top, 'Map projection', 'keep the prediction\nin free space', new=True)
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + W + 0.4, top + H / 2), (b - 0.4, top + H / 2))

box(xs[4], bot, 'Steering: hidden', 'aim at the obstacle\nedge, full speed', new=True)
box(xs[3], bot, 'Steering: visible', 'turn toward the box')
ax.text(xs[3] + W + 1.75, bot + H / 2, 'or', ha='center', va='center', fontsize=8, color='#6b7280')
box(xs[1] + 10, bot, 'Safety layer', 'limit speed\ntoward walls')
box(xs[0], bot, 'Camera motion', 'move and turn')

arrow((xs[4] + W / 2, top - 0.5), (xs[4] + W / 2, bot + H + 0.5))
arrow((xs[3] + W / 2, top - 0.5), (xs[3] + W / 2, bot + H + 0.5))
arrow((xs[3] - 0.4, bot + H / 2), (xs[1] + 10 + W + 0.4, bot + H / 2))
arrow((xs[1] + 10 - 0.4, bot + H / 2), (xs[0] + W + 0.4, bot + H / 2))
arrow((xs[0] + W / 2, bot + H + 0.5), (xs[0] + W / 2, top - 0.5))
ax.text(xs[0] + W / 2 + 1.5, (bot + H + top) / 2, 'next frame (every 50 ms)', ha='left', va='center',
        fontsize=7.6, color='#6b7280')

for out in OUTS:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches='tight')
print('saved', OUTS)
