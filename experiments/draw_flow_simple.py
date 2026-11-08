"""Simple flow chart of one control step (0.05 s) in the early 2D baseline."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 13), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')


def box(cx, cy, w, h, title, sub, fc, ec='#555', ls='-'):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0.008', lw=1.6, edgecolor=ec, facecolor=fc, linestyle=ls
        )
    )
    ax.text(cx, cy + (0.013 if sub else 0), title, ha='center', va='center', fontsize=13, fontweight='bold')
    if sub:
        ax.text(cx, cy - 0.017, sub, ha='center', va='center', fontsize=10.5, color='#444')


def arrow(p1, p2, label='', color='#444'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=20, lw=1.8, color=color))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.012, my, label, fontsize=10.5, color=color, ha='left', va='center')


X = 0.36
XR = 0.78
H = 0.062

box(
    X,
    0.955,
    0.42,
    H,
    '1. Engine renders a frame',
    'target mask -> detection box (position / width / height)',
    '#E8EEF7',
)
arrow((X, 0.955 - H / 2), (X, 0.873 + H / 2))

box(
    X,
    0.873,
    0.50,
    H,
    '2. Is the box healthy? (pixel drop = cut by a wall)',
    'healthy -> use it | cut -> edgemeas repairs it | gone -> no measurement',
    '#E8F3E8',
)
arrow((X, 0.873 - H / 2), (X, 0.791 + H / 2))

box(
    X,
    0.791,
    0.46,
    H,
    '3. Kalman filter',
    'predict one step; correct if measured -> belief (where the target is)',
    '#FDF3DC',
)
arrow((X, 0.791 - H / 2), (X, 0.709 + H / 2))

box(
    X,
    0.709,
    0.50,
    H,
    '4. mapclamp: belief check',
    'belief left the corridor? -> push it back (people stay in corridors)',
    '#FDF3DC',
    ec='#8B2FC9',
)
arrow((X, 0.709 - H / 2), (X, 0.615 + H / 2))

box(X, 0.615, 0.34, 0.070, '5. Is the target visible now?', '', '#FFFFFF', ec='#C0392B')
# visible branch (left down)
arrow((X - 0.10, 0.615 - 0.035), (X - 0.10, 0.520 + H / 2), 'visible', '#2E7D32')
box(
    X - 0.10,
    0.520,
    0.30,
    H,
    'Normal follow',
    'yaw: aim at box centre\nspeed: from distance error',
    '#EAF5EA',
    ec='#2E7D32',
)
# blind branch (right down)
arrow((X + 0.13, 0.615 - 0.035), (XR - 0.155, 0.520 + H / 2), 'hidden', '#C0392B')
box(
    XR,
    0.520,
    0.36,
    H,
    'Blind flight',
    'yaw: aim at the corner (edgeaim)\nspeed: fixed full speed (blindspeed)',
    '#FBE9E7',
    ec='#C0392B',
)
arrow((X - 0.10, 0.520 - H / 2), (X, 0.425 + H / 2))
arrow((XR, 0.520 - H / 2), (X + 0.08, 0.425 + H / 2))

box(
    X,
    0.425,
    0.50,
    H,
    '6. vcap: speed limit near walls',
    'the wall-ward part of the velocity is cut more near the wall',
    '#EDE7F6',
)
arrow((X, 0.425 - H / 2), (X, 0.343 + H / 2))

box(X, 0.343, 0.46, H, '7. Safety fuse (safe_move)', 'end point < 15 cm from a wall? -> slide along it', '#EDE7F6')
arrow((X, 0.343 - H / 2), (X, 0.261 + H / 2))

box(X, 0.261, 0.46, H, '8. Move the camera', 'write to the engine (body placed 30 cm behind it)', '#E0F2F1')

# loop back
ax.add_patch(FancyArrowPatch((X - 0.23, 0.261), (0.055, 0.261), arrowstyle='-', lw=1.6, color='#888'))
ax.add_patch(FancyArrowPatch((0.055, 0.261), (0.055, 0.955), arrowstyle='-', lw=1.6, color='#888'))
ax.add_patch(
    FancyArrowPatch((0.055, 0.955), (X - 0.21, 0.955), arrowstyle='-|>', mutation_scale=20, lw=1.6, color='#888')
)
ax.text(0.048, 0.60, 'next step\nafter 0.05 s', fontsize=11, color='#666', ha='right', style='italic')

# side note
ax.text(
    XR,
    0.34,
    'Design rule\nwhile hidden, yaw and speed\ndo not follow the belief\n'
    '(the belief is only scored\nwhen the target reappears)',
    fontsize=11,
    color='#C0392B',
    ha='center',
    bbox=dict(boxstyle='round', facecolor='#FFF5F5', edgecolor='#C0392B'),
)

ax.set_title('One control step (the loop runs every 0.05 s)', fontsize=16, fontweight='bold', pad=14)
for base in [r'D:\Aero2711\results\figures']:
    plt.savefig(base + r'\flow_simple.png', facecolor='white', bbox_inches='tight')
print('saved')
