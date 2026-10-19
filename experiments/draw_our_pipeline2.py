# -*- coding: utf-8 -*-
"""Our closed-loop pipeline, v2: two-row snake layout, generous spacing."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CHAR_W = 0.0072
LINE_H = 0.042


def box_size(title, lines):
    w = max([len(title)] + [len(s) for s in lines]) * CHAR_W + 0.030
    h = LINE_H * (len(lines) + 1) + 0.040
    return w, h


def draw_box(ax, cx, cy, title, lines, fc, ec='#555555', ls='-', tc='#111111'):
    w, h = box_size(title, lines)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.010', linewidth=1.5,
                                edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(cx, cy + h / 2 - 0.042, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color=tc)
    for i, s in enumerate(lines):
        ax.text(cx, cy + h / 2 - 0.042 - LINE_H * (i + 1), s,
                ha='center', va='center', fontsize=10.2, color='#333333')
    return w, h


def arrow(ax, p1, p2, color='#444444', ls='-', lw=1.8):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=18,
                                 linewidth=lw, color=color, linestyle=ls))


fig, ax = plt.subplots(figsize=(15, 9.5), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

X = [0.17, 0.50, 0.83]          # three columns
Y1, Y2 = 0.76, 0.40             # two rows

# row 1, left -> right
w1, h1 = draw_box(ax, X[0], Y1, 'UE4 Engine',
                  ['FlexibleRoom, L-corner (W=150)', 'd=250 cm, 20 Hz control'], '#E8EEF7')
w2, h2 = draw_box(ax, X[1], Y1, 'Oracle Detection',
                  ['object mask -> bbox (cx,cy,a,h)', 'occlusion gate <20 / >50 px'], '#E8F3E8')
w3, h3 = draw_box(ax, X[2], Y1, 'State Estimation (KF)',
                  ['none | CV | CA | +tight-R', 'coasting on blind frames',
                   'optional: chi2 innovation gate'], '#FDF3DC')
# row 2, right -> left
w4, h4 = draw_box(ax, X[2], Y2, 'P-Controller',
                  ['yaw <- bbox center offset', 'speed <- pinhole range vs 250'], '#FBE9E7')
w5, h5 = draw_box(ax, X[1], Y2, 'Safety Layer (3 tiers)',
                  ['1. normal-velocity cap (primary):', '   v_wall <= (clearance-15)/0.6s',
                   '2. hard veto+slide (backstop)', '3. wall push (deadlock escape)',
                   'exact ESDF, never touches yaw'], '#EDE7F6')
w6, h6 = draw_box(ax, X[0], Y2, 'Unicycle + Inverse Mount',
                  ['camera point = the vehicle', 'pawn re-placed 30 cm behind'], '#E0F2F1')

arrow(ax, (X[0] + w1 / 2, Y1), (X[1] - w2 / 2, Y1))
arrow(ax, (X[1] + w2 / 2, Y1), (X[2] - w3 / 2, Y1))
arrow(ax, (X[2], Y1 - h3 / 2), (X[2], Y2 + h4 / 2))
arrow(ax, (X[2] - w4 / 2, Y2), (X[1] + w5 / 2, Y2))
arrow(ax, (X[1] - w5 / 2, Y2), (X[0] + w6 / 2, Y2))
arrow(ax, (X[0], Y2 + h6 / 2), (X[0], Y1 - h1 / 2), color='#888888')
ax.text(X[0] - 0.105, (Y1 + Y2) / 2, 'set poses,\nnext step\n(dt = 0.05 s)',
        ha='center', fontsize=9.5, color='#666666', style='italic')

# planner slot: beside the KF -> P-Controller junction
sw, sh = draw_box(ax, X[1], 0.10, 'PLANNER  --  intentionally EMPTY',
                  ['no trajectories, no waypoints, no recovery paths',
                   "(OA-VAT's diffusion planner would sit here)"],
                  '#FFFFFF', ec='#C0392B', ls='--', tc='#C0392B')
ax.add_patch(FancyArrowPatch((X[1] + sw / 2, 0.10), (X[2], Y2 - h4 / 2 - 0.01),
                             arrowstyle='-|>', mutation_scale=15, linewidth=1.4,
                             color='#C0392B', linestyle='--',
                             connectionstyle='arc3,rad=-0.25'))
ax.text(X[1], 0.10 - sh / 2 - 0.045,
        'steering is purely reactive, one P-step at a time -- so performance differences are attributable to the estimator alone',
        ha='center', fontsize=9.5, color='#C0392B')

# metrics note, plain text (no box), bottom-left
ax.text(X[0], 0.115, 'Metrics\n' + r'r = 1 - |$\rho$-250|/750 - |$\theta$|/45' + '\n'
        'AR / EL / SR + re-acquisition error',
        ha='center', fontsize=10, color='#555555')

ax.set_title('Project 2711 - Closed-Loop Evaluation Pipeline (planner-free reactive baseline)',
             fontsize=15, fontweight='bold', pad=16)
for ext in ['png', 'pdf']:
    for base in [r'D:\Aero2711esultsigures']:
        out = base + f'\\our_pipeline.{ext}'
        plt.savefig(out, dpi=170 if ext == 'png' else None,
                    facecolor='white', bbox_inches='tight')
print('saved all')
