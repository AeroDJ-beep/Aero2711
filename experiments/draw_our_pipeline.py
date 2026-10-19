# -*- coding: utf-8 -*-
"""Our closed-loop experimental pipeline (planner-free), auto-width boxes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CHAR_W = 0.0058     # width per char (title-size), tuned for fontsize 10.5
LINE_H = 0.030

BOXES = [  # (title, lines, facecolor)
    ('UE4 Engine', ['FlexibleRoom, L-corner track', 'W=150 cm, d=250 cm, 20 Hz'], '#E8EEF7'),
    ('Oracle Detection', ['object mask -> bbox (cx,cy,a,h)', 'occlusion gate: <20 px / >50 px', '(hysteresis)'], '#E8F3E8'),
    ('State Estimation (KF)', ['one of: none | CV | CA | +tight-R', 'predict every step; update if seen', 'coasting on blind frames'], '#FDF3DC'),
    ('P-Controller', ['yaw <- bbox center offset', 'speed <- pinhole range vs 250', 'clip: v in [0,99], |w|<=30 deg/s'], '#FBE9E7'),
    ('Safety Filter', ['analytic ESDF on vector map', 'lookahead veto + slide + push', 'translation only - NEVER yaw'], '#EDE7F6'),
    ('Unicycle + Inverse Mount', ['camera point = the vehicle', 'pawn re-placed 30 cm behind', '(zero rotation radius)'], '#E0F2F1'),
]


def box_size(title, lines):
    w = max([len(title)] + [len(s) for s in lines]) * CHAR_W + 0.024
    h = LINE_H * (len(lines) + 1) + 0.030
    return w, h


def draw_box(ax, cx, cy, title, lines, fc, ec='#555555', ls='-', title_c='#111111'):
    w, h = box_size(title, lines)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.008', linewidth=1.4,
                                edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(cx, cy + h / 2 - 0.030, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=title_c)
    for i, s in enumerate(lines):
        ax.text(cx, cy + h / 2 - 0.030 - LINE_H * (i + 1), s,
                ha='center', va='center', fontsize=9.0, color='#333333')
    return w, h


fig, ax = plt.subplots(figsize=(19.5, 6.4), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# --- main chain, auto-spaced (rescale to fit if needed) ---
gap = 0.024
sizes = [box_size(t, ls_) for t, ls_, _ in BOXES]
total_w = sum(w for w, _ in sizes) + gap * (len(BOXES) - 1)
if total_w > 0.97:
    f = 0.97 / total_w
    CHAR_W *= f
    gap *= f
    sizes = [box_size(t, ls_) for t, ls_, _ in BOXES]
    total_w = sum(w for w, _ in sizes) + gap * (len(BOXES) - 1)
x = (1 - total_w) / 2
Y = 0.60
centers = []
for (title, lines, fc), (w, h) in zip(BOXES, sizes):
    cx = x + w / 2
    draw_box(ax, cx, Y, title, lines, fc)
    centers.append((cx, w, h))
    x += w + gap

for (c1, w1, _), (c2, w2, _) in zip(centers[:-1], centers[1:]):
    ax.add_patch(FancyArrowPatch((c1 + w1 / 2 + 0.002, Y), (c2 - w2 / 2 - 0.002, Y),
                                 arrowstyle='-|>', mutation_scale=16,
                                 linewidth=1.6, color='#444444'))

# feedback arrow: last box -> first box (over the top)
cL, wL, hL = centers[-1]
c0, w0, h0 = centers[0]
top = Y + max(h for _, h in sizes) / 2 + 0.10
ax.add_patch(FancyArrowPatch((cL, Y + hL / 2 + 0.004), (cL, top), arrowstyle='-',
                             linewidth=1.5, color='#888888'))
ax.add_patch(FancyArrowPatch((cL, top), (c0, top), arrowstyle='-',
                             linewidth=1.5, color='#888888'))
ax.add_patch(FancyArrowPatch((c0, top), (c0, Y + h0 / 2 + 0.004), arrowstyle='-|>',
                             mutation_scale=15, linewidth=1.5, color='#888888'))
ax.text((cL + c0) / 2, top + 0.028, 'set poses, next control step (dt = 0.05 s)',
        ha='center', fontsize=9.3, color='#666666', style='italic')

# --- the intentionally-empty planner slot ---
kf_c, kf_w, kf_h = centers[2]
pid_c, pid_w, pid_h = centers[3]
slot_cx = (kf_c + pid_c) / 2
slot_cy = Y - 0.30
sw, sh = draw_box(ax, slot_cx, slot_cy, 'PLANNER  (intentionally EMPTY)',
                  ['no trajectory generation, no waypoints,', 'no recovery paths - steering is purely',
                   'reactive, one P-step at a time'],
                  '#FFFFFF', ec='#C0392B', ls='--', title_c='#C0392B')
ax.add_patch(FancyArrowPatch((slot_cx, slot_cy + sh / 2 + 0.004),
                             ((kf_c + pid_c) / 2, Y - max(kf_h, pid_h) / 2 - 0.004),
                             arrowstyle='-|>', mutation_scale=13, linewidth=1.3,
                             color='#C0392B', linestyle='--'))
ax.text(slot_cx, slot_cy - sh / 2 - 0.075,
        "OA-VAT's diffusion planner would sit here; our baseline deliberately omits it\n"
        'so that survival/error differences are attributable to the state estimator alone.',
        ha='center', fontsize=9.0, color='#C0392B')

# --- metrics side box ---
met_cx = centers[1][0]
draw_box(ax, met_cx, Y - 0.30, 'Metrics (per step / episode)',
         ['r = 1 - |rho-250|/750 - |theta|/45', 'AR, EL, SR (500 steps, 50-step kill)',
          're-acquisition error (px)'], '#F5F5F5')

# --- map note under safety filter ---
sf_c, sf_w, sf_h = centers[4]
ax.text(sf_c, Y - sf_h / 2 - 0.055,
        'only consumer of the map:\nper-step veto (5 rectangles, exact ESDF)',
        ha='center', fontsize=8.8, color='#5E35B1')

ax.set_title('Project 2711 - Closed-Loop Evaluation Pipeline (planner-free reactive baseline)',
             fontsize=14.5, fontweight='bold', pad=14)
for ext in ['png', 'pdf']:
    for base in [r'D:\Aero2711esultsigures']:
        out = base + f'\\our_pipeline.{ext}'
        plt.savefig(out, dpi=170 if ext == 'png' else None,
                    facecolor='white', bbox_inches='tight')
        print('saved:', out)
