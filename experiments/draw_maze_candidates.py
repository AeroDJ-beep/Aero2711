# -*- coding: utf-8 -*-
"""Top-view sketches of candidate maze layouts buildable from the engine's
15 movable blocks (5 cubes = walls, 5 cylinders + 5 cones = pillars).
Units cm. Output results/figures/maze_candidates.png"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib import font_manager
for f in ['Microsoft YaHei', 'SimHei']:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
HERE = os.path.dirname(os.path.abspath(__file__))

# (x0, x1, y0, y1) rectangles, 100 cm thick walls, corridor 150 cm wide
A = {  # serpentine: 3 long walls staggered -> 4 corners, alternating handedness
    'walls': [(-800, -700, -600, 600), (-550, -450, -600, 350), (-300, -200, -350, 600),
              (-50, 50, -600, 350), (200, 300, -350, 600)],
    'pillars': [],
    'title': 'A · 蛇形 serpentine（5 墙，4 个拐角，方向交替）',
    'path': [(-625, -550), (-625, 450), (-375, 450), (-375, -250), (-125, -250), (-125, 450), (125, 450), (125, -250)]}
B = {  # T-junction / H: target may choose a branch at the junction
    'walls': [(-800, -700, -600, 600), (-550, -450, -600, 100), (-550, -450, 250, 600),
              (-300, -200, -600, 600), (-800, 300, 600, 700)],
    'pillars': [(-375, -400), (-375, -150)],
    'title': 'B · T 字路口 T-junction（分叉，belief 需多模态）',
    'path': [(-625, -550), (-625, 175), (-375, 175), (-375, 500)]}
C = {  # open hall with pillars: occlusion by columns, no corridor prior
    'walls': [(-800, -700, -600, 600), (-800, 300, 600, 700), (-800, 300, -700, -600), (200, 300, -600, 600), (-350, -250, -100, 100)],
    'pillars': [(-550, -350), (-550, 300), (-100, -350), (-100, 300), (-300, 0), (50, -50), (-500, 0), (0, 350), (-200, -500), (100, 150)],
    'title': 'C · 柱厅 pillar hall（开放空间 + 圆柱遮挡）',
    'path': [(-650, -500), (-450, -200), (-250, 150), (-50, -250), (150, 300)]}

fig, axs = plt.subplots(1, 3, figsize=(17, 6.5))
for ax, L in zip(axs, [A, B, C]):
    for (x0, x1, y0, y1) in L['walls']:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#8f8f8f', edgecolor='k'))
    for (px, py) in L['pillars']:
        ax.add_patch(mp.Circle((px, py), 40, facecolor='#c9a26b', edgecolor='k'))
    xs, ys = zip(*L['path']); ax.plot(xs, ys, '--', color='#E07B00', lw=1.5, label='目标路线示意')
    ax.plot(xs[0], ys[0], 'o', color='#E07B00'); ax.plot(xs[-1], ys[-1], 's', color='#E07B00')
    ax.set_xlim(-900, 400); ax.set_ylim(-800, 800); ax.set_aspect('equal'); ax.grid(alpha=.2)
    ax.set_title(L['title'], fontsize=11)
    ax.text(-880, -780, f"墙 {len(L['walls'])}/5 · 柱 {len(L['pillars'])}/10", fontsize=9)
axs[0].legend(loc='upper right', fontsize=8)
fig.suptitle('候选迷宫布局（引擎只有 5 个可移动方块 + 10 个圆柱/圆锥；墙厚 100 cm，走廊宽 150 cm）', fontsize=12)
fig.tight_layout()
p = os.path.join(HERE, '..', 'results', 'figures', 'maze_candidates.png'); fig.savefig(p, dpi=120); print('saved', p)
