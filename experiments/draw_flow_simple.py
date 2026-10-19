# -*- coding: utf-8 -*-
"""极简中文流程图：一个控制步(0.05s)的完整走向。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(11, 13), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')


def box(cx, cy, w, h, title, sub, fc, ec='#555', ls='-'):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.008', lw=1.6,
                                edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(cx, cy + (0.013 if sub else 0), title, ha='center', va='center',
            fontsize=13, fontweight='bold')
    if sub:
        ax.text(cx, cy - 0.017, sub, ha='center', va='center',
                fontsize=10.5, color='#444')


def arrow(p1, p2, label='', color='#444'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=20,
                                 lw=1.8, color=color))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.012, my, label, fontsize=10.5, color=color,
                ha='left', va='center')


X = 0.36
XR = 0.78
H = 0.062

box(X, 0.955, 0.42, H, '① 引擎拍一张照片', '目标的掩膜 → 检测框(位置/宽/高)', '#E8EEF7')
arrow((X, 0.955 - H / 2), (X, 0.873 + H / 2))

box(X, 0.873, 0.50, H, '② 框健康吗？(像素数骤降=被墙啃)',
    '健康→直接用 | 被啃→edgemeas 用干净边缘修框 | 全没了→本步无测量', '#E8F3E8')
arrow((X, 0.873 - H / 2), (X, 0.791 + H / 2))

box(X, 0.791, 0.46, H, '③ 卡尔曼滤波', '预测一步；有测量就修正 → 得到"信念"(他在哪)', '#FDF3DC')
arrow((X, 0.791 - H / 2), (X, 0.709 + H / 2))

box(X, 0.709, 0.50, H, '④ mapclamp：信念体检',
    '信念跑出走廊? → 按回走廊里(人只能在走廊跑)', '#FDF3DC', ec='#8B2FC9')
arrow((X, 0.709 - H / 2), (X, 0.615 + H / 2))

box(X, 0.615, 0.34, 0.070, '⑤ 现在看得见他吗？', '', '#FFFFFF', ec='#C0392B')
# visible branch (left down)
arrow((X - 0.10, 0.615 - 0.035), (X - 0.10, 0.520 + H / 2), '看得见', '#2E7D32')
box(X - 0.10, 0.520, 0.30, H, '正常追', '转向:瞄框中心\n速度:按距离误差', '#EAF5EA', ec='#2E7D32')
# blind branch (right down)
arrow((X + 0.13, 0.615 - 0.035), (XR - 0.155, 0.520 + H / 2), '看不见', '#C0392B')
box(XR, 0.520, 0.36, H, '盲飞策略', '转向:瞄拐角口(edgeaim)\n速度:固定全速(blindspeed)', '#FBE9E7', ec='#C0392B')
arrow((X - 0.10, 0.520 - H / 2), (X, 0.425 + H / 2))
arrow((XR, 0.520 - H / 2), (X + 0.08, 0.425 + H / 2))

box(X, 0.425, 0.50, H, '⑥ vcap：贴墙限速',
    '速度里"冲向墙"的那部分,离墙越近砍得越多;沿墙的不动', '#EDE7F6')
arrow((X, 0.425 - H / 2), (X, 0.343 + H / 2))

box(X, 0.343, 0.46, H, '⑦ 保险丝(safe_move)', '落点离墙<15cm? 不许走→沿墙滑/被推开', '#EDE7F6')
arrow((X, 0.343 - H / 2), (X, 0.261 + H / 2))

box(X, 0.261, 0.46, H, '⑧ 移动相机', '写入引擎(人偶摆到相机正后方30cm)', '#E0F2F1')

# loop back
ax.add_patch(FancyArrowPatch((X - 0.23, 0.261), (0.055, 0.261), arrowstyle='-', lw=1.6, color='#888'))
ax.add_patch(FancyArrowPatch((0.055, 0.261), (0.055, 0.955), arrowstyle='-', lw=1.6, color='#888'))
ax.add_patch(FancyArrowPatch((0.055, 0.955), (X - 0.21, 0.955), arrowstyle='-|>',
                             mutation_scale=20, lw=1.6, color='#888'))
ax.text(0.048, 0.60, '0.05秒后\n下一步', fontsize=11, color='#666', ha='right', style='italic')

# side note
ax.text(XR, 0.34,
        '设计铁律\n看不见时,转向和速度\n都不听"信念"的\n(信念只负责记账:\n重现时刻和它对答案)',
        fontsize=11, color='#C0392B', ha='center',
        bbox=dict(boxstyle='round', facecolor='#FFF5F5', edgecolor='#C0392B'))

ax.set_title('一个控制步的完整流程（每 0.05 秒转一圈）', fontsize=16, fontweight='bold', pad=14)
for base in [r'D:\Aero2711esultsigures']:
    plt.savefig(base + r'\flow_simple.png', facecolor='white', bbox_inches='tight')
print('saved')
