import json, collections, shutil, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import datetime as dt
import matplotlib.dates as mdates

OUT = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(OUT, exist_ok=True)
M = r'D:\Aero2711\results\metrics'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

# ---------- Fig: L-track top view ----------
WALLS = [(-685, -585, -530, 70), (-685, -585, 70, 395), (-435, -335, -530, 145),
         (-435, 40, 145, 245), (-710, -35, 395, 495)]
CORRIDOR = [(-565, -455, -680, 375), (-565, 880, 265, 375)]
fig, ax = plt.subplots(figsize=(7.2, 6.0))
ax.add_patch(Rectangle((-585, -700), 150, 1095, fc='#fbf3dc', ec='none'))
ax.add_patch(Rectangle((-585, 245), 700, 150, fc='#fbf3dc', ec='none'))
for (x0, x1, y0, y1) in WALLS:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='#9a9a9a', ec='k', lw=0.8))
for i, (x0, x1, y0, y1) in enumerate(CORRIDOR):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='none', ec='#2a7f3e', ls='--', lw=1.0,
                           label='reachable strips (map clamp)' if i == 0 else None))
# sample route
rng = np.random.default_rng(2005)
m = 40.0
sx = lambda: rng.uniform(-585 + m, -435 - m)
ey = lambda: rng.uniform(245 + m, 395 - m)
pts = [(sx(), -450.0)]
for y in np.sort(rng.uniform(-350, 120, rng.integers(2, 5))):
    pts.append((sx(), y))
c = rng.uniform(40, 60)
pts.append((-435.0 - c, 245.0 + c))
for x in np.sort(rng.uniform(-300, -50, rng.integers(2, 4))):
    pts.append((x, ey()))
pts.append((60.0, ey()))
pts = np.array(pts)
ax.plot(pts[:, 0], pts[:, 1], '-', color='#d35400', lw=2, label='example target route (seed 2005)')
ax.plot(*pts[0], 'o', color='#d35400', ms=7)
ax.plot(-510, -700, '^', color='#1f5f8b', ms=11, label='tracker start (250 cm behind)')
ax.plot(-435, 245, '*', color='crimson', ms=14, label='occluding corner apex (−435, 245)')
# sight line illustrating occlusion
cam = np.array([-500, 60.0]); tgt = np.array([-250, 300.0])
ax.plot([cam[0], tgt[0]], [cam[1], tgt[1]], ':', color='crimson', lw=1.4)
ax.plot(*cam, '^', color='#1f5f8b', ms=8, alpha=0.6)
ax.plot(*tgt, 'o', color='#d35400', ms=8, alpha=0.6)
ax.text(-240, 330, 'target hidden\nbehind corner', fontsize=8.5, color='crimson')
ax.annotate('', xy=(-585, -600), xytext=(-435, -600), arrowprops=dict(arrowstyle='<->'))
ax.text(-510, -640, 'W = 150 cm', ha='center', fontsize=9)
ax.text(-160, 520, 'walls 350 cm high, 100 cm thick', fontsize=8.5, color='#444')
ax.set_xlim(-850, 250); ax.set_ylim(-760, 620); ax.set_aspect('equal')
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
ax.set_title('L-corner track in FlexibleRoom (engine-verified wall bounds)')
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_track.png'), dpi=200); plt.close(fig)

# ---------- Fig: architecture ----------
fig, ax = plt.subplots(figsize=(8.0, 6.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
def box(x, y, w, h, title, body, fc, ec='#444', lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='top', fontweight='bold', fontsize=9.5)
    ax.text(x + w / 2, y + h - 7.0, body, ha='center', va='top', fontsize=8.2, linespacing=1.35)
def arr(a, b, **k):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12, lw=1.3, color='#333', **k))
box(2, 84, 30, 13, 'UE4 simulator', 'FlexibleRoom L-corner\nobject mask, 640x480', '#e6edf7')
box(36, 84, 30, 13, 'Detection', 'mask -> bbox (cx, cy, a, h)\noracle now; real detector later', '#e7f2e5')
box(70, 80, 28, 17, 'Measurement repair', '[edgemeas] pixel count < 50%\nof recent peak -> rebuild cx\nfrom un-clipped edge using\nmap-projected occluder column', '#fdf1dc', ec='#8e44ad', lw=1.8)
box(70, 58, 28, 16, 'Kalman filter', 'none | CV (8-D) | CA (10-D)\npredict every step,\nupdate when measured', '#fdf1dc')
box(70, 36, 28, 16, 'Map clamp', '[mapclamp] blind frames only:\nback-project (cx,h) -> world,\nproject into corridor strips,\nre-project; P untouched', '#fdf1dc', ec='#8e44ad', lw=1.8)
box(36, 58, 30, 16, 'Steering (visible)', 'yaw  <- bbox centre offset\nspeed <- pinhole range\n          vs 250 cm set-point', '#e8f5e9')
box(36, 36, 30, 16, 'Steering (blind / fading)', '[edgeaim] yaw -> occluder edge\n[blindspeed] speed = 99 cm/s\n(belief is NOT consumed)', '#fbe9e7', ec='#c0392b', lw=1.8)
box(2, 36, 30, 22, 'Safety layer', '[vcap] cap wall-approach\nvelocity: v_in <= (d - m)/0.6 s\nhard veto + axis slide\nwall-normal push-off\nnever changes yaw', '#efe8f6')
box(2, 12, 30, 16, 'Vehicle', 'unicycle, v <= 99 cm/s,\n|w| <= 30 deg/s; camera point\n= vehicle (zero turn radius)', '#e0f2f1')
box(36, 8, 62, 18, 'Evaluation (per step, 500 steps, 20 Hz)', 'score r = 1 - |rho - 250|/750 - |theta|/45  ->  AR, EL, SR\nre-acquisition error: predicted vs true bbox centre\nat the first re-appearance step (px)', '#f5f5f5')
arr((32, 90.5), (35.6, 90.5)); arr((66, 90.5), (69.6, 90.5))
arr((84, 79.4), (84, 74.6)); arr((84, 57.4), (84, 52.6))
arr((69.4, 66), (66.6, 66)); arr((69.4, 44), (66.6, 44))
arr((35.4, 62), (32.6, 52)); arr((35.4, 44), (32.6, 46))
arr((17, 35.4), (17, 28.6))
ax.annotate('', xy=(2.5, 90), xytext=(2.5, 20), arrowprops=dict(arrowstyle='-|>', color='#888', lw=1.2, connectionstyle='arc3,rad=-0.35'))
ax.text(-2.5, 75, 'next frame (dt = 0.05 s)', fontsize=7.5, color='#666', rotation=90, va='center', ha='center')
ax.text(50, 1.5, 'Purple = estimator-side additions;  red = behaviour decoupled from the belief during occlusion',
        ha='center', fontsize=8, color='#555')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_arch.png'), dpi=200); plt.close(fig)

# ---------- Fig: progression ----------
stages = ['Inherited\nnoise R', 'Calibrated R\n(x0.1)', '+ velocity-cap\nsafety', '+ measurement\nrepair + edge aim', '+ blind speed\n+ map clamp']
cv = [114.1, 101.3, 89.7, 28.7, 24.1]
ca = [113.4, 98.5, 86.5, 34.7, 41.2]
x = np.arange(len(stages)); w = 0.36
fig, ax = plt.subplots(figsize=(7.6, 3.9))
b1 = ax.bar(x - w / 2, cv, w, label='CV Kalman', color='#2e75b6')
b2 = ax.bar(x + w / 2, ca, w, label='CA Kalman', color='#c0392b')
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 3, f'{r.get_height():.0f}', ha='center', fontsize=8)
ax.axhline(93.1, color='#777', ls='--', lw=1)
ax.text(4.45, 96, 'no predictor, full system: 93 px', ha='right', fontsize=8, color='#555')
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8.5)
ax.set_ylabel('mean re-acquisition error (px)')
ax.set_title('Random target motion, 40 paired seeds (33-34 with occlusion); SR = 1.00 throughout', fontsize=10)
sec = ax.secondary_yaxis('right', functions=(lambda p: p * 0.69, lambda c: c / 0.69))
sec.set_ylabel('approx. cm at 250 cm range')
ax.legend(fontsize=8.5); ax.set_ylim(0, 140); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_progress.png'), dpi=200); plt.close(fig)

# ---------- Fig: final 3-mode comparison ----------
d = json.load(open(os.path.join(M, 'metrics_fullarch_none_cv_ca_2026-09-19.json')))
g = collections.defaultdict(dict)
for e in d:
    if e['reacq'] is not None:
        g[(e['profile'], e['mode'])][e['seed']] = e['reacq']
fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
cols = {'none': '#8c8c8c', 'cv': '#2e75b6', 'ca': '#c0392b'}
for ax, prof, ttl in zip(axs, ['p2', 'p2a'], ['P2: corner sprint (80 cm/s² to 99 cm/s)', 'P2a: random corner accel. in [−80, 80] cm/s²']):
    modes = ['none', 'cv', 'ca']
    seeds = sorted(set.intersection(*[set(g[(prof, mm)]) for mm in modes]))
    for s in seeds:
        ax.plot(range(3), [g[(prof, mm)][s] for mm in modes], '-', color='#ccc', lw=0.8, zorder=1)
    for i, mm in enumerate(modes):
        v = list(g[(prof, mm)].values())
        ax.scatter([i] * len(v), v, color=cols[mm], s=16, zorder=2)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color='k', lw=2, zorder=3)
        ax.text(i, 262, f'mean {np.mean(v):.1f}\n(n={len(v)})', ha='center', fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(['no predictor', 'CV', 'CA'])
    ax.set_title(ttl, fontsize=9); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 290)
axs[0].set_ylabel('re-acquisition error (px)')
fig.suptitle('Full architecture: per-seed re-acquisition error (grey lines join the same seed)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_final.png'), dpi=200); plt.close(fig)

# ---------- Fig: Gantt ----------
D = lambda s: dt.date.fromisoformat(s)
tasks = [  # (label, start, end, status)
    ('Reproduce OA-VAT (3 modules, Table 1)', '2026-07-20', '2026-08-01', 'done'),
    ('Literature survey', '2026-08-01', '2026-10-09', 'ongoing'),
    ('Pilot: accelerating / corner targets', '2026-08-05', '2026-08-12', 'done'),
    ('Environment v2 (camera, L-track, occlusion geometry)', '2026-08-15', '2026-08-25', 'done'),
    ('Closed-loop harness; CV vs CA test', '2026-08-28', '2026-09-06', 'done'),
    ('Safety layer (margin calib., velocity cap)', '2026-09-04', '2026-09-10', 'done'),
    ('MC-KF measurement repair + decoupled control', '2026-09-10', '2026-09-16', 'done'),
    ('MC-KF reachable-set projection + evaluation', '2026-09-14', '2026-09-20', 'done'),
    ('Progress report', '2026-09-24', '2026-10-02', 'ongoing'),
    ('Generalise corridor map (T-junction, S-bend)', '2026-10-05', '2026-10-23', 'planned'),
    ('Braking analysis, IMM baseline, 3 seed sets', '2026-10-12', '2026-10-23', 'planned'),
    ('Real detector in the loop (YOLOE/ORTrack) + gate', '2026-10-19', '2026-11-06', 'planned'),
    ('Stretch: learned predictor / Tello indoor test', '2026-10-26', '2026-11-13', 'stretch'),
    ('Final report + presentation', '2026-11-02', '2026-11-20', 'planned'),
]
col = {'done': '#2e7d32', 'ongoing': '#f39c12', 'planned': '#2e75b6', 'stretch': '#b0bec5'}
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for i, (lab, s, e, st) in enumerate(tasks):
    ax.barh(i, (D(e) - D(s)).days, left=D(s), color=col[st], edgecolor='k', lw=0.4,
            hatch='//' if st == 'stretch' else None)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks], fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.setp(ax.get_xticklabels(), fontsize=8, rotation=30)
ax.axvline(D('2026-09-27'), color='crimson', ls='--', lw=1.2)
ax.text(D('2026-09-28'), 8.4, 'today (27 Sep)', color='crimson', fontsize=8)
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=col[k], ec='k', hatch='//' if k == 'stretch' else None, label=k) for k in col],
          loc='lower left', fontsize=8, ncol=4)
ax.set_title('Revised project timeline (2026)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_gantt.png'), dpi=200); plt.close(fig)

shutil.copy(r'D:\Aero2711\results\figures\occlusion_window.png', os.path.join(OUT, 'fig_occlusion.png'))
print('ok')
