# -*- coding: utf-8 -*-
"""Record one maze episode and render
  1. first-person video: detection box (green), KF belief box (blue when the
     target is seen, dashed red when occluded), state text, range
  2. 2-D top-view video: maze walls, target / tracker trails, tracker field of
     view, KF belief projected on the floor (purple star), occluding corner
  3. a clean orthographic top-view image of the maze
usage: python uz_video_maze.py <maze> <mode> <seed> [profile] [--edgemeas ...]
outputs: results/videos/uz_<maze>_<mode>_s<seed>_{fp,2d}.mp4, results/figures/maze_<maze>_top.png"""
import os, sys, json, math, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R, occl_core as C
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.patches as mp

a = [x for x in sys.argv[1:] if not x.startswith('--')]
MAZE, MODE, SEED = a[0], a[1], int(a[2]); PROF = a[3] if len(a) > 3 else 'p2'
for f in R.FLAGS:
    R.FLAGS[f] = ('--' + f) in sys.argv
if R.FLAGS['dp']:
    R.dp_connect(); R.DP_LOG = []
if R.FLAGS['astar']:
    R.DP_LOG = []
if R.FLAGS['corridor']:
    R.CORR_LOG = []
if R.FLAGS['groundkf']:
    R.GK_LOG = []
PLANNER_NAME = 'DP PLANNER' if R.FLAGS['dp'] else ('A* PLANNER' if R.FLAGS['astar'] else 'PLANNER')
for x in sys.argv:                               # planner trigger / pure-pursuit lookahead (benchmark: --T=1 --L=30)
    if x.startswith('--T='):
        R.DP_TRIGGER = int(x.split('=')[1])
    if x.startswith('--L='):
        R.ASTAR_LOOK = float(x.split('=')[1])
    if x.startswith('--chase='):
        R.CHASE_D = float(x.split('=')[1])
    if x == '--spd':
        R.CORR_SPEEDS = (0.6, 1.0, 1.5)
    if x.startswith('--ring='):
        R.RING_R = float(x.split('=')[1])
    if x == '--flow':
        R.RING_FLOW = True
    if x == '--ib':
        R.CORR_INIT_BRANCH = True
    if x == '--stop':
        R.CORR_SPEEDS = (0.0, 0.6, 1.0, 1.5)
    if x == '--ff':
        R.FF_CTRL = True
    if x == '--ys':
        R.YAW_SMOOTH = True
    if x == '--bc':
        R.BREADCRUMB = True
    if x == '--kp':
        R.KP_HEADING = True
    if x == '--occ':
        R.CORR_OCC_INIT = True
    if x == '--kppos':
        R.KP_POS = True
    if x == '--kprange':
        R.KP_RANGE = True
    if x == '--rex':
        R.CORR_REEXPAND = True
if PROF == 'rw':                                  # random walk: UAV limits 1.5x the target (same as uz_sweep rw)
    R.C.V_CAM_MAX, R.C.W_CAM_MAX, R.DP_YAW_MAX = 150.0, 90.0, 90.0
if '--kpfirst' in sys.argv:
    R.KP_FIRST = True
if '--kinob' in sys.argv:
    R.KINO_BLIND = True
if '--kino' in sys.argv:
    R.ET_KINO = True
if '--et' in sys.argv:
    R.ET_PLANNER = True
if '--yi' in sys.argv:
    R.YAW_INERTIA = True
if '--cf' in sys.argv:
    R.CORR_CAME_FROM = True
if '--rch' in sys.argv:
    R.CORR_REACH = True
if '--anim' in sys.argv:                          # walking target; lit frames captured live (replay would freeze the animation)
    R.ANIM_TARGET = True; R.LIVE_LIT = '--live' in sys.argv
if '--single' in sys.argv:
    R.uz_maze.MazeTarget.MIN_AFTER = 600.0
tag = f'{MAZE}_{MODE}_s{SEED}_{PROF}' + ('_dp' if '--dp' in sys.argv else '') + ('_astar' if '--astar' in sys.argv else '') + \
      ('_ego' if '--egocomp' in sys.argv else '') + ('_wp' if '--worldpred' in sys.argv else '') + ('_corr' if '--corridor' in sys.argv else '') + ('_cv' if '--cleanvel' in sys.argv else '') + ('_gkf' if '--groundkf' in sys.argv else '') + ('_anim' if '--anim' in sys.argv else '') + ('_spd' if '--spd' in sys.argv else '') + (f'_chase{int(R.CHASE_D)}' if R.CHASE_D else '') + (f'_ring{int(R.RING_R)}' if R.RING_R else '') + ('_flow' if R.RING_FLOW else '') + ('_ib' if R.CORR_INIT_BRANCH else '') + ('_stop' if '--stop' in sys.argv else '') + ('_ff' if R.FF_CTRL else '') + ('_ys' if R.YAW_SMOOTH else '') + ('_bc' if R.BREADCRUMB else '') + ('_kp' if R.KP_HEADING else '') + ('_occ' if R.CORR_OCC_INIT else '') + ('_kppos' if R.KP_POS else '') + ('_kprange' if R.KP_RANGE else '') + ('_rex' if R.CORR_REEXPAND else '') + ('_cf' if R.CORR_CAME_FROM else '') + ('_rch' if R.CORR_REACH else '') + ('_yi' if R.YAW_INERTIA else '') + ('_et' if R.ET_PLANNER else '') + ('_kino' if R.ET_KINO else '') + ('_kinob' if R.KINO_BLIND else '') + ('_kpf' if R.KP_FIRST else '') + f'_T{R.DP_TRIGGER}'
VID = os.path.join(HERE, '..', 'results', 'videos'); FIG = os.path.join(HERE, '..', 'results', 'figures')
os.makedirs(VID, exist_ok=True)
FPS = 8

# ---------------- record ----------------
env = U.UZEnv('FlexibleRoom')
try:
    geo = R.setup_maze(env, MAZE)
    R.VIDEO = []
    log = []
    STEPS = int(next((x.split('=')[1] for x in sys.argv if x.startswith('--steps=')), 1200 if geo.info.get('gates') else 400))
    import time as _t; _t0 = _t.time()
    res = R.run_episode(env, MODE, SEED, PROF, steps=STEPS, log=log)
    res['wall_s'] = round(_t.time() - _t0, 1)   # closed loop at benchmark timing
    print('[ep]', json.dumps(res))
    L = np.array(log, float)
    np.save(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}_video.npy'), L)
    if R.DP_LOG is not None:
        json.dump(R.DP_LOG, open(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}_video_dpplans.json'), 'w'))
        print(f'DP plans: {len(R.DP_LOG)}, inference ms: {[round(p[1]) for p in R.DP_LOG]}')
    if R.GK_LOG is not None:
        json.dump(R.GK_LOG, open(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}_video_gkf.json'), 'w'))
    if R.CORR_LOG is not None:
        json.dump(R.CORR_LOG, open(os.path.join(HERE, '..', 'results', 'logs', f'uz_ep_{tag}_video_corr.json'), 'w'))
    n = min(len(L), len(R.VIDEO))
    DT_LOG = float(res['wall_s']) / max(res['EL'], 1) if 'wall_s' in res else 0.17
    SKEL_BONES = ['head', 'neck_01', 'spine_03', 'spine_01', 'pelvis', 'upperarm_l', 'lowerarm_l', 'hand_l',
                  'upperarm_r', 'lowerarm_r', 'hand_r', 'thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r']
    def skeleton_2d():
        """Person's bones projected into the current camera (true camera pose read back):
        {name: (u, v, visible)}; visible = in FOV and line of sight not blocked by a maze wall."""
        try:
            B = {b['bone_name']: b['transform'] for b in json.loads(env.c.request(f'vget /object/{env.tgt}/bones {",".join(SKEL_BONES)} world'))}
            cx_, cy_, cz_ = R.U._floats(env.c.request(f'vget /camera/{env.cam}/location'))[:3]
            cyaw = R.U._floats(env.c.request(f'vget /camera/{env.cam}/rotation'))[1]
        except Exception:
            return {}
        out = {}
        for n_, t in B.items():
            rel = (math.degrees(math.atan2(t['Y'] - cy_, t['X'] - cx_)) - cyaw + 180.0) % 360.0 - 180.0
            if abs(rel) > 60.0:
                continue
            dfw = math.hypot(t['X'] - cx_, t['Y'] - cy_) * math.cos(math.radians(rel))
            if dfw < 20.0:
                continue
            u = R.C.CX0 + R.C.FX * math.tan(math.radians(rel)); v = R.C.CY0 - R.C.FX * (t['Z'] - cz_) / dfw
            vis = abs(rel) <= 41.3 and not R.uz_maze.los_blocked(geo, (cx_, cy_), (t['X'], t['Y']))
            out[n_] = (u, v, vis)
        return out
    # replay: put target and tracker back at every logged pose and render the camera
    frames = [tuple(v) for v in R.VIDEO[:n]] if R.LIVE_LIT else []
    gyaw = math.degrees(math.atan2(L[min(3, n - 1), 4] - L[0, 4], L[min(3, n - 1), 3] - L[0, 3]))
    for k in (range(0) if R.LIVE_LIT else range(n)):
        gx, gy = L[k, 3], L[k, 4]
        j = min(k + 1, n - 1); dxy = (L[j, 3] - L[max(k - 1, 0), 3], L[j, 4] - L[max(k - 1, 0), 4])
        if math.hypot(*dxy) > 1.0:
            gyaw = math.degrees(math.atan2(dxy[1], dxy[0]))    # keep the last heading when the target stands
        if R.ANIM_TARGET:
            # keep the walk animation playing during the replay: drive the movement component at the
            # logged speed, then put the actor exactly on the logged pose before rendering
            v_k = math.hypot(*dxy) / max(2 * DT_LOG, 1e-3)
            env.c.request(f'vbp {env.tgt} set_move 0 {v_k:.1f}')
        R._teleport_target(env, gx, gy, gyaw)
        # in the loop, frame k is taken BEFORE the tracker moves at step k -> use its pose after step k-1
        kp = max(k - 1, 0)
        env.set_tracker(L[kp, 0], L[kp, 1], L[kp, 2])
        time.sleep(0.03)                                   # let the engine render the new poses
        sk_k = skeleton_2d()                               # bones read BEFORE the capture (the walker keeps moving)
        box_r, npx_r, _, img = env.frame(lit=True)         # re-detect on the REPLAYED image: box and picture match
        _, belief, box, det, faded = R.VIDEO[k]
        box = box_r if (box_r is not None and npx_r >= 30) else None
        frames.append((img, belief, box, det, faded, sk_k))
    print('replayed', n, 'frames')
finally:
    env.close()
rects = geo.rects

# ---------------- clean top view ----------------
ROUTE = None
if geo.info.get('gates'):
    _mt = R.uz_maze.MazeTarget(geo, np.random.default_rng(0), 'p2'); _mt.plan_gates(); ROUTE = _mt.P


def draw_walls(ax):
    for (x0, x1, y0, y1) in rects:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, color='#3a3a3a', zorder=2))
    inf = geo.info; W = inf['nx'] * inf['cell']; H = inf['ny'] * inf['cell']
    pad = 60 + (1.5 * inf['cell'] + 280 if inf.get('gates') else 0)
    ax.set_xlim(inf['x0'] - 60, inf['x0'] + W + 60); ax.set_ylim(inf['y0'] - pad, inf['y0'] + H + pad)
    if ROUTE is not None:
        ax.plot(ROUTE[:, 0], ROUTE[:, 1], ':', color='#999999', lw=1.2, zorder=1)
        ax.plot(*ROUTE[0], 'o', ms=14, color='#2E9E44', zorder=3); ax.text(ROUTE[0][0] + 40, ROUTE[0][1], 'START', color='#2E9E44', fontsize=10, fontweight='bold', va='center', zorder=3)
        ax.plot(*ROUTE[-1], 'o', ms=14, color='#C81E1E', zorder=3); ax.text(ROUTE[-1][0] - 40, ROUTE[-1][1], 'END', color='#C81E1E', fontsize=10, fontweight='bold', va='center', ha='right', zorder=3)
    ax.set_aspect('equal'); ax.grid(alpha=.15)

fig, ax = plt.subplots(figsize=(8, 8))
draw_walls(ax)
ax.set_title(f'{MAZE}: {geo.info["nx"]}x{geo.info["ny"]} cells of {geo.info["cell"]:.0f} cm, wall {geo.info.get("wall_t", 20):.0f} cm, {len(rects)} walls')
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
fig.savefig(os.path.join(FIG, f'maze_{MAZE}_top.png'), dpi=110); plt.close(fig)

# ---------------- first-person video ----------------
def dashed_rect(img, p0, p1, color, th=2, dash=10):
    (x0, y0), (x1, y1) = p0, p1
    for (a0, b0, a1, b1) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        L_ = int(math.hypot(a1 - a0, b1 - b0));
        for s in range(0, L_, 2 * dash):
            t0, t1 = s / max(L_, 1), min(s + dash, L_) / max(L_, 1)
            cv2.line(img, (int(a0 + (a1 - a0) * t0), int(b0 + (b1 - b0) * t0)), (int(a0 + (a1 - a0) * t1), int(b0 + (b1 - b0) * t1)), color, th)

SKEL_EDGES = [('head', 'neck_01'), ('neck_01', 'spine_03'), ('spine_03', 'spine_01'), ('spine_01', 'pelvis'),
              ('neck_01', 'upperarm_l'), ('upperarm_l', 'lowerarm_l'), ('lowerarm_l', 'hand_l'),
              ('neck_01', 'upperarm_r'), ('upperarm_r', 'lowerarm_r'), ('lowerarm_r', 'hand_r'),
              ('pelvis', 'thigh_l'), ('thigh_l', 'calf_l'), ('calf_l', 'foot_l'),
              ('pelvis', 'thigh_r'), ('thigh_r', 'calf_r'), ('calf_r', 'foot_r')]
TORSO_KP = set(R.KP_TORSO.split(','))                 # the joints the tracker actually uses (yellow)
p = os.path.join(VID, f'uz_{tag}_fp.mp4')
vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (640, 480))
for k in range(n):
    img, belief, box, det, faded = frames[k][:5]
    skel = frames[k][5] if len(frames[k]) > 5 else {}
    im = img.copy() if img is not None else np.zeros((480, 640, 3), np.uint8)
    for a_, b_ in SKEL_EDGES:                              # skeleton: visible bones white, behind a wall grey
        if a_ in skel and b_ in skel:
            pa, pb = skel[a_], skel[b_]
            cv2.line(im, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (235, 235, 235) if (pa[2] and pb[2]) else (120, 120, 120), 2 if (pa[2] and pb[2]) else 1, cv2.LINE_AA)
    for n_, (u_, v_, vis_) in skel.items():
        torso = n_ in TORSO_KP
        r_ = 6 if torso else 3; c_ = (0, 220, 255) if torso else (235, 235, 235)
        cv2.circle(im, (int(u_), int(v_)), r_, c_ if vis_ else (120, 120, 120), -1 if vis_ else 1, cv2.LINE_AA)
    if box is not None:
        cv2.rectangle(im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (60, 200, 60), 2)
    if belief is not None and np.all(np.isfinite(belief)):
        cx, cy, asp, h = belief; w = asp * h
        p0 = (int(cx - w / 2), int(cy - h / 2)); p1 = (int(cx + w / 2), int(cy + h / 2))
        if det:
            cv2.rectangle(im, p0, p1, (230, 140, 30), 2)
        else:
            dashed_rect(im, p0, p1, (40, 40, 230), 2)
        cv2.circle(im, (int(np.clip(cx, 0, 639)), int(np.clip(cy, 0, 479))), 4, (40, 40, 230) if not det else (230, 140, 30), -1)
    dp_on = L.shape[1] > 21 and L[k, 21] > 0.5
    state = (f'OCCLUDED - {PLANNER_NAME}' if dp_on else 'OCCLUDED') if not det else ('PARTIAL (edge rebuild)' if faded else 'VISIBLE')
    col = ((230, 0, 230) if dp_on else (40, 40, 230)) if not det else ((0, 170, 255) if faded else (60, 200, 60))
    cv2.rectangle(im, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(im, f'{MODE.upper()}  step {k:3d}  t={k / 7.5:4.1f}s  {state}', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(im, f'range {L[k, 11]:4.0f} cm', (500, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.rectangle(im, (0, 450), (640, 480), (0, 0, 0), -1)
    cv2.putText(im, 'green = detection  blue/red = KF belief  yellow = torso keypoints  grey = behind wall', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    vw.write(im)
vw.release(); print('saved', p)

# ---------------- 2-D top-view video ----------------
p2 = os.path.join(VID, f'uz_{tag}_2d.mp4')
fig, ax = plt.subplots(figsize=(8, 8), dpi=90)
fig.canvas.draw(); Wpx, Hpx = fig.canvas.get_width_height()
vw = cv2.VideoWriter(p2, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (Wpx, Hpx))
corner = None
for k in range(n):
    ax.clear(); draw_walls(ax)
    tx, ty, yaw, gx, gy, det = L[k, 0], L[k, 1], L[k, 2], L[k, 3], L[k, 4], L[k, 5] > 0.5
    ax.plot(L[:k + 1, 3], L[:k + 1, 4], '-', color='#E07B00', lw=1.6, zorder=4)
    ax.plot(L[:k + 1, 0], L[:k + 1, 1], '-', color='#1668C7', lw=1.6, zorder=4)
    # field of view wedge (82.6 deg, 6 m)
    fov = 41.3
    ax.add_patch(mp.Wedge((tx, ty), 600, yaw - fov, yaw + fov, color='#1668C7', alpha=0.08, zorder=3))
    ax.plot([tx, gx], [ty, gy], '--', color='#2E9E44' if det else '#C81E1E', lw=1, zorder=5)
    # KF belief on the floor + trail of the last 30 steps
    ph = []
    for j in range(max(0, k - 30), k + 1):
        b = frames[j][1]
        if b is not None and np.all(np.isfinite(b)) and b[3] > 1:
            jp = max(j - 1, 0)                     # belief j was estimated from the frame taken at pose j-1
            ph.append(C.back_project(b[0], b[3], L[jp, 0], L[jp, 1], L[jp, 2]))
    if ph:
        pa = np.array(ph); ax.plot(pa[:, 0], pa[:, 1], '-', color='#8B2FC9', lw=1.2, alpha=0.6, zorder=6)
        ax.plot(pa[-1, 0], pa[-1, 1], '*', ms=15, color='#8B2FC9', mec='k', zorder=7)
    # latest DP plan (if the planner is active at this step): planned waypoints in world
    if R.DP_LOG and L.shape[1] > 21 and L[k, 21] > 0.5:
        plans = [p for p in R.DP_LOG if p[0] <= k]
        if plans:
            wp = np.atleast_2d(np.array(plans[-1][3], float)) if len(plans[-1][3]) else np.zeros((0, 2))
            ax.plot(wp[:, 0], wp[:, 1], '-o', color='#D000D0', ms=4, lw=1.5, zorder=7, label='DP plan')
    ax.plot(gx, gy, 'o', ms=11, color='#E07B00', mec='k', zorder=8)
    ax.plot(tx, ty, marker=(3, 0, yaw - 90), ms=13, color='#1668C7', mec='k', zorder=8)
    ax.set_title(f'{MODE.upper()} seed {SEED}   step {k}   ' + ('VISIBLE' if det else 'OCCLUDED'),
                 color='#222222' if det else '#C81E1E', fontsize=12, fontweight='bold')
    ax.text(0.01, 0.01, 'orange = target   blue = tracker (wedge = camera FOV)   purple star = KF belief on the floor',
            transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(facecolor='white', alpha=0.8, lw=0))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    vw.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
vw.release(); print('saved', p2)
