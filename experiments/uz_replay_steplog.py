# -*- coding: utf-8 -*-
"""Replay a batch-run episode from its per-step log (results/logs/sweep_steps/*.npy + _corr.json)
into a phone video: left = rendered camera view (skeleton, detection box, predicted position as a
red box), right = top view with every corridor hypothesis (dot size = weight, star = best).
usage: python uz_replay_steplog.py <name, e.g. rwfix3_fix_cf_reach_s5> [k0] [k1] [height=360]
outputs: results/videos/<name>_k<k0>-<k1>_phone.mp4"""
import os, sys, json, math, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R, uz_maze as Z, occl_core as C
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.patches as mp

NAME = sys.argv[1]
LD = os.path.join(HERE, '..', 'results', 'logs', 'sweep_steps')
L = np.load(os.path.join(LD, NAME + '.npy'))
CORR = {e[0]: e for e in json.load(open(os.path.join(LD, NAME + '_corr.json')))}
k0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
k1 = min(int(sys.argv[3]) if len(sys.argv) > 3 else len(L) - 1, len(L) - 1)
MAZE = 'fr10x10l35t24g_s0'
SKEL = ['head', 'neck_01', 'spine_03', 'spine_01', 'pelvis', 'upperarm_l', 'lowerarm_l', 'hand_l', 'upperarm_r', 'lowerarm_r', 'hand_r',
        'thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r']
EDGES = [('head', 'neck_01'), ('neck_01', 'spine_03'), ('spine_03', 'spine_01'), ('spine_01', 'pelvis'),
         ('neck_01', 'upperarm_l'), ('upperarm_l', 'lowerarm_l'), ('lowerarm_l', 'hand_l'),
         ('neck_01', 'upperarm_r'), ('upperarm_r', 'lowerarm_r'), ('lowerarm_r', 'hand_r'),
         ('pelvis', 'thigh_l'), ('thigh_l', 'calf_l'), ('calf_l', 'foot_l'),
         ('pelvis', 'thigh_r'), ('thigh_r', 'calf_r'), ('calf_r', 'foot_r')]
TORSO = set(R.KP_TORSO.split(','))


def cam_pose(env):
    cx_, cy_, cz_ = U._floats(env.c.request(f'vget /camera/{env.cam}/location'))[:3]
    return cx_, cy_, cz_, U._floats(env.c.request(f'vget /camera/{env.cam}/rotation'))[1]


def project(cp, geo, X, Y, Z_):
    """World point -> (u, v, visible, forward distance) in the tracker camera pose cp."""
    cx_, cy_, cz_, cyaw = cp
    rel = (math.degrees(math.atan2(Y - cy_, X - cx_)) - cyaw + 180.0) % 360.0 - 180.0
    dfw = math.hypot(X - cx_, Y - cy_) * math.cos(math.radians(rel))
    if abs(rel) > 60.0 or dfw < 20.0:
        return None
    return (C.CX0 + C.FX * math.tan(math.radians(rel)), C.CY0 - C.FX * (Z_ - cz_) / dfw,
            abs(rel) <= 41.3 and not Z.los_blocked(geo, (cx_, cy_), (X, Y)), dfw)


env = U.UZEnv('FlexibleRoom'); env.verbose = False
frames = []
try:
    geo = R.setup_maze(env, MAZE)
    gyaw = 0.0
    for k in range(k0, k1 + 1):
        gx, gy = L[k, 3], L[k, 4]
        dxy = (L[min(k + 1, len(L) - 1), 3] - L[k - 1, 3], L[min(k + 1, len(L) - 1), 4] - L[k - 1, 4])
        if math.hypot(*dxy) > 1.0:
            gyaw = math.degrees(math.atan2(dxy[1], dxy[0]))
        env.c.request(f'vbp {env.tgt} set_move 0 {math.hypot(*dxy) / max(2 * L[k, 10], 1e-3):.1f}')   # keep the walk animation
        R._teleport_target(env, gx, gy, gyaw)
        env.set_tracker(L[k - 1, 0], L[k - 1, 1], L[k - 1, 2]); time.sleep(0.03)
        # bones and camera pose read ONCE, right before the image: the walker keeps moving (walk animation),
        # so anything read after the capture is drawn where she has already walked to
        cp = cam_pose(env)
        try:
            B = {b['bone_name']: b['transform'] for b in json.loads(env.c.request(f'vget /object/{env.tgt}/bones {",".join(SKEL)} world'))}
        except Exception:
            B = {}
        box, npx, _, img = env.frame(lit=True)
        skel = {n: project(cp, geo, t['X'], t['Y'], t['Z']) for n, t in B.items()}
        e = CORR.get(k); pred = None
        if e is not None:
            hb = e[1][e[2]]; pr = project(cp, geo, hb[0], hb[1], env.z)       # predicted position at the person's mid-height
            if pr is not None:
                pred = pr
        frames.append((img, box if (box is not None and npx >= 30) else None, skel, pred, cp))
    off = [frames[i][2]['spine_03'][0] - (frames[i][1][0] + frames[i][1][2] / 2) for i in range(len(frames))
           if frames[i][1] is not None and frames[i][2].get('spine_03') and frames[i][2]['spine_03'][2]]
    ZMID = env.z
    CAM_Z = U._floats(env.c.request(f'vget /camera/{env.cam}/location'))[2]
    Z_FLOOR = env.z_floor - 0.5 * C.H_PERSON * 1.05            # actor origin sits near mid-body: floor ~ origin - half height
    print('rendered', len(frames), '| chest column - box centre (px): median', round(float(np.median(off)), 1) if off else None,
          'median |.|', round(float(np.median(np.abs(off))), 1) if off else None, 'n', len(off))
finally:
    env.close()

# ---------------- compose ----------------
rects = geo.rects; inf = geo.info
fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
OVERLAY = len(sys.argv) > 5 and sys.argv[5] == 'overlay'      # draw the planner's future points and the A* viewing spot
if OVERLAY:
    import uz_astar
    PL = uz_astar.MazePlanner(geo, margin=R.MAZE_MARGIN)
out = os.path.join(HERE, '..', 'results', 'videos', f'{NAME}_k{k0}-{k1}{"_overlay" if OVERLAY else ""}_phone.mp4'); vw = None
for i, k in enumerate(range(k0, k1 + 1)):
    img, box, skel, pred, cp = frames[i][:5]
    im = img.copy()
    seen = L[k, 5] > 0
    zs = None; goal = None
    if OVERLAY:
        # recomputed with the planner's own code: visible -> KF floor point pushed along the corridor at the KF speed;
        # hidden -> best corridor guess pushed along its direction (speed multiple assumed 1: not in the log) + A* viewing spot
        tp, typ, yp = L[k - 1, 0], L[k - 1, 1], L[k - 1, 2]
        wv_ = (L[k, 22], L[k, 23]) if L.shape[1] > 23 else (0.0, 0.0); sp_ = math.hypot(*wv_)
        e_ = CORR.get(k)
        if seen and L[k, 7] > 5:
            ph_ = C.back_project(L[k, 6], L[k, 7], tp, typ, yp)
            p0_ = tuple(ph_) if geo.is_free(*ph_) else geo.nearest_free(*ph_)[:2]
            pr_ = Z.CorridorPredictor(geo, p0_, wv_ if sp_ > 1 else (1.0, 0.0), mults=(1.0,))
            zs = [pr_.peek(sp_ * t, (1e7, 1e7, 0.0)) for t in R.ET_TIMES]
        elif e_ is not None:
            hb_ = e_[1][e_[2]]; p0_ = (hb_[0], hb_[1]); s1_ = max(sp_, 1.0)
            if not geo.is_free(*p0_):
                p0_ = geo.nearest_free(*p0_)[:2]
            pr_ = Z.CorridorPredictor(geo, p0_, (hb_[2] * s1_, hb_[3] * s1_), mults=(1.0,))
            zs = [pr_.peek(sp_ * t, (tp, typ, yp)) for t in R.ET_TIMES]
            goal = PL.plan((tp, typ), p0_, d_set=R.ASTAR_DSET)[1]
    for a, b in EDGES:
        if skel.get(a) and skel.get(b):
            pa, pb = skel[a], skel[b]; vis = pa[2] and pb[2]
            cv2.line(im, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (235, 235, 235) if vis else (120, 120, 120), 2 if vis else 1, cv2.LINE_AA)
    for n, p in skel.items():
        if p:
            cv2.circle(im, (int(p[0]), int(p[1])), 6 if n in TORSO else 3, ((0, 220, 255) if n in TORSO else (235, 235, 235)) if p[2] else (120, 120, 120), -1 if p[2] else 1, cv2.LINE_AA)
    if box is not None:
        cv2.rectangle(im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (60, 200, 60), 2)
    if seen and L[k, 8] > 0 and L[k, 7] > 5:
        # KF belief box (blue): the step log keeps the KF centre column (col 6) and height (col 7); its centre row
        # follows the detection box centre (the KF's row measurement), else the camera geometry; width ~0.41 x height
        kh = float(L[k, 7]); kcx = float(L[k, 6]); dfw = C.FX * C.H_PERSON / kh
        kcy = (box[1] + box[3] / 2.0) if box is not None else C.CY0 + C.FX * (CAM_Z - (Z_FLOOR + 0.5 * C.H_PERSON)) / max(dfw, 1.0)
        kw = 0.41 * kh
        cv2.rectangle(im, (int(kcx - kw / 2), int(kcy - kh / 2)), (int(kcx + kw / 2), int(kcy + kh / 2)), (230, 140, 30), 2)
    if pred is not None and not seen:
        h = C.FX * C.H_PERSON / max(pred[3], 1.0); w = 0.4 * h
        cv2.rectangle(im, (int(pred[0] - w / 2), int(pred[1] - h / 2)), (int(pred[0] + w / 2), int(pred[1] + h / 2)), (40, 40, 230), 2)
    if zs:
        colz = (230, 140, 30) if seen else (40, 40, 230)
        for j_, (z_, lab) in enumerate(zip(zs, ('0.7', '1.4', '2.1'))):
            q_ = project(cp, geo, z_[0], z_[1], ZMID)
            if q_ is not None and 0 <= q_[0] < 640 and 0 <= q_[1] < 480:
                cv2.circle(im, (int(q_[0]), int(q_[1])), 4 + 2 * j_, colz, -1 if q_[2] else 1, cv2.LINE_AA)
                cv2.putText(im, lab, (int(q_[0]) + 8, int(q_[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colz, 1)
    cv2.rectangle(im, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(im, f'{NAME}  step {k}  {"VISIBLE" if seen else "HIDDEN"}', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 60) if seen else (40, 40, 230), 2)
    cv2.rectangle(im, (0, 455), (640, 480), (0, 0, 0), -1)
    cv2.putText(im, 'green = detection  blue = KF  red = predicted (hidden)  yellow = torso keypoints  grey = behind wall', (6, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
    # top view
    ax.clear()
    for (x0, x1, y0, y1) in rects:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, color='#3a3a3a'))
    j0 = max(k0, k - 80)
    ax.plot(L[j0:k + 1, 3], L[j0:k + 1, 4], '-', color='#E8902A', lw=1.5)
    ax.plot(L[j0:k + 1, 0], L[j0:k + 1, 1], '-', color='#2C6FBF', lw=1.5)
    tx, ty, yaw = L[k, 0], L[k, 1], L[k, 2]
    for s_ in (-41.3, 41.3):
        a_ = math.radians(yaw + s_); ax.plot([tx, tx + 400 * math.cos(a_)], [ty, ty + 400 * math.sin(a_)], color='#2C6FBF', lw=0.6, alpha=0.6)
    ax.plot(tx, ty, '^', ms=11, color='#2C6FBF')
    ax.plot(L[k, 3], L[k, 4], 'o', ms=10, color='#E8902A')
    e = CORR.get(k)
    if e is not None and not seen:
        Hs = e[1]
        for hh in Hs:
            ax.plot(hh[0], hh[1], 'o', ms=3 + 14 * hh[4], color='#7F77DD', alpha=0.7)
        hb = Hs[e[2]]; ax.plot(hb[0], hb[1], '*', ms=16, color='#534AB7')
    if zs:
        cz = '#2C6FBF' if seen else '#C0392B'
        for j_, z_ in enumerate(zs):
            ax.plot(z_[0], z_[1], 'o', ms=4 + 2 * j_, color=cz, mfc='none' if j_ else cz, mew=1.5)
        ax.plot([z_[0] for z_ in zs], [z_[1] for z_ in zs], ':', color=cz, lw=1)
    if goal is not None:
        ax.plot(goal[0], goal[1], 's', ms=10, color='#1D9E75')
        ax.plot([tx, goal[0]], [ty, goal[1]], '--', color='#1D9E75', lw=1)
    ax.set_xlim(min(tx, L[k, 3]) - 450, max(tx, L[k, 3]) + 450); ax.set_ylim(min(ty, L[k, 4]) - 450, max(ty, L[k, 4]) + 450)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'step {k}: blue = drone, orange = person, purple = guesses (star = best)'
                 + (chr(10) + 'dots = her future points 0.7/1.4/2.1 s (blue seen, red hidden), green square = A* viewing spot' if OVERLAY else ''), fontsize=8)
    fig.canvas.draw(); top = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 360
    a_ = cv2.resize(im, (int(640 * H / 480), H)); b_ = cv2.resize(top, (int(top.shape[1] * H / top.shape[0]), H))
    fr = np.hstack([a_, b_]); fr = fr[:, : fr.shape[1] // 2 * 2]
    if vw is None:
        vw = cv2.VideoWriter(out, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'avc1'), 8, (fr.shape[1], fr.shape[0]))
    vw.write(fr)
vw.release(); print('saved', out, f'{os.path.getsize(out) / 1048576:.1f} MB')
