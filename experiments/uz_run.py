# -*- coding: utf-8 -*-
"""Closed-loop occlusion-tracking evaluation on UnrealZoo scenes.

usage (unrealzoo conda env):
  python uz_run.py <MAP> <modes...> ['@p2' '@p2a'] [--seeds=N] [--steps=N]
                   [--edgemeas --edgeaim --blindspeed --mapclamp] [--out=file.json]
Full architecture = --edgemeas --edgeaim --blindspeed --mapclamp (dyn safety always on).
Real-time loop at ~10 Hz (DT = 0.1 s); the engine moves the target on its navmesh,
the tracker is teleported each step after the safety layer.
"""
import os, sys, json, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import re
import uz_env as U
import occl_core as C
import uz_routes
import uz_maze
import uz_astar

DT = 0.1
N_STEPS = 400
KILL_STREAK = 50
CV_STALE_S = 1.0         # --cleanvel: accept any visible frame after this long without a clean one
REACQ_MIN_S = 0.5        # an occlusion counts for the re-acquisition error if the target was unseen >= 0.5 s
V_WALK, V_SPRINT, V_BRAKE = 50.0, 99.0, 20.0
A_SPRINT = 80.0
CORNER_DEG = 40.0
FLAGS = dict(edgemeas=False, edgeaim=False, blindspeed=False, mapclamp=False, gate=False, mazectrl=False, dp=False, astar=False,
             egocomp=False, worldpred=False, corridor=False, ouspeed=False, cleanvel=False, groundkf=False)
WV_MAX = 150.0          # cap on the estimated target world speed (cm/s)
CORR_INIT_BRANCH = False # --corridor: at the occlusion start near a cell centre, also hypothesise the other open directions
CORR_UTURN = False       # --corridor: hypotheses turn back in dead ends
CORR_RESET = False       # --corridor: restart from the last seen point when every hypothesis is ruled out
CORR_CAME_FROM = False   # corridor start: 'back' = the cell the estimated walker entered from (trail), not -velocity axis
CORR_REACH = False       # corridor: when every hypothesis is ruled out, restart over the reachable set from the last seen point
CORR_REEXPAND = False    # corridor: when every hypothesis has been seen empty, re-expand from the current best point
CORR_OCC_INIT = False    # corridor start: directions whose next metre is in clear view get low weight (the person just vanished)
BODY_FIX = True          # read back the real camera position each step (engine depenetration of the tracker capsule)
SLIVER_GUARD = True      # a detection much narrower / shorter than predicted keeps the predicted range (cx only)
KP_POS = False           # torso keypoints (projected engine bones + noise) give the target's image column
KP_RANGE = False         # with KP_POS: range from the neck-pelvis pixel length (Fast-Tracker 2.0)
KP_PIX_SD = 2.0          # px, simulated keypoint noise
KP_HEADING = False       # body keypoints (engine bones): facing direction from shoulders/hips sets the corridor start direction
BREADCRUMB = False       # visible tracking: fly along the person's own trail when the straight line would cut a wall corner
RW_TO_EXIT = True        # random walk ends at the exit (start point -> end point), with detours
YAW_SMOOTH = False       # low-pass + angular-acceleration limit on the yaw-rate command
YAW_TAU = 0.3            # s, low-pass time constant
YAW_ACC = 180.0          # deg/s^2, max yaw acceleration
ET_KINO = False          # look-ahead planner searches (position, velocity) with acceleration primitives from the
KINO_TAU, KINO_DEPTH, KINO_BEAM, KINO_A = 0.35, 6, 80, 200.0   # drone's current velocity (s, steps, beam, cm/s^2)
KINO_GUIDE_W = 0.01      # cost per cm of path distance (around walls) from a search node to the A* viewing spot (< one blocked view)
KINO_BLIND = False       # hidden target: the same kinodynamic search, aimed at the corridor predictor's best guess
KP_FIRST = False         # valid torso keypoints win over the box-based guards (edge rebuild / fade / top-cut keep them)
ET_PLANNER = False       # visible target: Elastic-Tracker-style planning (predict 0.7/1.4/2.1 s ahead along the corridors,
ET_TIMES = (0.7, 1.4, 2.1)   # chain of A* viewpoints that see those future points, fly to the first one in time; any direction)
YAW_INERTIA = False      # vehicle physics (not a controller tweak): the ACTUAL yaw rate follows the command as a
YAW_DYN_TAU = 0.3        # first-order lag, tau in s (sys-id of a DJI M100: yaw-rate tau 0.32 s; Tello to be measured)
FF_CTRL = False          # tracking controller with target-velocity feedforward (forward + yaw), reverse allowed
FF_V_BACK = 50.0         # max reverse speed (cm/s) with FF_CTRL
RING_FLOW = False        # with RING_R: follow the WHOLE route to the prediction, speed = prediction speed + 2/s x (distance - R): never stop, ~R behind
RING_ARC = False         # with RING_R: never stop -- full speed to the circle, then move along it to the best-view point
RING_R = 0.0             # >0: while blind, follow the A* route to the prediction up to its FIRST point inside a circle of this radius (cm)
CHASE_D = 0.0            # >0: while blind, close in on the prediction to this stand-off (cm) at full speed
ASTAR_AHEAD = 0.0        # A* goal = corridor prediction this many seconds ahead (grows with blind time); 0 = off
CORR_SPEEDS = (1.0,)     # --corridor: speed hypotheses (multiples of the estimated speed)
CORR_VMIN = 0.0         # --corridor: floor on the blind prediction speed (cm/s)
OU_TAU = 1.0            # --ouspeed: blind speed relaxes from the last estimate to the target's typical speed (s)
ASTAR_PL = None         # uz_astar.MazePlanner (built lazily for the current maze)
ASTAR_EVERY = 1         # receding horizon: replan every step (plan ~16 ms median; was 5 = ~1.2 s at the 0.23 s video step -> stale goal, stop-and-go)
ASTAR_DSET = 250.0      # stand-off distance of the A* goal from the predicted target (cm)
ASTAR_LOOK = 30.0       # pure-pursuit lookahead (cm); sweep: 30 -> 3/3, 60 -> 2/3, 120 -> 2/3 finished

# ---------------- OA-VAT module-3 planner (diffusion policy) as a rescue mode ----------------
DP_TRIGGER = 6          # planner takes over after 6 lost frames (sweep 2026-10-02: 1->1/3, 3->2/3, 6->3/3 finished;
                        # the original OA-VAT eval used --kf_max_predict_frames 3)
DP_DT = 1.0             # original: --dp_dt 1 (s per action step)
DP_YAW_MAX = 60.0       # original dp_yaw_max (deg/s)
DP_PORT = 6010
DP_CKPT = None          # None = original checkpoint; else path (e.g. the pad_after=0 retrain)
DPC = None              # multiprocessing Client to experiments/dp_server.py (follow env)
DP_LOG = None           # list -> (step, inference ms, raw 8x3 action) per plan
GK_LOG = None           # list -> per-step ground-KF record (--groundkf)
CORR_LOG = None         # list -> (step, [(x, y, dx, dy, w)], best index, start point) per blind step (--corridor)


def dp_connect(start=True, timeout=120.0):
    """Connect to the planner service; start it in the `follow` env if needed."""
    global DPC
    from multiprocessing.connection import Client
    import subprocess
    t0 = time.time(); started = False
    while time.time() - t0 < timeout:
        try:
            DPC = Client(('127.0.0.1', DP_PORT), authkey=b'oavat'); return DPC
        except (ConnectionRefusedError, OSError):
            if start and not started:
                subprocess.Popen([r'C:\Users\AUFB\miniconda3\envs\follow\python.exe', os.path.join(HERE, 'dp_server.py'), str(DP_PORT)]
                                 + ([DP_CKPT] if DP_CKPT else []),
                                 stdout=open(os.path.join(HERE, '..', 'results', 'logs', 'dp_server.log'), 'w'),
                                 stderr=subprocess.STDOUT, creationflags=0x08000000)
                started = True
            time.sleep(2.0)
    raise RuntimeError('DP planner service not reachable')


def dp_plan(img, bbox_xywh):
    """Query the diffusion policy: returns (8, 3) waypoints [cm, body frame: x fwd, y right, z up]."""
    DPC.send({'img': np.ascontiguousarray(img).tobytes(), 'shape': img.shape, 'bbox': [float(v) for v in bbox_xywh]})
    r = DPC.recv()
    return np.array(r['action'], float), r['ms']


def path_corners(P, deg=CORNER_DEG):
    if len(P) < 3:
        return []
    d = np.diff(P, axis=0); hd = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.abs((np.diff(hd) + 180.0) % 360.0 - 180.0)
    return [P[i + 1] for i in np.where(turn > deg)[0]]


def los_blocked(nav, a, b, step=12.0):
    """True if the straight segment a->b crosses occupied navgrid cells."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
    for t in np.linspace(0.0, 1.0, n):
        if not nav.is_free(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])):
            return True
    return False


def occluding_corner(nav, path, dmin=500.0, dmax=900.0, deg=55.0, look=300.0):
    """First route corner (turn > deg) at path distance dmin..dmax whose approach
    and exit legs have no line of sight -> (corner_xy, dist_along) or None."""
    if len(path) < 3:
        return None
    seg = np.hypot(*np.diff(path, axis=0).T); cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < dmin + look:
        return None
    def point_at(s):
        i = int(np.searchsorted(cum, s, side='right') - 1); i = min(max(i, 0), len(path) - 2)
        f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-6)
        return path[i] + f * (path[i + 1] - path[i])
    # navmesh paths are smoothed (many small turns): measure the heading change over a
    # +-120 cm window along the path instead of per vertex
    win = 120.0
    for s_c in np.arange(dmin, min(dmax, cum[-1] - look) + 1e-6, 25.0):
        a0, a1 = point_at(max(s_c - win, 0.0)), point_at(s_c)
        b0, b1 = point_at(s_c), point_at(s_c + win)
        h_in = math.degrees(math.atan2(a1[1] - a0[1], a1[0] - a0[0])); h_out = math.degrees(math.atan2(b1[1] - b0[1], b1[0] - b0[0]))
        if abs((h_out - h_in + 180.0) % 360.0 - 180.0) > deg:
            a, b = point_at(max(s_c - look, 0.0)), point_at(s_c + look)
            if los_blocked(nav, a, b):
                return point_at(s_c), float(s_c)
    return None


def pick_route(env, rng, tries=8):
    """From the target's current position, find a goal whose route has an occluding corner."""
    gx, gy, _ = env.target_pose()
    best = None
    for _ in range(tries):
        goal = env.random_goal(rng, (gx, gy), dmin=900.0)
        path = env.nav_target(*goal)
        oc = occluding_corner(env.nav, path)
        if oc is not None:
            return goal, path, oc
        if best is None:
            best = (goal, path)
    return best[0], best[1], None


MAZE = None          # uz_maze.MazeGeo when running in a generated maze (--maze=NAME)
TARGET_ASSET = '/Game/HumanCharacter/ThirdPersonBP/Blueprints/target.target'   # same blueprint as the L-map targetbp2
TARGET_APP = 4       # light top + jeans
MAZE_MARGIN = 30.0   # camera-to-wall safety distance in generated mazes (exact walls; same as the L map)
ANIM_TARGET = False  # maze target walks with its own movement component (animation) instead of being teleported
LIVE_LIT = False     # capture the lit frame every step (live video; slows the loop)
SNAP = None          # (dir, every_k) -> save first-person lit frames during an episode
VIDEO = None         # list -> per-step (lit image, KF box [cx,cy,a,h] or None, detection box, detected, faded)


# ---------------- ground-plane KF (--groundkf) ----------------
GK_SIG_CX = 3.0          # px, box centre
GK_SIG_CX_TOUCH = 12.0   # px, box touching an occluding wall corner (may be cut)
GK_SIG_CX_FADE = 8.0     # px, edge measurement during fade-out
GK_SIG_H_WHOLE = 0.03    # fraction of box height: full-box ranging
GK_SIG_H_HEAD = 0.08     # fraction of box height: head-row ranging
GK_SKIP_UNCLEAN = False  # skip the update on frames that may be cut (touching a wall corner, fade, vclip)
GK_SIG_A = 50.0          # cm/s^2, white-acceleration process noise (offline replay: 30-50 best at occlusion start)


def _proj_cov(cx, h, yaw, s_cx, s_h):
    """Pixel noise (cx, h) -> floor-point covariance through back_project (first order)."""
    ang = math.atan((cx - C.CX0) / C.FX); b = math.radians(yaw) + ang
    rho = C.FX * C.H_PERSON / max(h, 1.0)
    dang = C.FX / (C.FX ** 2 + (cx - C.CX0) ** 2)
    drho = 0.0 if rho >= 900.0 else -rho / max(h, 1.0)
    rho = min(rho, 900.0)
    J = np.array([[-rho * math.sin(b) * dang, math.cos(b) * drho], [rho * math.cos(b) * dang, math.sin(b) * drho]])
    return J @ np.diag([s_cx ** 2, s_h ** 2]) @ J.T + np.eye(2)


def _gkf_predict(x, P, dt):
    F = np.eye(4); F[0, 2] = F[1, 3] = dt
    q = GK_SIG_A ** 2
    Q1 = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = Q1; Q[np.ix_([1, 3], [1, 3])] = Q1
    return F @ x, F @ P @ F.T + Q


def _gkf_update(x, P, z, R):
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
    return x + K @ (z - H @ x), (np.eye(4) - K @ H) @ P


def _touching_edge(env, box, meas, tx, ty, yaw):
    rel_c = math.degrees(math.atan2(box[0] + box[2] / 2.0 - C.CX0, C.FX)); rng_c = C.FX * C.H_PERSON / max(meas[3], 1.0)
    u_c = env.edge_u(tx, ty, yaw, tgt_rel=rel_c, tgt_range=rng_c)
    return u_c is not None and box[0] - 8 <= u_c <= box[0] + box[2] + 8


def _ring_arc_velocity(corr, centre, here, wv, R, n=24, near=30.0):
    """Inside / on the R-circle around the prediction: move ALONG the circle (no stop) to the
    point from which the most hypothesis weight is visible (next best view), plus the
    prediction's own velocity so the UAV moves with the circle.  None while still outside."""
    cx, cy = centre; tx, ty = here
    if math.hypot(tx - cx, ty - cy) > R + near:
        return None
    def sees(a, b):
        return 50.0 < math.hypot(b[0] - a[0], b[1] - a[1]) < 800.0 and not uz_maze.los_blocked(MAZE, a, b)
    seen_now = [sees((tx, ty), tuple(h['p'])) for h in corr.H]
    h0 = corr.H[corr.best]; fwd = (h0['d'][0], h0['d'][1])
    best, bq = -1e9, None
    for a in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
        q = (cx + R * math.cos(a), cy + R * math.sin(a))
        if MAZE.clearance(*q) < MAZE_MARGIN or uz_maze.los_blocked(MAZE, (tx, ty), q):
            continue
        # NEW information only: hypotheses visible from q that are not visible from here already
        score = sum(h['w'] for h, s in zip(corr.H, seen_now) if not s and sees(q, tuple(h['p'])))
        score += 0.0005 * ((q[0] - cx) * fwd[0] + (q[1] - cy) * fwd[1])   # tie-break: ahead of the prediction
        score -= 0.0005 * math.hypot(q[0] - tx, q[1] - ty)            # and close to the UAV
        if score > best:
            best, bq = score, q
    if bq is None:
        return None
    h = corr.H[corr.best]; sp = math.hypot(*wv) * h.get('m', 1.0)
    ffx, ffy = h['d'][0] * sp, h['d'][1] * sp                       # the circle moves with the prediction
    dx, dy = bq[0] - tx, bq[1] - ty; dd = math.hypot(dx, dy)
    v = min(C.V_CAM_MAX, max(60.0, 3.0 * dd)) if dd > 5.0 else 0.0  # keep moving: no stop at the point
    vx, vy = (v * dx / dd if dd > 1e-3 else 0.0) + ffx, (v * dy / dd if dd > 1e-3 else 0.0) + ffy
    s = math.hypot(vx, vy)
    if s > C.V_CAM_MAX:
        vx, vy = vx * C.V_CAM_MAX / s, vy * C.V_CAM_MAX / s
    return vx, vy


def _seg_clear(a, b, margin, step=10.0):
    """Every point of segment a-b keeps at least `margin` from the walls (MAZE)."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
    for t in np.linspace(0.0, 1.0, n):
        if MAZE.clearance(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) < margin:
            return False
    return True


KP_BONES = 'clavicle_l,clavicle_r,thigh_l,thigh_r,neck_01'


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-6 else None


def _body_facing(env, tx, ty, yaw):
    """Facing direction of the person from the shoulder and hip lines (engine bones, world frame).
    Used only if the neck is inside the camera view with a clear line of sight (a visible person).
    World x/y: at yaw 90 (facing +y) left = +x, so forward = (left - right) rotated by +90 deg: (-dy, dx)."""
    try:
        B = {b['bone_name']: b['transform'] for b in json.loads(env.c.request(f'vget /object/{env.tgt}/bones {KP_BONES} world'))}
    except Exception:
        return None
    if len(B) < 5:
        return None
    nk = (B['neck_01']['X'], B['neck_01']['Y'])
    rel = (math.degrees(math.atan2(nk[1] - ty, nk[0] - tx)) - yaw + 180.0) % 360.0 - 180.0
    if abs(rel) > 41.0 or (MAZE is not None and uz_maze.los_blocked(MAZE, (tx, ty), nk)):
        return None
    lr = [(B['clavicle_l']['X'] - B['clavicle_r']['X'], B['clavicle_l']['Y'] - B['clavicle_r']['Y']),
          (B['thigh_l']['X'] - B['thigh_r']['X'], B['thigh_l']['Y'] - B['thigh_r']['Y'])]
    sx = sum(u[0] / max(math.hypot(*u), 1e-6) for u in lr); sy = sum(u[1] / max(math.hypot(*u), 1e-6) for u in lr)
    return _unit((-sy, sx))


KP_TORSO = 'neck_01,spine_03,pelvis'
KP_TORSO_DZ = 50.4       # cm, neck-to-pelvis height of this character (measured once from its bones)


def _torso_column(env, tx, ty, yaw, rng, with_range=False):
    """Simulated torso keypoint detector (engine bones projected with the camera model, only the
    visible ones, +-KP_PIX_SD px noise).  Returns (mean image column or None, range or None).
    Range: rows of neck and pelvis (pitch 0) differ by FX * dz / d_forward, dz = their height
    difference (a fixed body parameter, like the known 168 cm height) -> d_forward -> range."""
    try:
        B = {b['bone_name']: b['transform'] for b in json.loads(env.c.request(f'vget /object/{env.tgt}/bones {KP_TORSO} world'))}
    except Exception:
        return None, None
    us = []; rows = {}; rels = {}
    for n_, t in B.items():
        x, y = t['X'], t['Y']
        rel = (math.degrees(math.atan2(y - ty, x - tx)) - yaw + 180.0) % 360.0 - 180.0
        if abs(rel) > 41.0 or (MAZE is not None and uz_maze.los_blocked(MAZE, (tx, ty), (x, y))):
            continue
        us.append(C.CX0 + C.FX * math.tan(math.radians(rel)) + float(rng.normal(0.0, KP_PIX_SD)))
        dfw = math.hypot(x - tx, y - ty) * math.cos(math.radians(rel))          # forward distance
        rows[n_] = C.CY0 - C.FX * (t['Z'] - env.cam_z) / max(dfw, 1.0) + float(rng.normal(0.0, KP_PIX_SD)); rels[n_] = rel
    u = float(np.mean(us)) if len(us) >= 2 else None
    rho = None
    if with_range and 'neck_01' in rows and 'pelvis' in rows and all(0 <= rows[k] < U.H for k in ('neck_01', 'pelvis')):
        lpx = rows['pelvis'] - rows['neck_01']
        if lpx > 5.0:
            dfw = C.FX * KP_TORSO_DZ / lpx
            rho = dfw / max(math.cos(math.radians(0.5 * (rels['neck_01'] + rels['pelvis']))), 0.3)
    return u, rho


def _et_velocity(tx, ty, ph, wv):
    """Elastic-Tracker-style translation while the target is visible (simplified, no trajectory optimisation):
    1. predict the walker ET_TIMES ahead along the corridor centre lines at the current speed, preferring
       to go straight (the 'safe, minimum-acceleration' continuation);
    2. for each future point an A* viewpoint 1.5-3.3 m away with line of sight to it, each search starting
       from the previous viewpoint (a chain of viewing spots over time);
    3. world-frame velocity along the A* path to the first viewpoint, fast enough to be there at ET_TIMES[0].
    Returns (vx, vy, chain_of_paths) or None."""
    global ASTAR_PL
    if ASTAR_PL is None or ASTAR_PL.geo is not MAZE:
        ASTAR_PL = uz_astar.MazePlanner(MAZE, margin=MAZE_MARGIN)
    p0 = tuple(ph) if MAZE.is_free(*ph) else MAZE.nearest_free(*ph)[:2]
    sp = math.hypot(*wv)
    v0 = wv if sp > 1.0 else (1.0, 0.0)
    pred = uz_maze.CorridorPredictor(MAZE, p0, v0, mults=(1.0,))
    blind = (1e7, 1e7, 0.0)                       # a camera that sees nothing: no negative information here
    zs = [pred.peek(sp * t, blind) for t in ET_TIMES]
    start = (tx, ty); chain = []
    for z in zs:
        path, goal = ASTAR_PL.plan(start, z, d_set=ASTAR_DSET)
        if not path:
            break
        chain.append(path); start = goal
    if not chain:
        return None
    k_ = 0
    if len(chain) > 1 and math.hypot(chain[0][-1][0] - tx, chain[0][-1][1] - ty) < 30.0:
        k_ = 1                                    # already at the first viewing spot: head for the next one
    path = [(tx, ty)] + list(chain[k_][1:])
    L_ = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))
    t_need = ET_TIMES[0]                          # the chain is spaced ET_TIMES[0] apart: next viewing spot in time
    spd = min(C.V_CAM_MAX, L_ / max(t_need, 0.2))
    cx_, cy_ = uz_astar.pursue(path, (tx, ty), lookahead=ASTAR_LOOK)
    ux, uy = cx_ - tx, cy_ - ty; n_ = math.hypot(ux, uy)
    if n_ < 1e-3:
        return 0.0, 0.0, chain
    return spd * ux / n_, spd * uy / n_, chain


def _los_blocked_many(R, P, b):
    """Vectorised segment-vs-rectangle test (as uz_maze.los_blocked) for many start points P (N x 2) to one point b."""
    N, M = len(P), len(R)
    ax = P[:, 0][:, None]; ay = P[:, 1][:, None]
    dx = (b[0] - P[:, 0])[:, None]; dy = (b[1] - P[:, 1])[:, None]
    x0, x1, y0, y1 = R[:, 0][None], R[:, 1][None], R[:, 2][None], R[:, 3][None]
    t0 = np.zeros((N, M)); t1 = np.ones((N, M)); ok = np.ones((N, M), bool)
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        p = np.broadcast_to(p, (N, M)); q = np.broadcast_to(q, (N, M))
        zero = np.abs(p) < 1e-12
        ok &= ~(zero & (q < 0))
        t = q / np.where(zero, 1.0, p)
        t0 = np.where((p < 0) & ~zero, np.maximum(t0, t), t0)
        t1 = np.where((p > 0) & ~zero, np.minimum(t1, t), t1)
    return (ok & (t0 <= t1)).any(axis=1)


_KINO_ACC = np.array([(0.0, 0.0)] + [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 45)])


def _et_kino_velocity(tx, ty, cvx, cvy, ph, wv):
    """Kinodynamic version of the look-ahead planner (Fast-Tracker-style front end): beam search over
    (position, velocity) with acceleration primitives (hold, or KINO_A in 8 directions, for KINO_TAU s) starting
    from the drone's CURRENT velocity, |v| <= V_CAM_MAX, inflated-wall free space; cost at 0.7/1.4/2.1 s:
    line of sight to the predicted walker blocked (4), outside 1.5-3.3 m (per 50 cm), |D - 250| (small), and a
    small cost per acceleration.  Returns the commanded velocity after the first primitive, or None."""
    global ASTAR_PL
    if ASTAR_PL is None or ASTAR_PL.geo is not MAZE:
        ASTAR_PL = uz_astar.MazePlanner(MAZE, margin=MAZE_MARGIN)
    pl = ASTAR_PL
    p0 = tuple(ph) if MAZE.is_free(*ph) else MAZE.nearest_free(*ph)[:2]
    sp = math.hypot(*wv)
    pred = uz_maze.CorridorPredictor(MAZE, p0, wv if sp > 1.0 else (1.0, 0.0), mults=(1.0,))
    zs = [np.array(pred.peek(sp * t, (1e7, 1e7, 0.0))) for t in ET_TIMES]
    return _kino_search(tx, ty, cvx, cvy, zs)


def _kino_search(tx, ty, cvx, cvy, zs, guide=None):
    """Beam search over (position, velocity) with acceleration primitives from the drone's current state;
    zs = target points at ET_TIMES (seen-or-not and distance are scored there).  guide = optional grid of path
    distances (cm) to a long-range goal (A* viewing spot): KINO_GUIDE_W x guide is added when ranking nodes, so the
    search still makes progress when nothing is visible within its short horizon.  Returns (vx, vy, cost) or None."""
    global ASTAR_PL
    if ASTAR_PL is None or ASTAR_PL.geo is not MAZE:
        ASTAR_PL = uz_astar.MazePlanner(MAZE, margin=MAZE_MARGIN)
    pl = ASTAR_PL
    zs = [np.asarray(z, float) for z in zs]
    R = np.array(MAZE.rects, float); acc = _KINO_ACC * KINO_A; na = len(acc); tau = KINO_TAU

    def free(Q):
        ix = np.floor((Q[:, 0] - pl.x0) / pl.cell).astype(int); iy = np.floor((Q[:, 1] - pl.y0) / pl.cell).astype(int)
        ins = (ix >= 0) & (ix < pl.nx) & (iy >= 0) & (iy < pl.ny)
        out = np.zeros(len(Q), bool); out[ins] = pl.free[iy[ins], ix[ins]]
        return out

    P = np.array([[tx, ty]], float); V = np.array([[cvx, cvy]], float); Cst = np.zeros(1); F = np.zeros(1, int)
    for d in range(1, KINO_DEPTH + 1):
        Pn = np.repeat(P, na, 0); Vn = np.repeat(V, na, 0); Cn = np.repeat(Cst, na); Fn = np.repeat(F, na)
        A = np.tile(acc, (len(P), 1))
        V2 = Vn + A * tau
        s_ = np.hypot(V2[:, 0], V2[:, 1]); V2 *= np.minimum(1.0, C.V_CAM_MAX / np.maximum(s_, 1e-6))[:, None]
        Pm = Pn + (0.75 * Vn + 0.25 * V2) * (tau / 2); P2 = Pn + 0.5 * (Vn + V2) * tau
        if d == 1:
            Fn = np.tile(np.arange(na), len(P))
        Cn = Cn + 0.05 * (np.hypot(A[:, 0], A[:, 1]) > 0)
        k_ = d * len(ET_TIMES) // KINO_DEPTH - 1 if d % (KINO_DEPTH // len(ET_TIMES)) == 0 else None
        if k_ is not None:
            z = zs[k_]; D = np.hypot(P2[:, 0] - z[0], P2[:, 1] - z[1])
            Cn = Cn + 4.0 * _los_blocked_many(R, P2, z) + np.maximum(0, 150 - D) / 50 + np.maximum(0, D - 330) / 50 + 0.002 * np.abs(D - 250)
        okm = free(Pm) & free(P2)
        if not okm.any():
            return None
        P2, V2, Cn, Fn = P2[okm], V2[okm], Cn[okm], Fn[okm]
        key = (np.floor(P2[:, 0] / 20).astype(np.int64) * 1000003 + np.floor(P2[:, 1] / 20).astype(np.int64)) * 10007 \
              + (np.floor(V2[:, 0] / 40).astype(np.int64) + 50) * 101 + (np.floor(V2[:, 1] / 40).astype(np.int64) + 50)
        rank = Cn.copy()
        if guide is not None:
            gx_ = np.clip(np.floor((P2[:, 0] - pl.x0) / pl.cell).astype(int), 0, pl.nx - 1)
            gy_ = np.clip(np.floor((P2[:, 1] - pl.y0) / pl.cell).astype(int), 0, pl.ny - 1)
            g_ = guide[gy_, gx_]; rank = rank + KINO_GUIDE_W * np.where(np.isfinite(g_), g_, 3000.0)
        o = np.lexsort((rank, key)); first = np.r_[True, key[o][1:] != key[o][:-1]]; o = o[first]
        o = o[np.argsort(rank[o])][:KINO_BEAM]
        P, V, Cst, F = P2[o], V2[o], Cn[o], Fn[o]
    a = acc[F[0]]
    v1 = np.array([cvx, cvy]) + a * tau
    n_ = np.hypot(*v1)
    if n_ > C.V_CAM_MAX:
        v1 *= C.V_CAM_MAX / n_
    return float(v1[0]), float(v1[1]), float(Cst[0])


def _cut_at_ring(path, centre, R, here):
    """First point of the polyline `path` that lies within R of `centre` (path truncated there).
    Already inside the circle -> stay where we are."""
    if math.hypot(here[0] - centre[0], here[1] - centre[1]) <= R:
        return [tuple(here)]
    out = [tuple(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.array(a, float), np.array(b, float); d = b - a; L = float(np.hypot(*d))
        n = max(2, int(L / 5.0))
        for t in np.linspace(0.0, 1.0, n):
            q = a + t * d
            if math.hypot(q[0] - centre[0], q[1] - centre[1]) <= R:
                out.append((float(q[0]), float(q[1]))); return out
        out.append(tuple(b))
    return out


def _teleport_target(env, x, y, yaw):
    env.c.request([f'vset /object/{env.tgt}/location {x} {y} {env.z}', f'vset /object/{env.tgt}/rotation 0 {yaw} 0'], -1)


def run_episode(env, mode, seed, profile='p2', steps=N_STEPS, log=None):
    rng = np.random.default_rng(seed)
    nav = env.nav
    mt = None
    if MAZE is not None:
        # --- scripted target in the generated maze ---
        corner_info = None
        for attempt in range(20):
            mt = uz_maze.MazeTarget(MAZE, rng, profile)
            if MAZE.info.get('gates') and profile == 'rw':
                mt.plan_random(to_exit=RW_TO_EXIT)     # random walk from the entrance (to the exit), random acceleration
            elif MAZE.info.get('gates'):
                mt.plan_gates()                        # fixed route: entrance (top-left) -> exit (bottom-right)
            elif not mt.plan():
                continue
            x0, y0, h0 = mt.point_at(0.0)
            _teleport_target(env, x0, y0, h0); time.sleep(0.3)
            tx, ty = mt.tracker_start; yaw = math.degrees(math.atan2(y0 - ty, x0 - tx))
            env.set_tracker(tx, ty, yaw); time.sleep(0.3)
            _, seen, _, _ = env.frame()
            if env.verbose:
                print(f'[uz] maze route: start ({x0:.0f},{y0:.0f}) corner {mt.corner} at {mt.s_corner:.0f} cm, tracker ({tx:.0f},{ty:.0f}), target px {seen}')
            if seen >= 400:
                corner_info = (np.array(mt.corner), mt.s_corner); break
        c_dist = round(mt.s_corner) if corner_info else None
        res = _loop(env, mode, seed, profile, steps, log, rng, nav, mt.sa, c_dist, tx, ty, yaw,
                    x0, y0, mt.point_at(1e9)[:2], [], True, mt)
        if ANIM_TARGET:
            env.c.request(f'vbp {env.tgt} set_move 0 0')                # stop the walking target
        return res
    return _run_nav(env, mode, seed, profile, steps, log, rng, nav)


def _run_nav(env, mode, seed, profile, steps, log, rng, nav):
    sprint_a = A_SPRINT if profile == 'p2' else float(rng.uniform(-A_SPRINT, A_SPRINT))
    P = env.nav.pts
    clear_pts = [i for i in range(len(P)) if env.nav.clearance(P[i, 0], P[i, 1]) > 70.0]
    corner_info = None
    bank = uz_routes.load_bank(env.map)
    n_try = len(bank) if bank else 24
    for attempt in range(n_try):
        if bank:                                       # offline corner-route bank (guaranteed occluding corner)
            r = bank[(seed + attempt) % len(bank)]
            s = [r['start'][0], r['start'][1], env.z]
        elif attempt < len(env.starts):
            s = env.starts[(seed + attempt) % len(env.starts)]
        else:
            i = int(rng.choice(clear_pts)); s = [float(P[i, 0]), float(P[i, 1]), env.z]
        # teleport first, then re-issue a nav goal AT the new spot so the movement
        # component forgets the previous goal (reverse order drags the target back)
        env.c.request([f'vset /object/{env.tgt}/location {s[0]} {s[1]} {env.z}', f'vbp {env.tgt} nav_to_goal_bypath {s[0]} {s[1]} {env.z_floor}'], -1)
        time.sleep(0.8)
        gx0, gy0, _ = env.target_pose()
        if math.hypot(gx0 - s[0], gy0 - s[1]) > 60.0:
            continue                                   # teleport did not take (navmesh rejected the spot)
        if bank:
            goal = tuple(r['goal']); path = np.array(r['path'], float)
            corner_info = (np.array(r['corner'], float), float(r['corner_dist'])); heading = float(r['heading'])
        else:
            goal, path, corner_info = pick_route(env, rng)
            heading = None
            if len(path) >= 2:
                j = min(len(path) - 1, 3)
                heading = math.degrees(math.atan2(path[j][1] - path[0][1], path[j][0] - path[0][0]))
        if env.verbose:
            print(f'[uz] attempt {attempt}: start ({s[0]:.0f},{s[1]:.0f}) goal {goal} corner {None if corner_info is None else (corner_info[0].round().tolist(), corner_info[1])}')
        if corner_info is None:
            continue                                   # no occluding corner reachable from here
        corners = [corner_info[0]]; sprinted = False
        env.set_target_speed(0.0, acc=200.0)          # hold the target still while the tracker is placed
        time.sleep(0.3)
        tx, ty, yaw, seen = env.place_tracker_behind(heading=heading)
        if seen >= 400:
            break
        corner_info = None
        if env.verbose:
            print(f'[uz] start attempt {attempt}: target not visible from any candidate, trying next start')
    if corner_info is None:                       # no occluding-corner route found: plain route, note it
        gx, gy, _ = env.target_pose()
        goal = env.random_goal(rng, (gx, gy)); path = env.nav_target(*goal); corners = path_corners(path); sprinted = False
        env.set_target_speed(V_WALK, acc=200.0)
        tx, ty, yaw, seen = env.place_tracker_behind()
    c_dist = round(corner_info[1]) if corner_info else None
    env.set_target_speed(V_WALK, acc=200.0)           # release the target now that the tracker is in place
    if corner_info is not None:
        env.nav_target(*goal)                         # re-issue the route (a zero-speed nav may have been dropped)
    gx, gy, _ = env.target_pose()
    return _loop(env, mode, seed, profile, steps, log, rng, nav, sprint_a, c_dist, tx, ty, yaw, gx, gy, goal, corners, sprinted, None)


def _loop(env, mode, seed, profile, steps, log, rng, nav, sprint_a, c_dist, tx, ty, yaw, gx, gy, goal, corners, sprinted, mt):
    cvx = cvy = 0.0
    kf = C.make_kf(mode); mean = cov = None; last_bbox = None
    npx_win, w_good, a_good, h_good = [], None, None, None
    blind_run = 0; streak = 0; AR = 0.0
    n_occl = n_fade = n_clamp = n_hit = n_cap = n_vclip = 0
    n_push = 0; n_et = 0; n_kpfirst = 0; n_kblind = 0
    dz_head = None
    done_run = 0; n_reacq = 0; reacq_list = []; reacq_ev = []        # multi-corner routes: every re-acquisition
    ph_prev = None                                      # belief's floor position at the previous step (no-wall-crossing clamp)
    dp_seq = None; dp_i = 0; dp_t = 0.0; n_dp_plans = 0; n_dp_steps = 0   # OA-VAT planner rescue state
    as_path = None; as_k = -999                                            # A* expert state
    wv = (0.0, 0.0)                                                        # target world velocity (cm/s), --worldpred
    corr = None                                                            # uz_maze.CorridorPredictor during a blind phase
    gk_x = gk_P = None; vis_pe = []; vis_ve = []; init_ev = []; vis_rng = []; n_rpfail = 0           # --groundkf state; visible-phase diagnostics
    ph_clean = None; t_clean = 0.0                             # --cleanvel: velocity from whole-body frames only
    v_sum = 0.0; v_n = 0; v_upd = 0; t_blind = 0.0                                    # --ouspeed: typical visible speed, time blind
    dp_log = DP_LOG
    reacq = reacq_cm = reacq_dr = None
    last_g = (gx, gy); still = 0
    drv = uz_maze.AnimatedDriver(mt, env) if (mt is not None and ANIM_TARGET) else None
    blind_t = 0.0
    gyaw = drv.yaw if drv is not None else 0.0
    w_prev = 0.0                                                           # last yaw-rate command (YAW_SMOOTH)
    w_act = 0.0                                                            # actual yaw rate of the vehicle (YAW_INERTIA)
    trail = []                                                             # person's recent floor positions (BREADCRUMB)
    n_kp = n_kpr = n_sliver = 0; kp_rng = np.random.default_rng(seed + 7919)               # own stream: must not change the target's random motion
    kp_head = None; kp_t = 99.0                                            # body facing (unit vector, world) and its age (KP_HEADING)
    t_prev = time.time(); dt = DT
    for k in range(steps):
        t0 = time.time()
        dt = float(np.clip(t0 - t_prev, 0.05, 0.3)) if k > 0 else DT   # integrate with the real loop period
        t_prev = t0
        if mt is not None and drv is None:                  # scripted target: advance and teleport, then look
            gx, gy, gyaw = mt.step(dt)
            _teleport_target(env, gx, gy, gyaw)
        elif drv is not None:                               # animated target: steer + set_move from its last pose
            drv.step(dt, (gx, gy, gyaw))
        if BODY_FIX and k > 0:                              # engine pushed the tracker capsule: trust the engine
            cl = U._floats(env.c.request(f'vget /camera/{env.cam}/location'))
            if len(cl) >= 2 and math.hypot(cl[0] - tx, cl[1] - ty) > 5.0:
                tx, ty = cl[0], cl[1]; n_push += 1
            if len(cl) >= 3:
                env.cam_z = cl[2]                           # true camera height (spawn-time z is read before the capsule settles)
        box, npx, _, img = env.frame(lit=LIVE_LIT or (bool(SNAP) and k % SNAP[1] == 0))   # no lit capture for VIDEO: keeps loop timing
        if img is not None and SNAP and k % SNAP[1] == 0:
            import cv2
            cv2.imwrite(os.path.join(SNAP[0], f'k{k:03d}.png'), img)
        if mt is None or drv is not None:
            gx, gy, gyaw = env.target_pose()
        # ---- target speed profile: manoeuvre at the first route corner ----
        if mt is None and not sprinted and corners and math.hypot(gx - corners[0][0], gy - corners[0][1]) < 150.0:
            env.set_target_speed(V_SPRINT if sprint_a >= 0 else V_BRAKE, acc=abs(sprint_a) if abs(sprint_a) > 5 else 5.0)
            sprinted = True
        # ---- measurement ----
        meas = None; faded = False; fade_meas = False; border = False
        if box is not None and npx >= 30:
            bx, by, bw, bh = box
            # range from the head row (box top): rho = dz_head / tan(elevation); the box
            # bottom is unreliable here (feet clipped by the frame, clutter, floor decal)
            eps = math.atan((C.CY0 - by) / C.FX) + math.radians(U.PITCH)
            whole = by > 2 and by + bh < U.H - 2                          # head and feet both inside the frame
            if whole:
                # full box height (as in the L map): twice as stable as the head row when the
                # head is only ~27 cm above the camera; also keeps the head-row model calibrated
                h_v = float(bh)
                dz_new = (C.FX * C.H_PERSON / bh) * math.tan(eps)
                dz_head = dz_new if dz_head is None else 0.9 * dz_head + 0.1 * dz_new
            else:
                if dz_head is None:
                    rho0 = math.hypot(gx - tx, gy - ty)                  # one-time calibration at spawn range
                    dz_head = rho0 * math.tan(eps)
                rho_h = dz_head / math.tan(eps) if abs(math.tan(eps)) > 1e-3 and dz_head * math.tan(eps) > 0 else None
                h_v = C.FX * C.H_PERSON / rho_h if rho_h and 60.0 < rho_h < 1500.0 else float(bh)
            meas = np.array([bx + bw / 2.0, by + h_v / 2.0, bw / max(h_v, 1), h_v], float)
            kp_r_ok = False; kp_c_ok = False
            if KP_POS:
                # torso keypoints (neck, upper spine, pelvis) as a simulated keypoint detector: engine bones
                # projected with the camera model, only the visible ones, +-KP_PIX_SD px noise; their mean
                # column replaces the box centre (arm / leg swing makes the box centre wobble with the gait)
                u_kp, rho_kp = _torso_column(env, tx, ty, yaw, kp_rng, with_range=KP_RANGE)
                if u_kp is not None:
                    meas[0] = u_kp; n_kp += 1; kp_c_ok = True
                if rho_kp is not None and 60.0 < rho_kp < 1500.0:
                    # Fast-Tracker 2.0 style range: neck-to-pelvis pixel length (invariant to arm / leg swing,
                    # works with the feet out of view) -> equivalent full-body box height for the KF
                    meas[3] = C.FX * C.H_PERSON / rho_kp; n_kpr += 1; kp_r_ok = True
            bh = h_v
            ref = max(npx_win) if npx_win else 0
            u = None
            if FLAGS['edgemeas'] and mean is not None and w_good is not None and ref > 0 and npx < C.FADE_FRAC * ref:
                # ratio trigger AND the box must actually touch the occluder-edge column
                rel_t = math.degrees(math.atan2(mean[0] - C.CX0, C.FX)); rng_t = C.FX * C.H_PERSON / max(mean[3], 1.0)
                u = env.edge_u(tx, ty, yaw, tgt_rel=rel_t, tgt_range=rng_t)
                if u is not None and not (bx - 8 <= u <= bx + bw + 8 and bw < 0.8 * w_good):
                    u = None
            vclip = False
            border = by <= 2                                    # head row cut by the image top (too close)
            prior = mean[:4] if mean is not None else last_bbox       # KF belief, or last box for 'none'
            top_pred = (prior[1] - prior[3] / 2.0) if prior is not None else None
            sliver = (SLIVER_GUARD and prior is not None and not border and not kp_r_ok
                      and (box[2] < 0.35 * prior[2] * prior[3] or box[3] < 0.5 * prior[3]))
            if sliver:
                # only a sliver of the person is left (a hand / leg past the wall edge): its height says
                # nothing about the range (a 6 x 54 px sliver read as 9 m) -> cx only, range from the prior
                meas = np.array([meas[0], prior[1], prior[2], prior[3]], float); vclip = True; n_vclip += 1; n_sliver += 1
            elif border and prior is not None:
                meas = np.array([meas[0], prior[1], prior[2], meas[3] if (KP_FIRST and kp_r_ok) else max(prior[3], bh)], float); vclip = True; n_vclip += 1
            elif (prior is not None and ref > 0 and npx < C.FADE_FRAC * ref and abs(box[1] - top_pred) > 0.15 * prior[3]
                  and _touching_edge(env, box, meas, tx, ty, yaw)):
                # (only at an occluding wall edge: a target walking away also shrinks and its box top moves,
                #  but it is whole -- without this guard the range froze on random routes)
                # the box top is not the head (sliver at fade-out, head already occluded):
                # the head-row range is invalid -> keep range/height from the belief, cx only
                meas = np.array([meas[0], prior[1], prior[2], meas[3] if (KP_FIRST and kp_r_ok) else prior[3]], float); vclip = True; n_vclip += 1
            elif False and u is None and mean is not None and h_good is not None:   # bottom-clip guard retired: h now comes from the head row
                # vertical clipping by clutter (plants, carts): the box gets shorter while its
                # top row stays where the belief predicts -> range from h would be wrong;
                # keep h / cy from the belief and let only cx carry innovation
                h_pred = mean[3]; top_pred = mean[1] - h_pred / 2.0
                if bh < 0.85 * h_pred and abs(by - top_pred) < 0.25 * h_pred:
                    meas = np.array([meas[0], mean[1], mean[2], h_pred], float); vclip = True; n_vclip += 1
            if u is not None and KP_FIRST and kp_c_ok:
                n_kpfirst += 1                         # half behind a wall corner, but the torso is seen: keep the keypoints
            elif u is not None:
                faded = True; n_fade += 1
                meas = C.edge_measurement(box, u, w_good, mean)
                fade_meas = True
            elif not vclip:
                npx_win.append(npx); npx_win = npx_win[-8:]
                if mean is not None:
                    w_good, a_good, h_good = mean[2] * mean[3], mean[2], mean[3]
                else:
                    w_good, a_good, h_good = meas[2] * meas[3], meas[2], meas[3]
        # ---- estimation ----
        bw = None                                           # blind belief kept in WORLD coordinates (not via the image box)
        if kf is not None and FLAGS['egocomp'] and mean is not None and ph_prev is not None:
            # ego-motion compensation: the belief lives in image coordinates, so when the camera
            # moves/turns it would move with the camera.  Re-anchor it: take last step's belief
            # floor point (world) and re-project it into the current camera pose.
            rp = C.re_project(ph_prev[0], ph_prev[1], tx, ty, yaw)
            if rp is not None:
                mean[0], mean[3] = rp
        if kf is not None:
            if meas is not None and mean is None:
                mean, cov = kf.initiate(meas)
            elif mean is not None:
                if mode.startswith('ca'):
                    mean, cov = kf.predict(mean, cov, coasting=(meas is None))
                else:
                    mean, cov = kf.predict(mean, cov)
                if meas is not None:
                    if blind_t >= REACQ_MIN_S - 1e-6:
                        e_cm, e_dr = C.world_reacq(mean[0], mean[3], tx, ty, yaw, gx, gy)
                        bx_, by_ = C.back_project(mean[0], mean[3], tx, ty, yaw)
                        if FLAGS['worldpred'] and ph_prev is not None:
                            # the blind belief lives in world coordinates: score that, not the display box
                            bx_, by_ = ph_prev; e_cm = math.hypot(bx_ - gx, by_ - gy)
                        n_reacq += 1; reacq_list.append(e_cm)
                        # per-event record: error split along / across the target's true motion
                        hx_, hy_ = gx - last_g[0], gy - last_g[1]; hn_ = math.hypot(hx_, hy_)
                        ex_, ey_ = bx_ - gx, by_ - gy
                        along = (ex_ * hx_ + ey_ * hy_) / hn_ if hn_ > 1e-3 else float('nan')
                        cross = (-ex_ * hy_ + ey_ * hx_) / hn_ if hn_ > 1e-3 else float('nan')
                        reacq_ev.append(dict(k=k, blind=blind_run, e=round(e_cm, 1), along=round(along, 1), cross=round(cross, 1),
                                             v_true=round(hn_ / dt, 1), v_est=round(math.hypot(*wv), 1),
                                             rng=round(math.hypot(gx - tx, gy - ty)), s=round(mt.s) if mt is not None else None,
                                             planner=int(as_path is not None or dp_seq is not None)))
                        if reacq is None:
                            reacq = abs(mean[0] - meas[0]); reacq_cm, reacq_dr = e_cm, e_dr
                    mean, cov = kf.update(mean, cov, meas)
            if FLAGS['worldpred'] and meas is None and mean is not None and ph_prev is not None:
                # blind prediction in WORLD coordinates: the image-space KF velocity would rotate with
                # the camera (belief swings in an arc around the UAV); push the floor point with the
                # target's world velocity estimated while it was visible instead
                if FLAGS['corridor'] and MAZE is not None:
                    # follow the maze corridors instead of a straight line (turns at walled-off cell centres)
                    if corr is None:
                        if not MAZE.is_free(*ph_prev):
                            fx_, fy_, _ = MAZE.nearest_free(*ph_prev); ph_prev = (fx_, fy_)   # start inside the maze
                        v_init = wv
                        if KP_HEADING and kp_head is not None and kp_t <= 0.5:
                            sp0 = max(math.hypot(*wv), 30.0)         # direction from the body, speed from the estimate
                            v_init = (kp_head[0] * sp0, kp_head[1] * sp0)
                        cf = None
                        if CORR_CAME_FROM and trail:
                            # most recent cell change on the ESTIMATED visible trail -> where the walker came from
                            inf_ = MAZE.info
                            def _cl(q):
                                return (int(math.floor((q[0] - inf_['x0']) / inf_['cell'])), int(math.floor((q[1] - inf_['y0']) / inf_['cell'])))
                            c_now = _cl(ph_prev)
                            for q_ in reversed(trail):
                                cq = _cl(q_)
                                if cq != c_now:
                                    dd = (cq[0] - c_now[0], cq[1] - c_now[1])
                                    cf = dd if dd in uz_maze.DIRS else None
                                    break
                        corr = uz_maze.CorridorPredictor(MAZE, ph_prev, v_init, mults=CORR_SPEEDS, uturn=CORR_UTURN, reset=CORR_RESET, init_branch=CORR_INIT_BRANCH,
                                                          cam=(tx, ty, yaw), occ_init=CORR_OCC_INIT, reexpand=CORR_REEXPAND,
                                                          came_from=cf, reach=CORR_REACH)
                        if mt is not None and dt > 1e-3:
                            vt_ = ((gx - last_g[0]) / dt, (gy - last_g[1]) / dt)
                            ax_t = max(uz_maze.DIRS, key=lambda e: e[0] * vt_[0] + e[1] * vt_[1])
                            init_ev.append(dict(k=k, verr=round(math.hypot(wv[0] - vt_[0], wv[1] - vt_[1]), 1),
                                                v_est=round(math.hypot(*wv), 1), v_true=round(math.hypot(*vt_), 1),
                                                axis_ok=int(corr.H[0]['d'] == ax_t)))
                    sp_ = math.hypot(*wv)
                    if FLAGS['ouspeed'] and v_n >= 10:
                        # mean-reverting speed: the target returns to its typical walking speed
                        v_typ = v_sum / v_n; t_blind += dt
                        sp_ = v_typ + (sp_ - v_typ) * math.exp(-t_blind / OU_TAU)
                    px_w, py_w = corr.step(max(sp_, CORR_VMIN) * dt, (tx, ty, yaw))
                    if CORR_LOG is not None:
                        CORR_LOG.append((k, [(float(h['p'][0]), float(h['p'][1]), h['d'][0], h['d'][1], round(h['w'], 3)) for h in corr.H],
                                         corr.best, [float(ph_prev[0]), float(ph_prev[1])], [float(wv[0]), float(wv[1])]))
                else:
                    px_w, py_w = ph_prev[0] + wv[0] * dt, ph_prev[1] + wv[1] * dt
                bw = (px_w, py_w)
                rp = C.re_project(px_w, py_w, tx, ty, yaw)
                if rp is not None:
                    mean[0], mean[3] = rp                       # the image box only DISPLAYS the prediction
                else:
                    n_rpfail += 1                               # (behind / beside the camera: box not updated)
            if FLAGS['mapclamp'] and meas is None and mean is not None and corr is None:
                # (not with the corridor predictor: its points are on corridor centre lines by construction,
                #  and a switch between hypotheses in different corridors is a jump across a wall -- the
                #  "no wall crossing" hold then froze the belief at the old point for the whole occlusion)
                px, py = bw if bw is not None else C.back_project(mean[0], mean[3], tx, ty, yaw)
                moved = False
                if MAZE is not None and ph_prev is not None and uz_maze.los_blocked(MAZE, ph_prev, (px, py)):
                    # the belief may not pass through a wall: slide along it (walls are axis-aligned),
                    # otherwise hold the previous position
                    qx, qy = ph_prev
                    for cx_, cy_ in ((px, ph_prev[1]), (ph_prev[0], py)):
                        if MAZE.is_free(cx_, cy_) and not uz_maze.los_blocked(MAZE, ph_prev, (cx_, cy_)):
                            qx, qy = cx_, cy_; break
                    moved = True
                else:
                    qx, qy, moved = env.map_clamp_xy(px, py)
                if moved:
                    n_clamp += 1
                    if bw is not None:
                        bw = (qx, qy)
                    rp = C.re_project(qx, qy, tx, ty, yaw)
                    if rp is not None:
                        mean[0], mean[3] = rp
            if mean is not None:
                ph_new = bw if bw is not None else C.back_project(mean[0], mean[3], tx, ty, yaw)
                if FLAGS['groundkf']:
                    # constant-velocity KF on the ground plane; the pixel noise of each measurement is
                    # projected to the floor (ellipse along the line of sight, growing with range)
                    if gk_x is not None:
                        gk_x, gk_P = _gkf_predict(gk_x, gk_P, dt)
                    gk_unclean = meas is not None and (faded or vclip or _touching_edge(env, box, meas, tx, ty, yaw))
                    if meas is not None and GK_SKIP_UNCLEAN and gk_unclean and gk_x is not None:
                        # a cut box is biased the same way for several frames in a row (the visible part
                        # trails the body): inflating its noise is not enough, skip it (predict only)
                        ph_new = (float(gk_x[0]), float(gk_x[1]))
                        rp = C.re_project(ph_new[0], ph_new[1], tx, ty, yaw)
                        if rp is not None:
                            mean[0], mean[3] = rp
                    elif meas is not None:
                        z = np.array(C.back_project(meas[0], meas[3], tx, ty, yaw))
                        s_cx, s_h = GK_SIG_CX, (GK_SIG_H_WHOLE if whole else GK_SIG_H_HEAD) * meas[3]
                        if _touching_edge(env, box, meas, tx, ty, yaw):
                            s_cx = GK_SIG_CX_TOUCH                  # box may already be cut by the wall edge
                        if faded:
                            s_cx, s_h = GK_SIG_CX_FADE, meas[3]     # edge measurement: cx rebuilt, h from the prior
                        elif vclip:
                            s_h = meas[3]                           # h taken from the prior: no range information
                        Rz = _proj_cov(meas[0], meas[3], yaw, s_cx, s_h)
                        if gk_x is None:
                            gk_x = np.array([z[0], z[1], 0.0, 0.0]); gk_P = np.diag([Rz[0, 0], Rz[1, 1], 100.0 ** 2, 100.0 ** 2])
                            gk_P[:2, :2] = Rz
                        else:
                            gk_x, gk_P = _gkf_update(gk_x, gk_P, z, Rz)
                        ph_new = (float(gk_x[0]), float(gk_x[1]))
                        if GK_LOG is not None:
                            GK_LOG.append(dict(k=k, z=[float(z[0]), float(z[1])], sd=[float(math.sqrt(Rz[0, 0])), float(math.sqrt(Rz[1, 1]))],
                                               x=[float(v) for v in gk_x], whole=int(whole), faded=int(faded), vclip=int(vclip),
                                               border=int(border), touch=int(s_cx == GK_SIG_CX_TOUCH), g=[float(gx), float(gy)],
                                               uav=[float(tx), float(ty), float(yaw)]))
                        rp = C.re_project(ph_new[0], ph_new[1], tx, ty, yaw)
                        if rp is not None:
                            mean[0], mean[3] = rp                   # image box follows the ground estimate
                        sp = math.hypot(gk_x[2], gk_x[3])
                        wv = (float(gk_x[2]), float(gk_x[3])) if sp <= WV_MAX else (float(gk_x[2]) * WV_MAX / sp, float(gk_x[3]) * WV_MAX / sp)
                    elif gk_x is not None:
                        gk_x[0], gk_x[1] = ph_new                   # blind: position from the corridor predictor, P keeps growing
                clean = True
                if FLAGS['cleanvel'] and not FLAGS['groundkf'] and meas is not None:
                    # only frames with the whole body visible: a box touching the image column of
                    # an occluding wall corner may already be cut (before the fade trigger fires),
                    # which shifts its centre.  (Box width is no cue: it also changes when the
                    # person turns or walks away.)
                    rel_c = math.degrees(math.atan2(box[0] + box[2] / 2.0 - C.CX0, C.FX)); rng_c = C.FX * C.H_PERSON / max(meas[3], 1.0)
                    u_c = env.edge_u(tx, ty, yaw, tgt_rel=rel_c, tgt_range=rng_c)
                    touching = u_c is not None and box[0] - 8 <= u_c <= box[0] + box[2] + 8
                    clean = not faded and not vclip and not touching
                if FLAGS['cleanvel'] and not FLAGS['groundkf'] and meas is not None and clean:
                    if ph_clean is not None and t_clean > 1e-3:
                        vxw, vyw = (ph_new[0] - ph_clean[0]) / t_clean, (ph_new[1] - ph_clean[1]) / t_clean
                        sp = math.hypot(vxw, vyw)
                        if sp > WV_MAX:
                            vxw, vyw = vxw * WV_MAX / sp, vyw * WV_MAX / sp
                        wv = (0.85 * wv[0] + 0.15 * vxw, 0.85 * wv[1] + 0.15 * vyw)
                    ph_clean, t_clean = ph_new, 0.0
                if (FLAGS['cleanvel'] and not FLAGS['groundkf'] and meas is not None and not clean
                        and t_clean > CV_STALE_S and ph_prev is not None):
                    # no clean frame for a while (random routes hug walls): update from consecutive frames
                    # every step until a clean frame returns -- a slightly biased velocity beats a stale one
                    vxw, vyw = (ph_new[0] - ph_prev[0]) / dt, (ph_new[1] - ph_prev[1]) / dt
                    sp = math.hypot(vxw, vyw)
                    if sp > WV_MAX:
                        vxw, vyw = vxw * WV_MAX / sp, vyw * WV_MAX / sp
                    wv = (0.85 * wv[0] + 0.15 * vxw, 0.85 * wv[1] + 0.15 * vyw)
                if FLAGS['cleanvel']:
                    t_clean += dt
                    if t_clean > 1.0:
                        ph_clean = None                             # too old to difference against
                if not FLAGS['cleanvel'] and not FLAGS['groundkf'] and meas is not None and ph_prev is not None:
                    # world velocity of the target from consecutive visible floor points (EMA, capped)
                    vxw, vyw = (ph_new[0] - ph_prev[0]) / dt, (ph_new[1] - ph_prev[1]) / dt
                    sp = math.hypot(vxw, vyw)
                    if sp > WV_MAX:
                        vxw, vyw = vxw * WV_MAX / sp, vyw * WV_MAX / sp
                    wv = (0.85 * wv[0] + 0.15 * vxw, 0.85 * wv[1] + 0.15 * vyw)
                    v_upd += 1
                    if v_upd > 20:                                  # skip the EMA warm-up from zero
                        v_sum += math.hypot(*wv); v_n += 1
                if meas is not None and mt is not None and dt > 1e-3:
                    vt_ = ((gx - last_g[0]) / dt, (gy - last_g[1]) / dt)
                    vis_pe.append(math.hypot(ph_new[0] - gx, ph_new[1] - gy))
                    vis_rng.append(math.hypot(gx - tx, gy - ty))
                    vis_ve.append(math.hypot(wv[0] - vt_[0], wv[1] - vt_[1]))
                if meas is not None:
                    trail = (trail + [tuple(ph_new)])[-60:]
                ph_prev = ph_new
                kp_t += dt
                if KP_HEADING and meas is not None:
                    hd = _body_facing(env, tx, ty, yaw)
                    if hd is not None:
                        kp_head = hd if (kp_head is None or kp_t > 0.5) else _unit((0.6 * kp_head[0] + 0.4 * hd[0], 0.6 * kp_head[1] + 0.4 * hd[1]))
                        kp_t = 0.0
            bbox_for_pid = mean[:4] if mean is not None else None
        else:
            if meas is not None:
                if blind_t >= REACQ_MIN_S - 1e-6 and last_bbox is not None:
                    e_cm, e_dr = C.world_reacq(last_bbox[0], last_bbox[3], tx, ty, yaw, gx, gy)
                    n_reacq += 1; reacq_list.append(e_cm)
                    if reacq is None:
                        reacq = abs(last_bbox[0] - meas[0]); reacq_cm, reacq_dr = e_cm, e_dr
                last_bbox = meas
            bbox_for_pid = last_bbox
        blind_run = 0 if meas is not None else blind_run + 1
        blind_t = 0.0 if meas is not None else blind_t + dt
        if meas is not None:
            corr = None; t_blind = 0.0
        if meas is None:
            n_occl += 1
        if VIDEO is not None:
            belief = (mean[:4].copy() if mean is not None else (None if bbox_for_pid is None else np.array(bbox_for_pid[:4], float)))
            VIDEO.append((img, belief, None if box is None else list(box), meas is not None, faded))
        # ---- control ----
        dp_vel = None
        if FLAGS['dp'] and meas is not None:
            dp_seq = None                                   # target re-found: leave the rescue mode
        if FLAGS['dp'] and meas is None and mean is not None and blind_run >= DP_TRIGGER:
            if dp_seq is None or dp_i >= len(dp_seq):
                # (re)plan, as in OA-VAT dp_rescue: current frame + KF predicted box -> 8 waypoints
                img_now = env.api.get_image(env.cam, 'lit')
                cxp, cyp, ap_, hp_ = mean[:4]; wp_ = ap_ * hp_
                bb = (np.clip(cxp - wp_ / 2, 0, U.W - 1), np.clip(cyp - hp_ / 2, 0, U.H - 1), max(wp_, 1.0), max(hp_, 1.0))
                act, ms = dp_plan(img_now, bb)
                pos = act[:, :2] - act[0:1, :2]               # original: shift by the first point,
                delta = np.vstack([[0.0, 0.0], np.diff(pos, axis=0)])   # then execute the increments
                c0, s0 = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                dp_seq = [(c0 * d[0] - s0 * d[1], s0 * d[0] + c0 * d[1]) for d in delta]   # body -> world (cm)
                dp_i, dp_t = 0, 0.0; n_dp_plans += 1
                if dp_log is not None:
                    wpts = np.array([tx, ty]) + np.cumsum(np.array(dp_seq), axis=0)
                    dp_log.append((k, ms, act.tolist(), wpts.tolist()))
            dx_, dy_ = dp_seq[dp_i]
            spd = min(math.hypot(dx_, dy_) / DP_DT, C.V_CAM_MAX)          # speed = |delta| / dp_dt
            if spd > 1e-3:
                head = math.degrees(math.atan2(dy_, dx_))                   # turn towards the motion direction
                dyaw = (head - yaw + 180.0) % 360.0 - 180.0
                w_dp = float(np.clip(dyaw / max(dt, 1e-3), -DP_YAW_MAX, DP_YAW_MAX))
                dp_vel = (spd * dx_ / math.hypot(dx_, dy_), spd * dy_ / math.hypot(dx_, dy_), w_dp)
            else:
                dp_vel = (0.0, 0.0, 0.0)
            dp_t += dt
            if dp_t >= DP_DT:
                dp_i += 1; dp_t = 0.0
            n_dp_steps += 1
        if FLAGS['astar'] and MAZE is not None and meas is None and mean is not None and blind_run >= DP_TRIGGER:
            global ASTAR_PL
            if ASTAR_PL is None or ASTAR_PL.geo is not MAZE:
                ASTAR_PL = uz_astar.MazePlanner(MAZE, margin=MAZE_MARGIN)
            bel = ph_prev if ph_prev is not None else C.back_project(mean[0], mean[3], tx, ty, yaw)
            if as_path is None or (k - as_k) >= ASTAR_EVERY:
                t_pl = time.time()
                aim = bel
                if ASTAR_AHEAD > 0 and corr is not None:
                    # look ahead: plan to view where the target will be, not where it is predicted now --
                    # otherwise the UAV hovers at a spot that already sees a (lagging) prediction.  The
                    # horizon grows the longer the target has been hidden.
                    tau = min(ASTAR_AHEAD * (1.0 + 0.5 * blind_t), 3.0)
                    aim = corr.peek(max(math.hypot(*wv), 40.0) * tau, (tx, ty, yaw))
                if RING_R > 0:
                    # user's ring rule: plan the route to the prediction itself, then stop at the first
                    # point of that route that enters the R-circle around it (stay on the route, don't cut in)
                    full, as_goal = ASTAR_PL.plan((tx, ty), aim, d_set=0.0, dmin=0.0, dmax=60.0)
                    as_path = (full if RING_FLOW else _cut_at_ring(full, aim, RING_R, (tx, ty))) if full else None
                elif CHASE_D > 0:
                    # chase: get right behind the prediction (the corner) instead of keeping 2.5 m, so
                    # the target is in view as soon as the UAV rounds the corner
                    as_path, as_goal = ASTAR_PL.plan((tx, ty), aim, d_set=CHASE_D, dmin=0.0, dmax=CHASE_D + 80.0)
                else:
                    as_path, as_goal = ASTAR_PL.plan((tx, ty), aim, d_set=ASTAR_DSET,
                                                     dmin=max(100.0, ASTAR_DSET - 100.0), dmax=ASTAR_DSET + 80.0)
                as_k = k; n_dp_plans += 1
                if dp_log is not None:
                    dp_log.append((k, (time.time() - t_pl) * 1000.0, None, as_path if as_path else []))
            arc_v = None
            if RING_ARC and RING_R > 0 and corr is not None:
                arc_v = _ring_arc_velocity(corr, bel, (tx, ty), wv, RING_R)
            if arc_v is not None:
                vx_, vy_ = arc_v
            elif as_path:
                cx_, cy_ = uz_astar.pursue(as_path, (tx, ty), lookahead=ASTAR_LOOK)
                dgx, dgy = as_path[-1][0] - tx, as_path[-1][1] - ty
                if RING_FLOW:
                    # keep flowing along the route ~R behind the prediction: its own speed plus a
                    # proportional term on the distance error (outside the circle -> full speed)
                    hb = corr.H[corr.best] if corr is not None else None
                    sp_pred = math.hypot(*wv) * (hb.get('m', 1.0) if hb else 1.0)
                    spd = float(np.clip(sp_pred + 2.0 * (math.hypot(bel[0] - tx, bel[1] - ty) - RING_R), 0.0, C.V_CAM_MAX))
                elif RING_ARC:
                    spd = C.V_CAM_MAX                                   # never brake towards the circle: no stop-and-go
                else:
                    spd = min(C.V_CAM_MAX, (3.0 if (CHASE_D > 0 or RING_R > 0) else 1.5) * math.hypot(dgx, dgy))   # slow down on arrival
                ux, uy = cx_ - tx, cy_ - ty; nrm = math.hypot(ux, uy)
                vx_, vy_ = (spd * ux / nrm, spd * uy / nrm) if nrm > 1e-3 else (0.0, 0.0)
            else:
                vx_, vy_ = 0.0, 0.0
            if KINO_BLIND and corr is not None and MAZE is not None:
                # same look-ahead search as while visible, aimed at the best corridor guess 0.7/1.4/2.1 s ahead
                hb_ = corr.H[corr.best]
                sp_b = max(math.hypot(*wv), CORR_VMIN) * hb_.get('m', 1.0)
                zs_b = [corr.peek(sp_b * t_, (tx, ty, yaw)) for t_ in ET_TIMES]
                guide_ = ASTAR_PL.dijkstra(as_goal, max_cost=3000.0)[0] if (as_path and as_goal is not None) else None
                kb_ = _kino_search(tx, ty, cvx, cvy, zs_b, guide=guide_)
                if kb_ is not None:
                    vx_, vy_ = kb_[0], kb_[1]; n_kblind += 1
            dyaw = (math.degrees(math.atan2(bel[1] - ty, bel[0] - tx)) - yaw + 180.0) % 360.0 - 180.0
            dp_vel = (vx_, vy_, float(np.clip(3.0 * dyaw, -DP_YAW_MAX, DP_YAW_MAX)))   # camera on the predicted target
            n_dp_steps += 1
        elif FLAGS['astar'] and meas is not None:
            as_path = None
        if bbox_for_pid is not None:
            cx_aim = bbox_for_pid[0]
            if FLAGS['edgeaim'] and (meas is None or faded):
                rel_t = math.degrees(math.atan2(bbox_for_pid[0] - C.CX0, C.FX)); rng_t = C.FX * C.H_PERSON / max(bbox_for_pid[3], 1.0)
                u = env.edge_u(tx, ty, yaw, tgt_rel=rel_t, tgt_range=rng_t)
                if u is not None:
                    cx_aim = u
            v_cmd, w_cmd = C.control(cx_aim, bbox_for_pid[3])
            if FF_CTRL and meas is not None and ph_prev is not None:
                # feedforward of the target's own motion: radial part -> forward speed, tangential part
                # -> yaw rate; the P terms only correct the residual (range 2.5 m, centred), so the UAV
                # no longer has to fall back to ~3 m to match a walker; small reverse allowed
                ux_, uy_ = ph_prev[0] - tx, ph_prev[1] - ty; rr = math.hypot(ux_, uy_)
                if rr > 1.0:
                    ux_, uy_ = ux_ / rr, uy_ / rr
                    v_rad = wv[0] * ux_ + wv[1] * uy_; v_tan = -wv[0] * uy_ + wv[1] * ux_
                    rho_ = C.FX * C.H_PERSON / max(bbox_for_pid[3], 1.0)
                    ang_ = math.degrees(math.atan2(cx_aim - C.CX0, C.FX))
                    v_cmd = float(np.clip(v_rad + 2.0 * (rho_ - C.D_SET), -FF_V_BACK, C.V_CAM_MAX))
                    w_cmd = float(np.clip(math.degrees(v_tan / rr) + 3.0 * ang_, -C.W_CAM_MAX, C.W_CAM_MAX))
            if FLAGS['blindspeed'] and meas is None:      # only when truly blind (cluttered scenes: fade frames are unreliable)
                v_cmd = C.V_CAM_MAX
            if border:                                     # target fills the frame: back off instead of closing in
                v_cmd = -40.0
        else:
            v_cmd, w_cmd = 0.0, 0.0
        if dp_vel is not None:                          # planner owns the vehicle during the rescue
            w_cmd = dp_vel[2]
        if YAW_SMOOTH:
            # low-pass the yaw-rate command (time constant YAW_TAU) and bound the yaw acceleration, so
            # box-centre wobble from the walking gait does not make the camera swing left and right
            a_ = dt / (YAW_TAU + dt)
            w_f = w_prev + a_ * (w_cmd - w_prev)
            w_cmd = w_prev + float(np.clip(w_f - w_prev, -YAW_ACC * dt, YAW_ACC * dt))
        w_prev = w_cmd
        if YAW_INERTIA:
            # the vehicle cannot change its turn rate instantly: exact first-order step for the variable dt
            w_act += (1.0 - math.exp(-dt / YAW_DYN_TAU)) * (w_cmd - w_act)
            w_cmd = w_act
        if dp_vel is not None:
            yaw += w_cmd * dt
            vdx, vdy = dp_vel[0], dp_vel[1]
        else:
            yaw += w_cmd * dt
            vdx, vdy = v_cmd * math.cos(math.radians(yaw)), v_cmd * math.sin(math.radians(yaw))
            if BREADCRUMB and MAZE is not None and v_cmd > 0 and trail and ph_prev is not None:
                # follow the person's own path: if the straight line to the person passes too close to a
                # wall (it would cut a corner -- e.g. at the maze gate -- and slide along the wrong side of
                # the wall), fly towards the newest trail point that is reachable in a straight line
                goal_bc = None
                if not _seg_clear((tx, ty), ph_prev, MAZE_MARGIN):
                    for q in reversed(trail):
                        if _seg_clear((tx, ty), q, MAZE_MARGIN):
                            goal_bc = q; break
                if goal_bc is not None:
                    ux_, uy_ = goal_bc[0] - tx, goal_bc[1] - ty; nn = math.hypot(ux_, uy_)
                    if nn > 1.0:
                        vdx, vdy = v_cmd * ux_ / nn, v_cmd * uy_ / nn
        if ET_PLANNER and dp_vel is None and meas is not None and not border and MAZE is not None and ph_prev is not None:
            et_ = (_et_kino_velocity(tx, ty, cvx, cvy, ph_prev, wv) if ET_KINO else None) or _et_velocity(tx, ty, ph_prev, wv)
            if et_ is not None:
                vdx, vdy = et_[0], et_[1]; n_et += 1
        margin = MAZE_MARGIN if MAZE is not None else C.MARGIN   # exact maze walls: real 30 cm; navgrid is already inset
        cvx, cvy, braked = C.dyn_step(nav, tx, ty, cvx, cvy, vdx, vdy, dt, margin=margin); n_cap += int(braked)
        px0, py0 = tx, ty
        qx, qy = tx + cvx * dt, ty + cvy * dt
        tx, ty, hit = C.safe_move(nav, tx, ty, qx, qy, math.hypot(cvx, cvy) * dt, margin=margin); n_hit += int(hit)
        cvx, cvy = (tx - px0) / dt, (ty - py0) / dt
        env.set_tracker(tx, ty, yaw)
        # ---- scoring ----
        r, visible, rho, bearing = C.score_step(tx, ty, yaw, gx, gy); AR += r
        streak = 0 if visible else streak + 1
        if log is not None:
            log.append((tx, ty, yaw, gx, gy, int(meas is not None), mean[0] if mean is not None else 320.0, mean[3] if mean is not None else 1.0,
                        mean is not None, npx, time.time() - t0, rho, bearing, v_cmd, math.hypot(cvx, cvy),
                        *(box if box is not None else [-1, -1, -1, -1]), int(faded), int(vclip) if box is not None else 0,
                        int(dp_vel is not None), float(wv[0]), float(wv[1])))
        # ---- route upkeep ----
        moved = math.hypot(gx - last_g[0], gy - last_g[1]); last_g = (gx, gy); still = still + 1 if moved < 2.0 else 0
        if mt is None and (math.hypot(gx - goal[0], gy - goal[1]) < 80.0 or still > 15):
            goal = env.random_goal(rng, (gx, gy)); path = env.nav_target(*goal); still = 0
            if not sprinted:
                corners = path_corners(path)
        if mt is not None and getattr(mt, 'gates', False):
            done_run = done_run + 1 if mt.done() else 0
            if done_run >= (1 if profile == 'rw' else 25):   # gated route: out of the exit for ~3 s; random walk: end of its walk = success
                return dict(map=env.map, mode=mode, profile=profile, seed=seed, c=c_dist, sa=round(sprint_a, 1), AR=round(AR, 1), EL=k + 1,
                            SR=int(visible), finished=1, occl=n_occl, hits=n_hit, push=n_push, n_et=n_et, n_kpfirst=n_kpfirst, n_kblind=n_kblind, caps=n_cap, fade=n_fade, vclip=n_vclip, clamp=n_clamp,
                            reacq=reacq and round(reacq, 1), reacq_cm=reacq_cm and round(reacq_cm, 1), reacq_dr=reacq_dr and round(reacq_dr, 1),
                            n_reacq=n_reacq, dp_plans=n_dp_plans, dp_steps=n_dp_steps, reacq_all_cm=round(float(np.mean(reacq_list)), 1) if reacq_list else None, reacq_ev=reacq_ev, vis_pos_cm=round(float(np.mean(vis_pe)), 1) if vis_pe else None, vis_vel_cms=round(float(np.mean(vis_ve)), 1) if vis_ve else None, vis_rng_cm=round(float(np.mean(vis_rng)), 1) if vis_rng else None, rpfail=n_rpfail, n_kp=n_kp, n_kpr=n_kpr, n_sliver=n_sliver, init_ev=init_ev,
                            progress=round(mt.s / mt.cum[-1], 3))
        if streak >= KILL_STREAK:
            return dict(map=env.map, mode=mode, profile=profile, seed=seed, c=c_dist, sa=round(sprint_a, 1), AR=round(AR, 1), EL=k + 1, SR=0,
                        occl=n_occl, hits=n_hit, push=n_push, n_et=n_et, n_kpfirst=n_kpfirst, n_kblind=n_kblind, caps=n_cap, fade=n_fade, vclip=n_vclip, clamp=n_clamp, reacq=reacq and round(reacq, 1),
                        reacq_cm=reacq_cm and round(reacq_cm, 1), reacq_dr=reacq_dr and round(reacq_dr, 1),
                        n_reacq=n_reacq, dp_plans=n_dp_plans, dp_steps=n_dp_steps, reacq_all_cm=round(float(np.mean(reacq_list)), 1) if reacq_list else None, reacq_ev=reacq_ev, vis_pos_cm=round(float(np.mean(vis_pe)), 1) if vis_pe else None, vis_vel_cms=round(float(np.mean(vis_ve)), 1) if vis_ve else None, vis_rng_cm=round(float(np.mean(vis_rng)), 1) if vis_rng else None, rpfail=n_rpfail, n_kp=n_kp, n_kpr=n_kpr, n_sliver=n_sliver, init_ev=init_ev,
                        progress=round(mt.s / mt.cum[-1], 3) if mt is not None else None)
        el = time.time() - t0
        if el < DT:
            time.sleep(DT - el)
    return dict(map=env.map, mode=mode, profile=profile, seed=seed, c=c_dist, sa=round(sprint_a, 1), AR=round(AR, 1), EL=steps, SR=int(visible),
                occl=n_occl, hits=n_hit, push=n_push, n_et=n_et, n_kpfirst=n_kpfirst, n_kblind=n_kblind, caps=n_cap, fade=n_fade, vclip=n_vclip, clamp=n_clamp, reacq=reacq and round(reacq, 1),
                reacq_cm=reacq_cm and round(reacq_cm, 1), reacq_dr=reacq_dr and round(reacq_dr, 1),
                n_reacq=n_reacq, dp_plans=n_dp_plans, dp_steps=n_dp_steps, reacq_all_cm=round(float(np.mean(reacq_list)), 1) if reacq_list else None, reacq_ev=reacq_ev, vis_pos_cm=round(float(np.mean(vis_pe)), 1) if vis_pe else None, vis_vel_cms=round(float(np.mean(vis_ve)), 1) if vis_ve else None, vis_rng_cm=round(float(np.mean(vis_rng)), 1) if vis_rng else None, rpfail=n_rpfail, n_kp=n_kp, n_kpr=n_kpr, n_sliver=n_sliver, init_ev=init_ev,
                progress=round(mt.s / mt.cum[-1], 3) if mt is not None else None)


def setup_maze(env, maze):
    """Load (or generate) a maze layout, build its walls in FlexibleRoom, install the geometry."""
    global MAZE
    try:
        rects, info = uz_maze.load_layout(maze)
    except FileNotFoundError:
        m = re.match(r'fr(\d+)x(\d+)(?:c(\d+))?(?:l(\d+))?(?:t(\d+))?(g)?_s(\d+)', maze)
        rects, info = uz_maze.make_maze(int(m.group(1)), int(m.group(2)), float(m.group(3) or 200), int(m.group(7)),
                                        loops=int(m.group(4)) / 100.0 if m.group(4) else 0.15,
                                        wall_t=float(m.group(5)) if m.group(5) else 20.0, gates=bool(m.group(6)))
        uz_maze.save_layout(maze, rects, info)
    cx0, cy0 = uz_maze.cell_center(info, 0, 0)
    # target: the L-map target blueprint at the L-map scale (same size / skeleton, H ~168 cm)
    env.spawn(tgt_xy=(cx0, cy0), trk_xy=(cx0 + 60, cy0), yaw=90.0, tgt_asset=TARGET_ASSET, tgt_app=TARGET_APP, tgt_scale=0.85)
    fb = U._floats(env.c.request('vget /object/FLOOR/bounds'))
    uz_maze.build_in_engine(env, rects, fb[5] if len(fb) == 6 else env.z_floor - 98.0)
    MAZE = uz_maze.MazeGeo(rects, info); env.nav = MAZE; env.map = f'maze:{maze}'
    print(f'[uz] maze {maze}: {len(rects)} walls built')
    return MAZE


def main():
    args = sys.argv[1:]
    MAP = args[0] if args and not args[0].startswith('-') and not args[0].startswith('@') else 'Hotel_Corridor'
    modes = [a for a in args[1:] if not a.startswith('@') and not a.startswith('--')] or ['ca']
    profiles = [a[1:] for a in args if a.startswith('@')] or ['p2']
    seeds = list(range(int(next((a.split('=')[1] for a in args if a.startswith('--seeds=')), 3))))
    steps = int(next((a.split('=')[1] for a in args if a.startswith('--steps=')), N_STEPS))
    out = next((a.split('=', 1)[1] for a in args if a.startswith('--out=')), os.path.join(HERE, '..', 'results', 'metrics', f'uz_{MAP}_metrics.json'))
    for f in FLAGS:
        FLAGS[f] = ('--' + f) in args
    C.MAZE_CTRL = FLAGS['mazectrl']
    if FLAGS['dp']:
        dp_connect()
    if '--single' in args:
        uz_maze.MazeTarget.MIN_AFTER = 600.0
    maze =next((a.split('=', 1)[1] for a in args if a.startswith('--maze=')), None)
    if maze:
        MAP = 'FlexibleRoom'
        out = next((a.split('=', 1)[1] for a in args if a.startswith('--out=')), os.path.join(HERE, '..', 'results', 'metrics', f'uz_maze_{maze}.json'))
    env = U.UZEnv(MAP)
    results = []
    try:
        if maze:
            setup_maze(env, maze)
        else:
            env.spawn(); env.build_navgrid()
        for profile in profiles:
            for mode in modes:
                for seed in seeds:
                    t0 = time.time()
                    res = run_episode(env, mode, seed, profile, steps)
                    res['wall_s'] = round(time.time() - t0, 1); results.append(res)
                    print('[ep]', json.dumps(res)); sys.stdout.flush()
    finally:
        json.dump(results, open(out, 'w'), indent=1); print('saved', out)
        env.close()


if __name__ == '__main__':
    main()
