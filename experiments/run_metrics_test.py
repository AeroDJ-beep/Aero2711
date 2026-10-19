# -*- coding: utf-8 -*-
"""
Closed-loop smoke test of the frozen v2 environment: AR / EL / SR metrics.

Pipeline per step (dt = 0.05 simulated, quasi-static stepping):
  oracle detection (engine object mask -> bbox)  ->  KF (none / cv / ca)
  ->  P-controller (yaw from bbox center, speed from pinhole distance)
  ->  map-privileged safety filter (one-step lookahead, speed clamp + wall slide)
  ->  unicycle camera kinematics (v<=99 cm/s, |omega|<=30 deg/s)

Target: controlled-random route through the L-corner (corner clearance c
sampled in [40,60] so occlusion is guaranteed by the leak rule d>4c, d=250),
random speed re-targeting in [40,99] with |a|<=80 cm/s^2.

Modes x seeds, same-seed pairing. Protocol: 500 steps, visible = rho<=750
and |bearing|<=45 deg, kill = 50 consecutive invisible steps,
r = 1 - |rho-250|/750 - |theta|/45.
"""
import os
import sys
import time
import math
import json
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OAVAT = r'D:\Aero2711\OA-VAT'
GYMUCV = os.path.join(OAVAT, 'gym-unrealcv')
sys.path.insert(0, os.path.join(GYMUCV, 'gym_unrealcv', 'envs', 'utils'))
sys.path.insert(0, OAVAT)

import cv2
import env_unreal          # noqa: E402
import unrealcv_basic      # noqa: E402
from kalman_filter import KalmanFilter          # noqa: E402
from kalman_filter_ca import KalmanFilterCA     # noqa: E402


class TightRMixin:
    """R shrunk to R_SCALE x recipe (sigma h/20 -> R_SCALE*h/20); Q untouched."""
    R_SCALE = 0.1

    def project(self, mean, covariance):
        rs = self.R_SCALE
        std = [rs * self._std_weight_position * mean[3],
               rs * self._std_weight_position * mean[3],
               rs * 1e-1,
               rs * self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean_p = np.dot(self._update_mat, mean)
        cov_p = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean_p, cov_p + innovation_cov


class KFTightR(TightRMixin, KalmanFilter):
    pass


class KFCATightR(TightRMixin, KalmanFilterCA):
    pass


def world_reacq(cx_est, h_est, tx, ty, yaw_deg, gx, gy):
    """World-space re-acquisition error (cm) and range error (%) of a belief
    (cx, h) against the true target position (gx, gy)."""
    ang = math.degrees(math.atan2(cx_est - CX0, FX))
    b = math.radians(yaw_deg + ang)
    rho = min(FX * H_PERSON / max(h_est, 1.0), 900.0)
    px_, py_ = tx + rho * math.cos(b), ty + rho * math.sin(b)
    rng = math.hypot(gx - tx, gy - ty)
    return math.hypot(px_ - gx, py_ - gy), 100.0 * (rho - rng) / max(rng, 1.0)


def kf_update_rscaled(kf, mean, cov, meas, sig_scale):
    """Correction step with the measurement std inflated by sig_scale
    (R' = sig_scale^2 * R); Q, P untouched.  Used on fade frames where the
    edge-reconstructed cx carries a bias of unknown size (2026-09-28)."""
    import scipy.linalg
    H = kf._update_mat
    pm, pc = kf.project(mean, cov)               # pc = H P H^T + R
    HPH = np.linalg.multi_dot((H, cov, H.T))
    R = pc - HPH
    pc2 = HPH + (sig_scale ** 2) * R
    chol, lower = scipy.linalg.cho_factor(pc2, lower=True, check_finite=False)
    K = scipy.linalg.cho_solve((chol, lower), np.dot(cov, H.T).T, check_finite=False).T
    y = meas - pm
    return mean + np.dot(y, K.T), cov - np.linalg.multi_dot((K, pc2, K.T))


def gate_d2(mean, cov, meas):
    """Mahalanobis distance of the innovation, using the STANDARD-R noise
    recipe (h/20) regardless of mode -- the gate asks 'is this plausible as
    a detection of my target', on the detector-variability scale (pose
    swings included), not on the oracle's near-zero pixel noise."""
    h = mean[3]
    std = np.array([h / 20, h / 20, 1e-1, h / 20])
    S = cov[:4, :4] + np.diag(np.square(std))
    y = meas - mean[:4]
    return float(y @ np.linalg.solve(S, y))


class HLock:
    """Hysteresis h-freeze (user-designed guard, 2026-09-07):
    lock KF h when vh collapses (the wall eating the box during fade-out),
    hold the last healthy h; release when a fresh detection returns with
    near-full height. Prevents blind-phase range divergence."""

    def __init__(self, th_lock=-5.0, rel_frac=0.8):
        self.th_lock = th_lock
        self.rel_frac = rel_frac
        self.locked = False
        self.h_hold = None
        self.h_good = None

    def apply(self, mean, meas):
        if mean is None:
            return mean
        if not self.locked:
            if mean[7] < self.th_lock and self.h_good is not None:
                self.locked = True
                self.h_hold = self.h_good
            else:
                self.h_good = mean[3]
        if self.locked:
            if meas is not None and meas[3] >= self.rel_frac * self.h_hold:
                self.locked = False
                self.h_good = meas[3]
            else:
                mean[3] = self.h_hold
                mean[7] = 0.0
                if meas is None and len(mean) >= 10:
                    mean[8:10] = 0.0   # fake accel learned from the same
                                       # contaminated frames: zero it too
        return mean

# ---------------- frozen v2 configuration ----------------
DT = 0.05
N_STEPS = 500
KILL_STREAK = 50
D_SET = 250.0
V_CAM_MAX = 99.0
W_CAM_MAX = 30.0          # deg/s
FX = 364.3
CX0 = 320.0
H_PERSON = 168.0
HLOCK = False             # hysteresis h-freeze guard (enable with --hlock)
GATE = False              # innovation gate on updates (enable with --gate)
GATE_CHI2 = 9.4877        # chi-square 95%, 4-dim measurement
GATE_REINIT = 3           # consecutive rejected detections -> re-initiate track
VCAP = False              # heading-aware speed limit from wall clearance (--vcap)
VCAP_T = 0.6              # s: allowed time-to-reach the margin boundary
FADEGATE = False          # physical partial-occlusion gate (--fadegate):
                          # mask pixels < 50% of recent peak => the wall is
                          # clipping the body => stop feeding the KF, coast
                          # from the still-clean state
FADE_FRAC = 0.5
FADE_DECAY = 0.995        # per-step decay of the rolling npx peak
FADEFREEZE = False        # freeze a/h at last-healthy values during fade
                          # (--fadefreeze, user-designed): shape cannot
                          # change just because a wall hides it
BLINDSPEED = False        # blind-phase speed decoupled from the belief
                          # (--blindspeed): fixed full speed toward the
                          # occlusion edge; safety layer still protects
MAPCLAMP = False          # map-constrained extrapolation (--mapclamp): during
                          # blind coasting, back-project the belief to a world
                          # point; a target cannot be inside a wall, so push
                          # the point out to body clearance and re-project
BODY_CLR = 20.0           # target body radius (cm) for the clamp
# corridor free-space strips (interior faces shrunk by BODY_CLR):
# the target can ONLY be in here -- clamping to "not inside a wall" (v2)
# wrongly pushed the phantom past walls it could never reach
CORRIDOR = [(-565.0, -455.0, -680.0, 375.0),   # south leg
            (-565.0, 880.0, 265.0, 375.0)]     # east leg
EDGEAIM = False           # blind-phase steering re-aim (--edgeaim): while no
                          # detection, yaw steers at the map-projected
                          # occlusion edge (where reappearance happens)
                          # instead of at the KF belief (behind the wall)
EDGEGEO = False           # geometric fade trigger: box touches occluder edge (--edgegeo)
EDGE_TOUCH = 6.0          # px tolerance for the geometric trigger
EDGEGEO_BEHAV = False     # let the geometric trigger also drive edgeaim/blindspeed
FADE_RSCALE = 1.0         # sigma multiplier for edge-reconstructed cx (--fader=N)
WMODEL = None             # width model for edgemeas: None | 'oracle' | 'est' (--wmodel=)
FADEMEAS = 'edge'         # fade-frame measurement: 'edge' (clean edge +- w_good/2) | 'template' (--fademeas=)
TMPL_MINFRAC = 0.4        # template: reject when visible mask < this fraction of the template
FADE_RAUTO = False        # sigma scale = (1/visible fraction)^2 on template frames (--fader=auto)
QVEL = None               # override KF velocity process-noise weight: std = h/QVEL (--qvel=N; recipe = 40)
TMPL_SHIFT = 120          # template: max |shift| searched (px)
WMODEL_COEF = None        # loaded from env_config/data/width_model.json


def width_model(speed, sin_phi, h):
    """Predicted full bbox width (px) of the target at the given speed (cm/s),
    |sin(heading vs line of sight)| and box height h.  Fitted on healthy
    frames (calib_width.py): aspect = c0 + c1*v + c2*s + c3*v*s."""
    global WMODEL_COEF
    if WMODEL_COEF is None:
        import json as _j
        WMODEL_COEF = _j.load(open(os.path.join(SCRATCH, '..', 'env_config', 'data', 'width_model.json')))
    c = WMODEL_COEF
    a = c['c0'] + c['c1'] * speed + c['c2'] * sin_phi + c['c3'] * speed * sin_phi
    return max(a, 0.15) * h
DEBUG_EDGE = None         # list -> per-frame edgemeas debug rows
DEBUG_IMG = None          # (dir, kmin, kmax) -> dump annotated frames
EDGEMEAS = False          # partial measurement during fade (--edgemeas):
                          # reconstruct the box center from the un-clipped
                          # edge (map-projected occluder column decides the
                          # side); cy/a/h take predicted values
CORNER_APEX = (-435.0, 245.0)   # the L-track's occluding vertical edge
DYN = False               # acceleration-first camera dynamics (--dyn):
                          # velocity is a state, |a| <= A_CAM, sqrt braking law
A_CAM = 200.0             # cm/s^2: camera accel/brake bound (2 m/s^2, Tello-class)
MARGIN_CAM = 30.0         # camera-point clearance. 15 = render limit (near-clip ~10.5);
                          # raised to 30 on 2026-09-21 = body radius + localisation slack
MARGIN_BODY = 5.0         # body point must stay outside walls
CAM_FWD = 30.0            # camera forward offset on the body
PITCH = -15.0

# L-track wall rectangles (x0, x1, y0, y1), engine-verified bounds
WALLS = [(-685.0, -585.0, -530.0, 70.0),    # cube1
         (-685.0, -585.0, 70.0, 395.0),     # cube5
         (-435.0, -335.0, -530.0, 145.0),   # cube2_7
         (-435.0, 40.0, 145.0, 245.0),      # cube4
         (-710.0, -35.0, 395.0, 495.0)]     # cube3
WALL_SPECS = {                               # name -> (cx, cy, sx, sy)
    'cube1':   (-635.0, -230.0, 1.0, 6.0),
    'cube5':   (-635.0, 232.5, 1.0, 3.25),
    'cube2_7': (-385.0, -192.5, 1.0, 6.75),
    'cube4':   (-197.5, 195.0, 4.75, 1.0),
    'cube3':   (-372.5, 445.0, 6.75, 1.0)}
WALL_SZ = 3.5                                # 350 cm tall
PARK = ['cylinder1', 'cylinder2', 'cylinder3', 'cylinder4', 'cylinder5',
        'cone1', 'cone2', 'cone3', 'cone4', 'cone5']

# ---------------- random route (topology-locked, occlusion guaranteed) ----
C_MIN, C_MAX = 40.0, 60.0
V_LO, V_HI = 40.0, 99.0
A_MAX = 80.0


def gen_route(rng):
    m = 40.0
    sx = lambda: rng.uniform(-585 + m, -435 - m)
    ey = lambda: rng.uniform(245 + m, 395 - m)
    pts = [(sx(), -450.0)]
    for y in np.sort(rng.uniform(-350, 120, rng.integers(2, 5))):
        pts.append((sx(), y))
    c = rng.uniform(C_MIN, C_MAX)
    pts.append((-435.0 - c, 245.0 + c))
    for x in np.sort(rng.uniform(-300, -50, rng.integers(2, 4))):
        pts.append((x, ey()))
    pts.append((60.0, ey()))
    return np.array(pts), c


def walk(route, rng, n_steps, sprint_s=None, rand_accel=False, sprint_a=A_MAX):
    """sprint_s: arc-length of the corner point; from there on the target
    sprints (accelerates at sprint_a to V_HI and holds) -- the P2 profile.
    rand_accel: draw a fresh ramp acceleration in [20, A_MAX] at every speed
    re-target instead of always ramping at the physical maximum."""
    seg = np.diff(route, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = seg_len.sum()
    v = rng.uniform(V_LO, V_HI)
    v_star = rng.uniform(V_LO, V_HI)
    t_next = rng.uniform(2, 4)
    a_ramp = A_MAX
    s, t = 0.0, 0.0
    S = []
    for _ in range(n_steps):
        if sprint_s is not None and s >= sprint_s:
            # occlusion-triggered maneuver: sprint (a>0) or brake (a<0);
            # speed floor 20 keeps the target moving toward the exit.
            v = float(np.clip(v + sprint_a * DT, 20.0, V_HI))
        else:
            if t >= t_next:
                v_star = rng.uniform(V_LO, V_HI)
                t_next = t + rng.uniform(2, 4)
                if rand_accel:
                    a_ramp = rng.uniform(0.0, A_MAX)   # only the max is fixed
            dv = np.clip(v_star - v, -a_ramp * DT, a_ramp * DT)
            v = np.clip(v + dv, V_LO, V_HI)
        s = min(s + v * DT, total)
        t += DT
        S.append(s)
    S = np.array(S)
    idx = np.clip(np.searchsorted(cum, S, side='right') - 1, 0, len(seg) - 1)
    frac = (S - cum[idx]) / np.maximum(seg_len[idx], 1e-9)
    P = route[idx] + seg[idx] * frac[:, None]
    yaw = np.degrees(np.arctan2(seg[idx, 1], seg[idx, 0]))
    return P, yaw


# ---------------- CorridorMap safety filter ----------------
def dist_to_walls(x, y):
    d = 1e9
    for (x0, x1, y0, y1) in WALLS:
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        d = min(d, math.hypot(dx, dy))
    return d


def wall_push_dir(x, y):
    """Outward unit normal from the wall nearest to point (x, y)."""
    best, nx, ny = 1e9, 0.0, 0.0
    for (x0, x1, y0, y1) in WALLS:
        cxp = min(max(x, x0), x1)
        cyp = min(max(y, y0), y1)
        dx, dy = x - cxp, y - cyp
        d = math.hypot(dx, dy)
        if 1e-9 < d < best:
            best, nx, ny = d, dx / d, dy / d
    return nx, ny


def safe_move(px, py, qx, qy, step):
    """One-step lookahead on the CAMERA point (the vehicle): accept,
    axis-slide, wall-normal push, or stay. Heading-independent -- the
    camera rotates in place with zero radius (inverse-mount bookkeeping:
    the pawn is re-placed 30 cm behind the camera every step)."""
    if dist_to_walls(qx, qy) >= MARGIN_CAM:
        return qx, qy, False
    if dist_to_walls(qx, py) >= MARGIN_CAM:
        return qx, py, True
    if dist_to_walls(px, qy) >= MARGIN_CAM:
        return px, qy, True
    nx, ny = wall_push_dir(px, py)
    ex, ey = px + max(step, 2.0) * nx, py + max(step, 2.0) * ny
    if dist_to_walls(ex, ey) > dist_to_walls(px, py):
        return ex, ey, True
    return px, py, True


# ---------------- controller ----------------
def vcap_step(x, y, vx, vy):
    """User-designed velocity limit, v2 (2026-09-09): cap only the
    wall-approach (normal) component of the velocity by the remaining
    clearance -- tangential speed is preserved, so wall-adjacent flight
    and blind-phase sliding progress are unaffected. v1 (total-speed cap
    along heading) stalled the tracker facing walls: killed, archived.
    Returns the possibly-modified (vx, vy) and whether it intervened."""
    d = dist_to_walls(x, y)
    if d > MARGIN_CAM + 60.0:            # far from any wall: free
        return vx, vy, False
    nx, ny = wall_push_dir(x, y)         # outward normal of nearest wall
    v_in = -(vx * nx + vy * ny)          # approach speed toward the wall
    cap = max(0.0, d - MARGIN_CAM) / VCAP_T
    if v_in <= cap:
        return vx, vy, False
    excess = v_in - cap
    return vx + excess * nx, vy + excess * ny, True


def vcap_sqrt(x, y, vx, vy):
    """Braking-distance speed limit (physics version of vcap): the
    wall-approach component may never exceed sqrt(2*A_CAM*(clearance-15)),
    i.e. the speed the bounded deceleration can absorb before the margin.
    Tangential component untouched."""
    d = dist_to_walls(x, y)
    if d > MARGIN_CAM + 80.0:
        return vx, vy, False
    nx, ny = wall_push_dir(x, y)
    v_in = -(vx * nx + vy * ny)
    cap = math.sqrt(2.0 * A_CAM * max(d - MARGIN_CAM, 0.0))
    if v_in <= cap:
        return vx, vy, False
    e = v_in - cap
    return vx + e * nx, vy + e * ny, True


def dyn_step(x, y, cvx, cvy, vdx, vdy):
    """Acceleration-first camera dynamics (rewritten 2026-09-28).
    State = actual velocity (cvx, cvy).  Per step the velocity may change
    by at most A_CAM*DT (inertia).  The wall-approach component u is
    additionally bounded by the discrete-safe braking law
        u <= sqrt((A*dt)^2 + 2*A*(d - margin)) - A*dt
    which is the exact condition that braking at A from the NEXT position
    stops before the margin (semi-implicit Euler, forward-invariant).
    Order matters: the braking bound is applied to the *integrated* velocity,
    not to the controller's wish, so the realised motion always satisfies it.
    Normal component gets the accel budget first, tangential takes the rest.
    Returns (vx, vy, braked)."""
    lim = A_CAM * DT
    d = dist_to_walls(x, y)
    nx, ny = wall_push_dir(x, y)                 # outward normal of nearest wall
    # decompose previous and desired velocity: u = toward wall, t = along wall
    tx_, ty_ = -ny, nx                           # tangent unit vector
    u0 = -(cvx * nx + cvy * ny); t0 = cvx * tx_ + cvy * ty_
    ud = -(vdx * nx + vdy * ny); td = vdx * tx_ + vdy * ty_
    s_ = max(d - MARGIN_CAM, 0.0)
    u_cap = math.sqrt(lim * lim + 2.0 * A_CAM * s_) - lim
    u_hi = min(u0 + lim, u_cap)
    u_lo = u0 - lim
    braked = ud > u_cap
    u1 = min(max(ud, u_lo), max(u_hi, u_lo))     # if cap < u0-lim: brake at max
    rem = math.sqrt(max(lim * lim - (u1 - u0) ** 2, 0.0))
    t1 = min(max(td, t0 - rem), t0 + rem)
    vx = -u1 * nx + t1 * tx_
    vy = -u1 * ny + t1 * ty_
    return vx, vy, braked


def edge_u(x, y, yaw_deg):
    """Pixel column of the occluding corner edge, from map + camera pose.
    None if the edge is outside the horizontal field of view."""
    bear = math.degrees(math.atan2(CORNER_APEX[1] - y, CORNER_APEX[0] - x))
    rel = (bear - yaw_deg + 180.0) % 360.0 - 180.0
    if abs(rel) > 55.0:
        return None
    return CX0 + FX * math.tan(math.radians(rel))


def map_clamp(mean, cam_x, cam_y, yaw_deg):
    """Clamp the blind-phase belief to physically reachable space:
    back-project (cx, h) -> world target point; if it sits inside a wall
    (closer than BODY_CLR), push it out along the wall normal; re-project
    the corrected point back into (cx, h). Velocities left untouched."""
    ang = math.degrees(math.atan2(mean[0] - CX0, FX))
    rel = math.radians(yaw_deg + ang)
    rho = FX * H_PERSON / max(mean[3], 1.0)
    xt = cam_x + rho * math.cos(rel)
    yt = cam_y + rho * math.sin(rel)
    # v3: the target can only be inside the corridor free space, so
    # project the phantom onto the nearest point of the corridor strips.
    # (v1 pushed to the nearest wall face -> wrong near side, retired;
    #  v2 ray-marched through ANY wall -> escaped past unreachable
    #  walls like the north end wall, retired.)
    best, bx_, by_ = 1e9, xt, yt
    for (x0, x1, y0, y1) in CORRIDOR:
        cxp = min(max(xt, x0), x1)
        cyp = min(max(yt, y0), y1)
        d = math.hypot(xt - cxp, yt - cyp)
        if d < best:
            best, bx_, by_ = d, cxp, cyp
    if best < 1.0:                           # already inside the corridor
        return mean, False
    # re-project the corrected world point into the image
    bear = math.degrees(math.atan2(by_ - cam_y, bx_ - cam_x))
    rel2 = (bear - yaw_deg + 180.0) % 360.0 - 180.0
    rho2 = math.hypot(bx_ - cam_x, by_ - cam_y)
    if abs(rel2) < 80.0 and rho2 > 30.0:
        mean[0] = CX0 + FX * math.tan(math.radians(rel2))
        mean[3] = FX * H_PERSON / rho2
    return mean, True


def control(cx, h):
    ang = math.degrees(math.atan2(cx - CX0, FX))
    w = float(np.clip(3.0 * ang, -W_CAM_MAX, W_CAM_MAX))
    rho_est = FX * H_PERSON / max(h, 1.0)
    v = float(np.clip(2.0 * (rho_est - D_SET), 0.0, V_CAM_MAX))
    return v, w


# ---------------- engine setup ----------------
def boot():
    runner = env_unreal.RunUnreal(
        ENV_BIN='FlexibleRoom_Win_v0\\FlexibleRoom\\Binaries\\Win64\\FlexibleRoom.exe')
    ip, port = runner.start(docker=False, resolution=(640, 480))
    ucv = unrealcv_basic.UnrealCv(port=port, ip=ip, env=runner.path2env,
                                  cam_id=1, resolution=(640, 480))
    # build L walls
    for name, (cx, cy, sx, sy) in WALL_SPECS.items():
        ucv.set_obj_scale(name, [sx, sy, WALL_SZ])
        ucv.set_obj_location(name, [cx, cy, 100])
    # park unused blocks far away
    for i, name in enumerate(PARK):
        ucv.set_obj_location(name, [800, -1300 + i * 120, 100])
    # target: scale 0.85, red mask color
    ucv.set_obj_scale('targetbp2', [0.85, 0.85, 0.85])
    ucv.set_obj_color('targetbp2', [255, 0, 0])
    ucv.cam[1]['fov'] = None
    ucv.set_fov(1, 82.6)
    time.sleep(1.0)
    z_trk = ucv.get_obj_location('targetbp')[2]
    print(f'[boot] tracker z = {z_trk:.1f}, fov readback = {ucv.get_fov(1)}')
    return runner, ucv, z_trk


# ---------------- one episode ----------------
def run_episode(ucv, z_trk, mode, seed, profile='rand', steplog=None):
    rng = np.random.default_rng(seed)
    route, c = gen_route(rng)
    sprint_s, sprint_a = None, A_MAX
    if profile in ('p2', 'p2a'):
        corner = np.array([-435.0 - c, 245.0 + c])
        idx = int(np.argmin(np.linalg.norm(route - corner, axis=1)))
        seg_len = np.linalg.norm(np.diff(route, axis=0), axis=1)
        sprint_s = float(seg_len[:idx].sum())
        if profile == 'p2a':
            # corner maneuver accel: only |a| <= A_MAX is fixed; sign free
            # (positive = sprint, negative = brake behind the wall)
            sprint_a = float(rng.uniform(-A_MAX, A_MAX))
    P, YAW = walk(route, rng, N_STEPS, sprint_s=sprint_s,
                  rand_accel=(profile in ('randa', 'p2a')), sprint_a=sprint_a)

    # tracker starts 250 cm behind the entry, aligned with first segment
    d0 = route[1] - route[0]
    u0 = d0 / np.linalg.norm(d0)
    tx, ty = route[0] - u0 * D_SET
    tyaw = math.degrees(math.atan2(u0[1], u0[0]))

    def place_tracker():
        # inverse mount: (tx, ty) is the CAMERA point; the pawn trails 30 cm
        # behind it so the rigid +30 attachment lands the camera exactly there.
        hx = math.cos(math.radians(tyaw))
        hy = math.sin(math.radians(tyaw))
        ucv.set_obj_location('targetbp', [tx - CAM_FWD * hx, ty - CAM_FWD * hy, z_trk])
        ucv.set_obj_rotation('targetbp', [0, tyaw, 0])
        ucv.set_rotation(1, [0, tyaw, PITCH])

    ucv.set_obj_location('targetbp2', [P[0, 0], P[0, 1], 183.6])
    ucv.set_obj_rotation('targetbp2', [0, YAW[0], 0])
    place_tracker()
    time.sleep(0.3)

    kf = None
    if mode == 'cv':
        kf = KalmanFilter()
    elif mode == 'ca':
        kf = KalmanFilterCA()
    elif mode == 'cv_r01':
        kf = KFTightR()
    elif mode == 'ca_r01':
        kf = KFCATightR()
    if kf is not None and QVEL is not None:
        kf._std_weight_velocity = 1.0 / QVEL      # physics-anchored velocity process noise
    mean = cov = None
    hlock = HLock() if (kf is not None and HLOCK) else None
    last_bbox = None          # (cx, cy, a, h) for mode 'none'
    det_visible = False       # hysteresis state on mask pixels
    blind_run = 0
    reacq_err = None
    reacq_cm = reacq_dr = None
    gate_rej = 0              # consecutive gate rejections (valid det present)
    n_gated = 0
    n_reinit = 0

    AR, streak = 0.0, 0
    v_cmd = 0.0
    n_hit = 0
    n_cap = 0
    cvx = cvy = 0.0           # camera velocity state (used in --dyn mode)
    npx_win = []              # last-8 healthy-frame pixel counts (fade trigger)
    tmpl = None               # (column profile, box centre cx, mask px) of last un-clipped frame
    w_good = None             # last healthy full-body pixel width (a*h)
    a_good = h_good = None    # last healthy aspect/height (fade-freeze)
    n_fade = 0
    n_clamp = 0
    occl_steps = 0
    for k in range(N_STEPS):
        # --- advance target (ground truth motion) ---
        ucv.set_obj_location('targetbp2', [P[k, 0], P[k, 1], 183.6])
        ucv.set_obj_rotation('targetbp2', [0, YAW[k], 0])

        # --- oracle detection from object mask ---
        img = ucv.read_image(1, 'object_mask', 'fast')
        mask, box = ucv.get_bbox(img, 'targetbp2', normalize=False)
        npx = cv2.countNonZero(mask)
        if DEBUG_IMG is not None and DEBUG_IMG[1] <= k <= DEBUG_IMG[2]:
            lit = ucv.read_image(1, 'lit', 'fast').copy()
            bear = math.degrees(math.atan2(P[k, 1] - ty, P[k, 0] - tx))
            rel = (bear - tyaw + 180.0) % 360.0 - 180.0
            u_true = int(CX0 + FX * math.tan(math.radians(rel)))
            ue = edge_u(tx, ty, tyaw)
            bx_, by_, bw_, bh_ = [int(v) for v in box]
            cv2.rectangle(lit, (bx_, by_), (bx_ + bw_, by_ + bh_), (0, 255, 0), 1)
            cv2.line(lit, (u_true, 0), (u_true, 479), (0, 0, 255), 1)
            if ue is not None:
                cv2.line(lit, (int(ue), 0), (int(ue), 479), (255, 0, 255), 1)
            cv2.putText(lit, f'k={k} box[{bx_},{bx_+bw_}] w={bw_} true_cx={u_true} edge={ue if ue is None else int(ue)} npx={npx}',
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.line(m3, (u_true, 0), (u_true, 479), (0, 0, 255), 1)
            cv2.imwrite(os.path.join(DEBUG_IMG[0], f'frame_{k:03d}.png'), np.hstack([lit, m3]))
        if det_visible:
            det_ok = npx >= 20
        else:
            det_ok = npx > 50
        det_visible = det_ok
        if not det_ok:
            occl_steps += 1

        meas = None
        if det_ok and box[2] > 0 and box[3] > 0:
            bx, by, bw, bh = box
            meas = np.array([bx + bw / 2, by + bh / 2, bw / max(bh, 1), bh],
                            dtype=float)
        faded_now = False
        fade_meas = False
        fade_frac = 1.0
        if meas is not None and (FADEGATE or EDGEMEAS):
            # physical partial-occlusion trigger: pixel count collapsing
            # vs the last few healthy frames = the wall is clipping NOW.
            # (an 8-frame window is immune to the slow npx drift caused by
            # distance/aspect changes -- the long-decay-peak version of this
            # trigger mislabeled normal frames and is retired)
            ref = max(npx_win) if npx_win else 0
            # geometric trigger (2026-09-28, --edgegeo): the box already
            # overlaps the occluder-edge column and the target is beyond the
            # apex -> the wall is clipping it NOW, whatever the pixel count.
            # Closes the 100%->50% visibility band that the ratio trigger
            # misses (KF lagged ~30 px / vx -35% at blind onset).
            geo = False
            if EDGEGEO and EDGEMEAS and mean is not None and w_good is not None:
                ug = edge_u(tx, ty, tyaw)
                if ug is not None:
                    bl_, br_ = box[0], box[0] + box[2]
                    rho_t = FX * H_PERSON / max(h_good if h_good else mean[3], 1.0)
                    rho_a = math.hypot(CORNER_APEX[0] - tx, CORNER_APEX[1] - ty)
                    geo = (bl_ - EDGE_TOUCH <= ug <= br_ + EDGE_TOUCH) and rho_t > rho_a
            ratio_hit = ref > 0 and npx < FADE_FRAC * ref
            if ratio_hit or geo:
                n_fade += 1
                faded_now = ratio_hit or (geo and EDGEGEO_BEHAV)
                if EDGEMEAS and mean is not None and w_good is not None:
                    u = edge_u(tx, ty, tyaw)
                    if u is None:
                        meas = None
                    else:
                        bl, br = box[0], box[0] + box[2]
                        right_clipped = abs(br - u) <= abs(bl - u)
                        w_use = w_good
                        use_tmpl = FADEMEAS == 'template' and tmpl is not None
                        if use_tmpl:
                            # match the visible column profile of the mask against the
                            # last un-clipped silhouette (2026-09-28): unbiased while the
                            # torso is visible; reject (coast) once too little is left
                            pT, cxT, npxT = tmpl
                            pc = (mask > 0).sum(axis=0).astype(np.float32)
                            cols = np.arange(pc.shape[0])
                            vis = (cols < u - 2) if right_clipped else (cols > u + 2)
                            frac = float(pc[vis].sum()) / max(npxT, 1.0)
                            fade_frac = frac
                            if frac < TMPL_MINFRAC:
                                meas = None
                            else:
                                best, bd = None, 0
                                for d in range(-TMPL_SHIFT, TMPL_SHIFT + 1):
                                    e = float(np.sum((pc[vis] - np.roll(pT, d)[vis]) ** 2))
                                    if best is None or e < best:
                                        best, bd = e, d
                                meas = np.array([cxT + bd, mean[1], mean[2], mean[3]])
                                fade_meas = True
                            if DEBUG_EDGE is not None:
                                DEBUG_EDGE.append((k, bl, br, u, w_good, meas[0] if meas is not None else np.nan, npx, ratio_hit, geo))
                        if use_tmpl:
                            pass
                        elif WMODEL == 'oracle':
                            # ceiling test: true target speed & heading from the engine
                            vt = (P[k] - P[k - 1]) / DT if k > 0 else np.zeros(2)
                            los = P[k, :2] - np.array([tx, ty])
                            spd = float(np.hypot(vt[0], vt[1]))
                            ph = math.atan2(vt[1], vt[0]) - math.atan2(los[1], los[0])
                            w_use = width_model(spd, abs(math.sin(ph)), h_good if h_good else mean[3])
                        elif WMODEL == 'est':
                            w_use = width_model(est_speed, est_sphi, h_good if h_good else mean[3])
                        if not use_tmpl:
                            if right_clipped:
                                cxc = bl + w_use / 2.0    # right side clipped
                            else:
                                cxc = br - w_use / 2.0    # left side clipped
                            # only cx carries innovation; cy/a/h = predicted
                            meas = np.array([cxc, mean[1], mean[2], mean[3]])
                            fade_meas = True
                            if DEBUG_EDGE is not None:
                                DEBUG_EDGE.append((k, bl, br, u, w_good, cxc, npx, ratio_hit, geo))
                else:
                    meas = None                       # plain rejection
            else:
                if DEBUG_EDGE is not None:
                    DEBUG_EDGE.append((k, box[0], box[0] + box[2], -1.0, w_good if w_good else -1.0,
                                       box[0] + box[2] / 2.0, npx, False, False))
                if FADEMEAS == 'template':
                    ut_ = edge_u(tx, ty, tyaw)
                    if ut_ is None or box[0] > ut_ + 3 or box[0] + box[2] < ut_ - 3:
                        tmpl = ((mask > 0).sum(axis=0).astype(np.float32), box[0] + box[2] / 2.0, float(npx))
                npx_win.append(npx)
                if len(npx_win) > 8:
                    npx_win.pop(0)
                if mean is not None:
                    w_good = mean[2] * mean[3]        # healthy full width
                    a_good, h_good = mean[2], mean[3]
                else:
                    w_good = meas[2] * meas[3]
                    a_good, h_good = meas[2], meas[3]

        # --- state estimation ---
        if kf is not None:
            if meas is not None and mean is None:
                mean, cov = kf.initiate(meas)
            elif mean is not None:
                if mode.startswith('ca'):
                    mean, cov = kf.predict(mean, cov, coasting=(meas is None))
                else:
                    mean, cov = kf.predict(mean, cov)
                if meas is not None and GATE:
                    if gate_d2(mean, cov, meas) > GATE_CHI2:
                        gate_rej += 1
                        n_gated += 1
                        # re-initiate only on a full-looking target (never
                        # on fade-out fragments -- the h-lock lesson):
                        if gate_rej >= GATE_REINIT and npx >= 2000:
                            if blind_run >= 5 and reacq_err is None:
                                reacq_err = abs(mean[0] - meas[0])
                                reacq_cm, reacq_dr = world_reacq(mean[0], mean[3], tx, ty, tyaw, P[k, 0], P[k, 1])
                            mean, cov = kf.initiate(meas)   # track reset
                            n_reinit += 1
                            gate_rej = 0
                        else:
                            meas = None                     # coast this frame
                    else:
                        gate_rej = 0
                if meas is not None:
                    if blind_run >= 5 and reacq_err is None:
                        reacq_err = abs(mean[0] - meas[0])
                        reacq_cm, reacq_dr = world_reacq(mean[0], mean[3], tx, ty, tyaw, P[k, 0], P[k, 1])
                    if fade_meas and (FADE_RSCALE != 1.0 or FADE_RAUTO):
                        sc = min((1.0 / max(fade_frac, 0.05)) ** 2, 10.0) if FADE_RAUTO else FADE_RSCALE
                        mean, cov = kf_update_rscaled(kf, mean, cov, meas, sc)
                    else:
                        mean, cov = kf.update(mean, cov, meas)
            if hlock is not None:
                mean = hlock.apply(mean, meas)
            if FADEFREEZE and faded_now and mean is not None and h_good is not None:
                # user-designed (2026-09-16): body shape cannot change just
                # because a wall hides it -- pin a/h at the last healthy
                # values and zero their velocities until reappearance
                mean[2], mean[3] = a_good, h_good
                mean[6], mean[7] = 0.0, 0.0
            if MAPCLAMP and meas is None and mean is not None:
                mean, clamped = map_clamp(mean, tx, ty, tyaw)
                n_clamp += int(clamped)
            bbox_for_pid = mean[:4] if mean is not None else None
        else:
            if meas is not None:
                if blind_run >= 5 and reacq_err is None and last_bbox is not None:
                    reacq_err = abs(last_bbox[0] - meas[0])
                    reacq_cm, reacq_dr = world_reacq(last_bbox[0], last_bbox[3], tx, ty, tyaw, P[k, 0], P[k, 1])
                last_bbox = meas
            bbox_for_pid = last_bbox
        blind_run = 0 if meas is not None else blind_run + 1

        # --- control + safety filter + kinematics ---
        if bbox_for_pid is not None:
            cx_aim = bbox_for_pid[0]
            if EDGEAIM and (meas is None or faded_now):
                # blind OR fading: aim the yaw at the occlusion edge, not
                # the belief (the target vanishes and reappears at the edge)
                u = edge_u(tx, ty, tyaw)
                if u is not None:
                    cx_aim = u
            v_cmd, w_cmd = control(cx_aim, bbox_for_pid[3])
            if BLINDSPEED and (meas is None or faded_now):
                v_cmd = V_CAM_MAX     # speed decoupled from the belief too
        else:
            w_cmd = 0.0          # never seen anything yet
        tyaw += w_cmd * DT
        hx = math.cos(math.radians(tyaw))
        hy = math.sin(math.radians(tyaw))
        vx, vy = v_cmd * hx, v_cmd * hy
        if DYN:
            # acceleration-first dynamics (v2, 2026-09-28): integrate the
            # controller's wish under |dv| <= A_CAM*DT, with the braking
            # law enforced on the realised velocity (see dyn_step)
            cvx, cvy, capped = dyn_step(tx, ty, cvx, cvy, vx, vy)
            n_cap += int(capped)
            vx, vy = cvx, cvy
        elif VCAP:
            vx, vy, capped = vcap_step(tx, ty, vx, vy)
            n_cap += int(capped)
        step_len = math.hypot(vx, vy) * DT
        px0, py0 = tx, ty
        qx, qy = tx + vx * DT, ty + vy * DT
        tx, ty, hit = safe_move(tx, ty, qx, qy, step_len)
        n_hit += int(hit)
        if DYN:
            # sync the velocity state to the realized motion (wall contact
            # absorbs momentum)
            cvx, cvy = (tx - px0) / DT, (ty - py0) / DT
        place_tracker()
        if steplog is not None:
            steplog.append((tx, ty, tyaw, P[k, 0], P[k, 1], det_ok,
                            mean[0] if mean is not None else 320.0,
                            mean[3] if mean is not None else 1.0,
                            mean is not None))

        # --- protocol scoring (geometric, from ground truth poses) ---
        gx, gy = P[k]
        rho = math.hypot(gx - tx, gy - ty)
        bearing = (math.degrees(math.atan2(gy - ty, gx - tx)) - tyaw + 180) % 360 - 180
        visible = (rho <= 750.0) and (abs(bearing) <= 45.0)
        r = 1.0 - abs(rho - D_SET) / 750.0 - abs(bearing) / 45.0
        AR += max(min(r, 1.0), -1.0)
        streak = 0 if visible else streak + 1
        if streak >= KILL_STREAK:
            return dict(mode=mode, profile=profile, seed=seed, c=round(c, 1),
                        sa=round(sprint_a, 1) if profile == 'p2a' else None,
                        AR=round(AR, 1), EL=k + 1, SR=0, occl=occl_steps,
                        hits=n_hit, caps=n_cap, gated=n_gated, reinit=n_reinit,
                        fade=n_fade, clamp=n_clamp,
                        reacq=None if reacq_err is None else round(reacq_err, 1),
                reacq_cm=None if reacq_cm is None else round(reacq_cm, 1),
                reacq_dr=None if reacq_dr is None else round(reacq_dr, 1))
    return dict(mode=mode, profile=profile, seed=seed, c=round(c, 1),
                sa=round(sprint_a, 1) if profile == 'p2a' else None,
                AR=round(AR, 1), EL=N_STEPS, SR=int(visible), occl=occl_steps,
                hits=n_hit, caps=n_cap, gated=n_gated, reinit=n_reinit,
                fade=n_fade, clamp=n_clamp,
                reacq=None if reacq_err is None else round(reacq_err, 1),
                reacq_cm=None if reacq_cm is None else round(reacq_cm, 1),
                reacq_dr=None if reacq_dr is None else round(reacq_dr, 1))


def main():
    runner, ucv, z_trk = boot()
    seeds = list(range(2000, 2020))
    args = sys.argv[1:]
    global HLOCK, GATE, VCAP, DYN, FADEGATE
    HLOCK = '--hlock' in args
    GATE = '--gate' in args
    VCAP = '--vcap' in args
    DYN = '--dyn' in args
    FADEGATE = '--fadegate' in args
    global EDGEMEAS, EDGEAIM, MAPCLAMP, BLINDSPEED
    EDGEMEAS = '--edgemeas' in args
    EDGEAIM = '--edgeaim' in args
    MAPCLAMP = '--mapclamp' in args
    BLINDSPEED = '--blindspeed' in args
    global FADEFREEZE, EDGEGEO
    FADEFREEZE = '--fadefreeze' in args
    EDGEGEO = '--edgegeo' in args
    global A_CAM, FADE_RSCALE, WMODEL, FADEMEAS, FADE_RAUTO, TMPL_MINFRAC, QVEL
    for a in args:
        if a.startswith('--seeds='):
            seeds = seeds[:int(a.split('=')[1])]
        if a.startswith('--acc='):
            A_CAM = float(a.split('=')[1])
        if a.startswith('--fader='):
            if a.endswith('auto'):
                FADE_RAUTO = True
            else:
                FADE_RSCALE = float(a.split('=')[1])
        if a.startswith('--tmplmin='):
            TMPL_MINFRAC = float(a.split('=')[1])
        if a.startswith('--qvel='):
            QVEL = float(a.split('=')[1])
        if a.startswith('--wmodel='):
            WMODEL = a.split('=')[1]
        if a.startswith('--fademeas='):
            FADEMEAS = a.split('=')[1]
    modes = [a for a in args if not a.startswith('@') and not a.startswith('--')] \
        or ['none', 'cv', 'ca', 'ca_r01']
    profiles = [a[1:] for a in args if a.startswith('@')] or ['rand', 'p2']
    results = []
    try:
        for profile in profiles:
            for mode in modes:
                for seed in seeds:
                    t0 = time.time()
                    res = run_episode(ucv, z_trk, mode, seed, profile)
                    res['wall_s'] = round(time.time() - t0, 1)
                    results.append(res)
                    print('[ep]', json.dumps(res))
                    sys.stdout.flush()
    finally:
        out = os.path.join(SCRATCH, 'metrics_smoke.json')
        for a in sys.argv[1:]:
            if a.startswith('--out='):
                out = a.split('=', 1)[1]
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        print('saved:', out)
        try:
            runner.close()
        except Exception:
            pass
    # summary
    for profile in profiles:
        for mode in modes:
            rs = [r for r in results if r['mode'] == mode and r['profile'] == profile]
            if rs:
                rq = [r['reacq'] for r in rs if r['reacq'] is not None]
                print(f"{profile:4s} {mode:7s}  AR={np.mean([r['AR'] for r in rs]):7.1f}  "
                      f"EL={np.mean([r['EL'] for r in rs]):5.1f}  "
                      f"SR={np.mean([r['SR'] for r in rs]):.2f}  "
                      f"reacq={np.mean(rq) if rq else float('nan'):6.1f} (n={len(rq)})")


if __name__ == '__main__':
    main()
