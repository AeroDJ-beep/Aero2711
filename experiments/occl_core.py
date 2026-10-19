# -*- coding: utf-8 -*-
"""Engine-independent core of the occlusion-tracking pipeline (ported from
run_metrics_test.py so it can run on UnrealZoo scenes):

  * KF factories (cv / ca, plain or tight-R)         make_kf(mode)
  * P controller                                     control(cx, h)
  * safety layer: inertia + braking law (dyn_step),  hard veto / slide / push (safe_move)
  * fade-frame measurement (edge reconstruction)     edge_measurement(...)
  * scaled-R update, world-space re-acquisition error
Geometry is injected through a `geo` object exposing clearance(x, y) -> cm and
push_dir(x, y) -> outward unit normal, so any map representation works.
"""
import os, sys, math
import numpy as np
import scipy.linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'OA-VAT'))
from kalman_filter import KalmanFilter          # noqa: E402
from kalman_filter_ca import KalmanFilterCA     # noqa: E402

# camera / control constants (identical to the L-map setup)
FX, CX0, CY0 = 364.3, 320.0, 240.0
H_PERSON = 168.0
D_SET, V_CAM_MAX, W_CAM_MAX = 250.0, 99.0, 30.0
A_CAM, MARGIN = 200.0, 5.0            # cm/s^2 accel bound; camera-point clearance beyond the (already inset) navmesh free space
FADE_FRAC = 0.5                       # fade trigger: npx < FADE_FRAC * recent healthy peak


class TightRMixin:
    R_SCALE = 0.1

    def project(self, mean, covariance):
        rs = self.R_SCALE
        std = [rs * self._std_weight_position * mean[3], rs * self._std_weight_position * mean[3],
               rs * 1e-1, rs * self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean_p = np.dot(self._update_mat, mean)
        cov_p = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean_p, cov_p + innovation_cov


class KFTightR(TightRMixin, KalmanFilter):
    pass


class KFCATightR(TightRMixin, KalmanFilterCA):
    pass


def make_kf(mode, qvel=None):
    kf = {'cv': KalmanFilter, 'ca': KalmanFilterCA, 'cv_r01': KFTightR, 'ca_r01': KFCATightR}.get(mode)
    kf = kf() if kf else None
    if kf is not None and qvel:
        kf._std_weight_velocity = 1.0 / qvel
    return kf


W_MAZE, V_BACK = 60.0, 50.0          # maze controller: faster yaw, allow backing off when too close
MAZE_CTRL = False


def control(cx, h):
    ang = math.degrees(math.atan2(cx - CX0, FX))
    wmax = W_MAZE if MAZE_CTRL else W_CAM_MAX
    w = float(np.clip(3.0 * ang, -wmax, wmax))
    rho = FX * H_PERSON / max(h, 1.0)
    vmin = -V_BACK if MAZE_CTRL else 0.0
    v = float(np.clip(2.0 * (rho - D_SET), vmin, V_CAM_MAX))
    return v, w


def dyn_step(geo, x, y, cvx, cvy, vdx, vdy, dt, a_cam=A_CAM, margin=MARGIN):
    """Acceleration-first dynamics: |dv| <= A*dt, wall-approach component bounded
    by the discrete-safe braking law u <= sqrt((A dt)^2 + 2 A (d - margin)) - A dt."""
    lim = a_cam * dt
    d = geo.clearance(x, y)
    nx, ny = geo.push_dir(x, y)
    tx_, ty_ = -ny, nx
    u0 = -(cvx * nx + cvy * ny); t0 = cvx * tx_ + cvy * ty_
    ud = -(vdx * nx + vdy * ny); td = vdx * tx_ + vdy * ty_
    s_ = max(d - margin, 0.0)
    u_cap = math.sqrt(lim * lim + 2.0 * a_cam * s_) - lim
    u_hi = min(u0 + lim, u_cap); u_lo = u0 - lim
    braked = ud > u_cap
    u1 = min(max(ud, u_lo), max(u_hi, u_lo))
    rem = math.sqrt(max(lim * lim - (u1 - u0) ** 2, 0.0))
    t1 = min(max(td, t0 - rem), t0 + rem)
    return -u1 * nx + t1 * tx_, -u1 * ny + t1 * ty_, braked


def safe_move(geo, px, py, qx, qy, step, margin=MARGIN):
    """One-step lookahead on the camera point: accept, axis-slide, push, or stay."""
    dq = geo.clearance(qx, qy)
    if dq >= margin:
        return qx, qy, False
    dp = geo.clearance(px, py)
    if dp < margin and dq > dp:
        return qx, qy, True                 # already inside the margin: any move that gains clearance is allowed
    if geo.clearance(qx, py) >= margin:
        return qx, py, True
    if geo.clearance(px, qy) >= margin:
        return px, qy, True
    nx, ny = geo.push_dir(px, py)
    ex, ey = px + max(step, 2.0) * nx, py + max(step, 2.0) * ny
    if geo.clearance(ex, ey) > geo.clearance(px, py):
        return ex, ey, True
    return px, py, True


def kf_update_rscaled(kf, mean, cov, meas, sig_scale):
    Hm = kf._update_mat
    pm, pc = kf.project(mean, cov)
    HPH = np.linalg.multi_dot((Hm, cov, Hm.T))
    R = pc - HPH
    pc2 = HPH + (sig_scale ** 2) * R
    chol, lower = scipy.linalg.cho_factor(pc2, lower=True, check_finite=False)
    K = scipy.linalg.cho_solve((chol, lower), np.dot(cov, Hm.T).T, check_finite=False).T
    y = meas - pm
    return mean + np.dot(y, K.T), cov - np.linalg.multi_dot((K, pc2, K.T))


def gate_d2(kf, mean, cov, meas):
    std = [mean[3] / 20.0, mean[3] / 20.0, 1e-1, mean[3] / 20.0]
    Hm = kf._update_mat
    S = np.linalg.multi_dot((Hm, cov, Hm.T)) + np.diag(np.square(std))
    y = meas - np.dot(Hm, mean)
    return float(y @ np.linalg.solve(S, y))


def edge_measurement(box, u, w_good, mean):
    """Reconstruct the target centre from the un-clipped box edge and the last
    healthy width; only cx carries innovation (cy, a, h = predicted)."""
    bl, br = box[0], box[0] + box[2]
    if abs(br - u) <= abs(bl - u):
        cxc = bl + w_good / 2.0          # right side clipped
    else:
        cxc = br - w_good / 2.0          # left side clipped
    return np.array([cxc, mean[1], mean[2], mean[3]], float)


def back_project(cx, h, tx, ty, yaw):
    ang = math.degrees(math.atan2(cx - CX0, FX))
    b = math.radians(yaw + ang)
    rho = min(FX * H_PERSON / max(h, 1.0), 900.0)
    return tx + rho * math.cos(b), ty + rho * math.sin(b)


def re_project(px, py, tx, ty, yaw):
    bear = math.degrees(math.atan2(py - ty, px - tx))
    rel = (bear - yaw + 180.0) % 360.0 - 180.0
    rho = math.hypot(px - tx, py - ty)
    if abs(rel) < 80.0 and rho > 30.0:
        return CX0 + FX * math.tan(math.radians(rel)), FX * H_PERSON / rho
    return None


def world_reacq(cx_est, h_est, tx, ty, yaw, gx, gy):
    px, py = back_project(cx_est, h_est, tx, ty, yaw)
    rng = math.hypot(gx - tx, gy - ty)
    rho = min(FX * H_PERSON / max(h_est, 1.0), 900.0)
    return math.hypot(px - gx, py - gy), 100.0 * (rho - rng) / max(rng, 1.0)


def score_step(tx, ty, yaw, gx, gy):
    rho = math.hypot(gx - tx, gy - ty)
    bearing = (math.degrees(math.atan2(gy - ty, gx - tx)) - yaw + 180.0) % 360.0 - 180.0
    visible = (rho <= 750.0) and (abs(bearing) <= 45.0)
    r = 1.0 - abs(rho - D_SET) / 750.0 - abs(bearing) / 45.0
    return max(min(r, 1.0), -1.0), visible, rho, bearing
