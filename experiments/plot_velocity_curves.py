# -*- coding: utf-8 -*-
"""Seed-2005: record kinematic-vcap vs dynamic (--dyn) episodes, plot
UAV velocity decomposed into x/y components + trajectory + clearance."""
import os, sys, time, math
import numpy as np

SCRATCH_OLD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH_OLD)
import run_metrics_test as M
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.collections import LineCollection

SEED = 2005
OUT = r'D:\Aero2711\results\figures\velocity_curves_seed2005.png'


def episode(ucv, z_trk, dyn):
    rng = np.random.default_rng(SEED)
    route, c = M.gen_route(rng)
    corner = np.array([-435.0 - c, 245.0 + c])
    idx = int(np.argmin(np.linalg.norm(route - corner, axis=1)))
    seg_len = np.linalg.norm(np.diff(route, axis=0), axis=1)
    sprint_s = float(seg_len[:idx].sum())
    sprint_a = float(rng.uniform(-M.A_MAX, M.A_MAX))
    P, YAW = M.walk(route, rng, M.N_STEPS, sprint_s=sprint_s,
                    rand_accel=True, sprint_a=sprint_a)
    d0 = route[1] - route[0]
    u0 = d0 / np.linalg.norm(d0)
    tx, ty = route[0] - u0 * M.D_SET
    tyaw = math.degrees(math.atan2(u0[1], u0[0]))

    def place():
        hx, hy = math.cos(math.radians(tyaw)), math.sin(math.radians(tyaw))
        ucv.set_obj_location('targetbp', [tx - M.CAM_FWD * hx, ty - M.CAM_FWD * hy, z_trk])
        ucv.set_obj_rotation('targetbp', [0, tyaw, 0])
        ucv.set_rotation(1, [0, tyaw, M.PITCH])

    ucv.set_obj_location('targetbp2', [P[0, 0], P[0, 1], 183.6])
    ucv.set_obj_rotation('targetbp2', [0, YAW[0], 0])
    place()
    time.sleep(0.3)

    kf = M.KFCATightR()
    mean = cov = None
    det_visible = False
    v_cmd = 0.0
    cvx = cvy = 0.0
    L = {k: [] for k in ['cam', 'det', 'vx', 'vy', 'vdes', 'clr']}
    for k in range(M.N_STEPS):
        ucv.set_obj_location('targetbp2', [P[k, 0], P[k, 1], 183.6])
        ucv.set_obj_rotation('targetbp2', [0, YAW[k], 0])
        img = ucv.read_image(1, 'object_mask', 'fast')
        mask, box = ucv.get_bbox(img, 'targetbp2', normalize=False)
        npx = cv2.countNonZero(mask)
        det_ok = (npx >= 20) if det_visible else (npx > 50)
        det_visible = det_ok
        meas = None
        if det_ok and box[2] > 0 and box[3] > 0:
            bx, by, bw, bh = box
            meas = np.array([bx + bw / 2, by + bh / 2, bw / max(bh, 1), bh], dtype=float)
        if meas is not None and mean is None:
            mean, cov = kf.initiate(meas)
        elif mean is not None:
            mean, cov = kf.predict(mean, cov, coasting=(meas is None))
            if meas is not None:
                mean, cov = kf.update(mean, cov, meas)
        if mean is not None:
            v_cmd, w_cmd = M.control(mean[0], mean[3])
        else:
            w_cmd = 0.0
        tyaw += w_cmd * M.DT
        hx, hy = math.cos(math.radians(tyaw)), math.sin(math.radians(tyaw))
        vx, vy = v_cmd * hx, v_cmd * hy
        if dyn:
            vx, vy, _ = M.vcap_sqrt(tx, ty, vx, vy)
            dvx, dvy = vx - cvx, vy - cvy
            dn = math.hypot(dvx, dvy)
            lim = M.A_CAM * M.DT
            if dn > lim:
                dvx, dvy = dvx * lim / dn, dvy * lim / dn
            cvx, cvy = cvx + dvx, cvy + dvy
            vx, vy = cvx, cvy
        else:
            vx, vy, _ = M.vcap_step(tx, ty, vx, vy)
        px0, py0 = tx, ty
        qx, qy = tx + vx * M.DT, ty + vy * M.DT
        tx, ty, _ = M.safe_move(tx, ty, qx, qy, math.hypot(vx, vy) * M.DT)
        if dyn:
            cvx, cvy = (tx - px0) / M.DT, (ty - py0) / M.DT
        place()
        L['cam'].append((tx, ty))
        L['det'].append(det_ok)
        L['vx'].append((tx - px0) / M.DT)     # realized velocity components
        L['vy'].append((ty - py0) / M.DT)
        L['vdes'].append(v_cmd)
        L['clr'].append(M.dist_to_walls(tx, ty))
    return {k: np.array(v) for k, v in L.items()}


if __name__ == '__main__':
    runner, ucv, z_trk = M.boot()
    try:
        Lk = episode(ucv, z_trk, dyn=False)
        print('kinematic vcap done, occl =', int((~Lk['det']).sum()))
        Ld = episode(ucv, z_trk, dyn=True)
        print('dynamic --dyn done,  occl =', int((~Ld['det']).sum()))
    finally:
        try:
            runner.close()
        except Exception:
            pass

    t = np.arange(M.N_STEPS) * M.DT
    fig = plt.figure(figsize=(14, 12), dpi=130)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.32)

    # --- panel 1: trajectory colored by speed (dyn run) ---
    ax0 = fig.add_subplot(gs[0])
    for (x0, x1, y0, y1) in M.WALLS:
        ax0.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor='#8f8f8f', edgecolor='k', zorder=3))
    cam = Ld['cam']
    spd = np.hypot(Ld['vx'], Ld['vy'])
    pts = cam.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='viridis', norm=plt.Normalize(0, 99), lw=2.6, zorder=5)
    lc.set_array(spd[:-1])
    ax0.add_collection(lc)
    plt.colorbar(lc, ax=ax0, fraction=0.025, pad=0.01, label='UAV speed (cm/s)')
    blind = ~Ld['det']
    ax0.plot(cam[blind, 0], cam[blind, 1], '.', ms=3, color='#C81E1E', zorder=6,
             label='blind steps')
    ax0.set_xlim(-760, 180); ax0.set_ylim(-740, 470)
    ax0.set_aspect('equal'); ax0.grid(alpha=0.2); ax0.legend(loc='lower right', fontsize=9)
    ax0.set_title(f'UAV trajectory colored by speed - seed {SEED}, dynamic mode '
                  f'(A_CAM={M.A_CAM:.0f} cm/s², sqrt braking law)', fontsize=12, fontweight='bold')

    # --- panel 2: velocity components ---
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, Ld['vx'], color='#1668C7', lw=1.4, label='vx (dyn)')
    ax1.plot(t, Ld['vy'], color='#E07B00', lw=1.4, label='vy (dyn)')
    ax1.plot(t, Lk['vx'], color='#1668C7', lw=0.8, alpha=0.35, label='vx (kinematic vcap)')
    ax1.plot(t, Lk['vy'], color='#E07B00', lw=0.8, alpha=0.35, label='vy (kinematic vcap)')
    for s in np.where(blind)[0]:
        ax1.axvspan(t[s], t[s] + M.DT, color='#C81E1E', alpha=0.06, lw=0)
    ax1.set_ylabel('velocity (cm/s)'); ax1.grid(alpha=0.25)
    ax1.legend(loc='lower left', fontsize=8.5, ncol=2)
    ax1.set_title('UAV velocity, x / y components (red band = occlusion)', fontsize=11)

    # --- panel 3: speed magnitude + clearance ---
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(t, np.hypot(Lk['vx'], Lk['vy']), color='#888888', lw=1.0, label='|v| kinematic vcap')
    ax2.plot(t, spd, color='#2E7D32', lw=1.4, label='|v| dynamic')
    ax2.plot(t, Ld['vdes'], color='#999999', lw=0.8, ls=':', label='controller desired |v|')
    ax2.set_ylabel('speed (cm/s)'); ax2.set_xlabel('time (s)')
    ax2b = ax2.twinx()
    ax2b.plot(t, Ld['clr'], color='#8B2FC9', lw=1.0, alpha=0.7)
    ax2b.axhline(15, color='#8B2FC9', ls='--', lw=0.8, alpha=0.7)
    ax2b.set_ylabel('wall clearance (cm)', color='#8B2FC9')
    ax2b.tick_params(axis='y', labelcolor='#8B2FC9')
    for s in np.where(blind)[0]:
        ax2.axvspan(t[s], t[s] + M.DT, color='#C81E1E', alpha=0.06, lw=0)
    ax2.grid(alpha=0.25); ax2.legend(loc='lower left', fontsize=8.5)
    ax2.set_title('speed magnitude vs clearance (purple, right axis; dashed = 15 cm margin)',
                  fontsize=11)

    plt.savefig(OUT, facecolor='white', bbox_inches='tight')
    print('saved:', OUT)
