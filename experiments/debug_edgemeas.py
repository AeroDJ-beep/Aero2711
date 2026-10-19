# -*- coding: utf-8 -*-
"""Log-only diagnosis of edge_u / side selection during the fade, seed 2005.
Runs a NORMAL episode (no edgemeas applied) and prints what edgemeas WOULD do."""
import os, sys, time, math
import numpy as np
SCRATCH_OLD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH_OLD)
import run_metrics_test as M
import cv2

SEED = 2005
runner, ucv, z_trk = M.boot()
try:
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
    place(); time.sleep(0.3)

    kf = M.KFCATightR()
    mean = cov = None
    det_visible = False
    v_cmd = 0.0
    npx_peak = 0.0
    print(f'{"k":>4} {"npx":>6} {"bl":>5} {"br":>5} {"meascx":>6} {"u_edge":>7} '
          f'{"side":>6} {"cxc":>6} | {"camx":>6} {"camy":>6} {"yaw":>5} | {"tgt_bear_px":>10}')
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
        # log-only edgemeas analysis on fade frames
        if meas is not None and npx_peak > 0 and npx < M.FADE_FRAC * npx_peak and mean is not None:
            u = M.edge_u(tx, ty, tyaw)
            bl, br = box[0], box[0] + box[2]
            w_exp = mean[2] * mean[3]
            if u is not None:
                right_clipped = abs(br - u) <= abs(bl - u)
                side = 'RIGHT' if right_clipped else 'LEFT'
                cxc = (bl + w_exp / 2) if right_clipped else (br - w_exp / 2)
            else:
                side, cxc = 'OOV', float('nan')
            # ground truth: pixel column of the true target center
            gb = math.degrees(math.atan2(P[k, 1] - ty, P[k, 0] - tx))
            rel = (gb - tyaw + 180) % 360 - 180
            tgt_px = M.CX0 + M.FX * math.tan(math.radians(rel))
            print(f'{k:4d} {npx:6d} {bl:5.0f} {br:5.0f} {meas[0]:6.0f} '
                  f'{u if u is not None else float("nan"):7.0f} {side:>6} {cxc:6.0f} | '
                  f'{tx:6.0f} {ty:6.0f} {tyaw:5.0f} | {tgt_px:10.0f}')
        if meas is not None:
            if npx >= M.FADE_FRAC * max(npx_peak, 1.0):
                npx_peak = max(npx, npx_peak)
        npx_peak *= M.FADE_DECAY
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
        vx, vy, _ = M.vcap_step(tx, ty, vx, vy)
        qx, qy = tx + vx * M.DT, ty + vy * M.DT
        tx, ty, _ = M.safe_move(tx, ty, qx, qy, math.hypot(vx, vy) * M.DT)
        place()
finally:
    runner.close()
