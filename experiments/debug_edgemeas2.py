# -*- coding: utf-8 -*-
"""Seed-2000/p2/cv_r01: edgemeas OFF vs ON, find the divergence point."""
import os, sys, time, math
import numpy as np
SCRATCH_OLD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH_OLD)
import run_metrics_test as M
import cv2

SEED = 2000


def episode(ucv, z_trk, use_edge):
    rng = np.random.default_rng(SEED)
    route, c = M.gen_route(rng)
    corner = np.array([-435.0 - c, 245.0 + c])
    idx = int(np.argmin(np.linalg.norm(route - corner, axis=1)))
    seg_len = np.linalg.norm(np.diff(route, axis=0), axis=1)
    sprint_s = float(seg_len[:idx].sum())
    P, YAW = M.walk(route, rng, M.N_STEPS, sprint_s=sprint_s)   # p2 profile
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

    kf = M.KFTightR()
    mean = cov = None
    det_visible = False
    v_cmd = 0.0
    npx_win, w_good = [], None
    rows = []
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
        raw_cx = meas[0] if meas is not None else float('nan')
        flag = ''
        if use_edge and meas is not None:
            ref = max(npx_win) if npx_win else 0
            if ref > 0 and npx < M.FADE_FRAC * ref:
                if mean is not None and w_good is not None:
                    u = M.edge_u(tx, ty, tyaw)
                    if u is None:
                        meas = None
                        flag = 'REJ-oov'
                    else:
                        bl, br = box[0], box[0] + box[2]
                        if abs(br - u) <= abs(bl - u):
                            cxc = bl + w_good / 2.0
                            flag = f'EDGE-R u={u:.0f}'
                        else:
                            cxc = br - w_good / 2.0
                            flag = f'EDGE-L u={u:.0f}'
                        meas = np.array([cxc, mean[1], mean[2], mean[3]])
                else:
                    meas = None
                    flag = 'REJ'
            else:
                npx_win.append(npx)
                if len(npx_win) > 8:
                    npx_win.pop(0)
                if mean is not None:
                    w_good = mean[2] * mean[3]
                else:
                    w_good = meas[2] * meas[3]
        if meas is not None and mean is None:
            mean, cov = kf.initiate(meas)
        elif mean is not None:
            mean, cov = kf.predict(mean, cov)
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
        rows.append((k, npx, raw_cx, meas[0] if meas is not None else float('nan'),
                     mean[0] if mean is not None else float('nan'),
                     tx, ty, tyaw, flag))
    return rows


runner, ucv, z_trk = M.boot()
try:
    A = episode(ucv, z_trk, use_edge=False)
    B = episode(ucv, z_trk, use_edge=True)
finally:
    runner.close()

np.savez(r'C:\Users\AUFB\AppData\Local\Temp\claude\D--Aero2711\940a7311-f1f8-450b-95dd-fac3d5206417\scratchpad\edgedbg.npz',
         A=np.array([r[:8] for r in A], dtype=float),
         B=np.array([r[:8] for r in B], dtype=float))

# divergence point: first step where camera positions differ > 5 cm
div = None
for a, b in zip(A, B):
    if math.hypot(a[5] - b[5], a[6] - b[6]) > 5.0:
        div = a[0]
        break
print('divergence step:', div)
lo = max((div or 0) - 6, 0)
print(f'{"k":>4} | {"npx":>6} {"usedcx":>6} {"KFcx":>6} {"camx":>6} {"camy":>6} {"yaw":>5} || (edge ON columns)')
for a, b in zip(A[lo:lo + 120:3], B[lo:lo + 120:3]):
    print(f'{a[0]:4d} | OFF {a[1]:6d} {a[3]:6.0f} {a[4]:6.0f} {a[5]:6.0f} {a[6]:6.0f} {a[7]:5.0f} || '
          f'ON {b[1]:6d} {b[3]:6.0f} {b[4]:6.0f} {b[5]:6.0f} {b[6]:6.0f} {b[7]:5.0f} {b[8]:>12}')
