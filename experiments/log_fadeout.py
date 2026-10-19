# -*- coding: utf-8 -*-
"""Per-frame w/h/cx during the fade-out, seed 2005 sprint, mode ca_r01."""
import os, sys, time, math
import numpy as np
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
import run_metrics_test as M
import cv2

runner, ucv, z_trk = M.boot()
try:
    seed = 2005
    rng = np.random.default_rng(seed)
    route, c = M.gen_route(rng)
    corner = np.array([-435.0 - c, 245.0 + c])
    idx = int(np.argmin(np.linalg.norm(route - corner, axis=1)))
    seg_len = np.linalg.norm(np.diff(route, axis=0), axis=1)
    sprint_s = float(seg_len[:idx].sum())
    sprint_a = float(rng.uniform(-M.A_MAX, M.A_MAX))
    P, YAW = M.walk(route, rng, M.N_STEPS, sprint_s=sprint_s, rand_accel=True, sprint_a=sprint_a)

    d0 = route[1] - route[0]; u0 = d0 / np.linalg.norm(d0)
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
    gate_rej = 0
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
        d2, decision = None, ''
        if meas is not None and mean is None:
            mean, cov = kf.initiate(meas)
            decision = 'init'
        elif mean is not None:
            mean, cov = kf.predict(mean, cov, coasting=(meas is None))
            if meas is not None:
                d2 = M.gate_d2(mean, cov, meas)
                if d2 > M.GATE_CHI2:
                    gate_rej += 1
                    if gate_rej >= M.GATE_REINIT:
                        mean, cov = kf.initiate(meas)
                        gate_rej = 0
                        decision = 'REINIT'
                    else:
                        meas = None
                        decision = 'GATED'
                else:
                    gate_rej = 0
                    decision = 'ok'
            if meas is not None and decision == 'ok':
                mean, cov = kf.update(mean, cov, meas)
        if mean is not None:
            v_cmd, w_cmd = M.control(mean[0], mean[3])
        else:
            w_cmd = 0.0
        tyaw += w_cmd * M.DT
        hx, hy = math.cos(math.radians(tyaw)), math.sin(math.radians(tyaw))
        qx, qy = tx + v_cmd * M.DT * hx, ty + v_cmd * M.DT * hy
        tx, ty, hit = M.safe_move(tx, ty, qx, qy, v_cmd * M.DT)
        place()
        rows.append((k, npx, box[2], box[3],
                     box[0] + box[2] / 2 if box[3] > 0 else -1,
                     mean.copy() if mean is not None else None,
                     d2, decision))

    # find fade-out: last stretch npx>0 before first npx==0 run
    npxs = np.array([r[1] for r in rows])
    zero = np.where(npxs == 0)[0]
    k0 = zero[zero > 100][0]           # first blind step (skip warmup)
    reapp = np.where((npxs > 50) & (np.arange(len(npxs)) > k0))[0]
    k1 = int(reapp[0]) if len(reapp) else k0
    print(f'blind starts at step {k0}, reappearance at step {k1}')
    print('----- detector box (mask oracle) ----- | -------------------- KF state (10-dim) -------------------- |')
    print(f'{"step":>4} {"npx":>6} {"w":>4} {"h":>4} {"cx":>5} {"a_det":>6} |'
          f'{"cx":>6} {"cy":>6} {"a":>6} {"h":>6} |'
          f'{"vx":>6} {"vy":>6} {"va":>7} {"vh":>6} |'
          f'{"ax":>6} {"ay":>6} | {"rho_est":>7} | {"d2":>8} {"gate":>6}')
    for r in rows[k0 - 22:k0 + 12]:
        k, npx, w, h, cxd, m, d2, dec = r
        a_det = (w / h) if h > 0 else float('nan')
        det = f'{k:4d} {npx:6d} {w:4.0f} {h:4.0f} {cxd:5.0f} {a_det:6.3f} |'
        if m is None:
            print(det + '   (no KF state yet)')
            continue
        rho = M.FX * 168.0 / max(m[3], 1.0)
        d2s = f'{d2:8.1f}' if d2 is not None else ' ' * 8
        print(det +
              f'{m[0]:6.0f} {m[1]:6.0f} {m[2]:6.3f} {m[3]:6.1f} |'
              f'{m[4]:6.2f} {m[5]:6.2f} {m[6]:7.4f} {m[7]:6.2f} |'
              f'{m[8]:6.3f} {m[9]:6.3f} | {rho:7.0f} | {d2s} {dec:>6}')
    print('--- reappearance window ---')
    for r in rows[max(k1 - 2, 0):k1 + 9]:
        k, npx, w, h, cxd, m, d2, dec = r
        a_det = (w / h) if h > 0 else float('nan')
        det = f'{k:4d} {npx:6d} {w:4.0f} {h:4.0f} {cxd:5.0f} {a_det:6.3f} |'
        if m is None:
            print(det + '   (no KF state)')
            continue
        rho = M.FX * 168.0 / max(m[3], 1.0)
        d2s = f'{d2:8.1f}' if d2 is not None else ' ' * 8
        print(det +
              f'{m[0]:6.0f} {m[1]:6.0f} {m[2]:6.3f} {m[3]:6.1f} |'
              f'{m[4]:6.2f} {m[5]:6.2f} {m[6]:7.4f} {m[7]:6.2f} |'
              f'{m[8]:6.3f} {m[9]:6.3f} | {rho:7.0f} | {d2s} {dec:>6}')
finally:
    runner.close()
