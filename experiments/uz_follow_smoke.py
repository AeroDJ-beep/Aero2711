# -*- coding: utf-8 -*-
"""Smoke test of UZEnv with the navmesh grid: target on navmesh routes, naive
P-controller tracker (no KF) with clearance veto, measure step rate /
occlusion / clearance.  Outputs results/figures/uz_map_<MAP>.png and
results/logs/uz_smoke/<MAP>_follow_*.png"""
import os, sys, time, math
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_smoke')
MARGIN = 30.0
env = U.UZEnv(MAP)
try:
    env.spawn()
    nav = env.build_navgrid()
    gx, gy, _ = env.target_pose()
    rng = np.random.default_rng(0)
    goal = env.random_goal(rng, (gx, gy)); path = env.nav_target(*goal); env.set_target_speed(60)
    print('   goal', goal, 'path pts', len(path))
    tx, ty, yaw = env.place_tracker_behind()
    print('   clearance at spawn: tgt %.0f trk %.0f (trk at %.0f,%.0f yaw %.0f)' % (nav.clearance(gx, gy), nav.clearance(tx, ty), tx, ty, yaw))
    log = []; occl = 0; hits = 0; dts = []; still = 0; last_g = (gx, gy)
    for k in range(STEPS):
        t1 = time.time()
        box, npx, m, img = env.frame(lit=(k % 50 == 0))
        if img is not None:
            cv2.imwrite(os.path.join(OUT, f'{MAP}_follow_{k:03d}.png'), img)
        gx, gy, gyaw = env.target_pose()
        if box is not None and npx > 200:
            cx, h = box[0] + box[2] / 2.0, box[3]
            ang = math.degrees(math.atan2(cx - U.CX0, U.FX)); w = float(np.clip(3.0 * ang, -30, 30))
            rho = U.FX * U.H_PERSON / max(h, 1.0); v = float(np.clip(2.0 * (rho - 250.0), 0, 99))
        else:
            occl += 1; w, v = 0.0, 0.0
        dt = 0.1
        yaw += w * dt
        nx_, ny_ = tx + v * dt * math.cos(math.radians(yaw)), ty + v * dt * math.sin(math.radians(yaw))
        if nav.clearance(nx_, ny_) >= MARGIN:
            tx, ty = nx_, ny_
        elif nav.clearance(nx_, ty) >= MARGIN:
            tx = nx_; hits += 1
        elif nav.clearance(tx, ny_) >= MARGIN:
            ty = ny_; hits += 1
        else:
            hits += 1
        env.set_tracker(tx, ty, yaw)
        # re-route when the target reaches its goal or stalls
        moved = math.hypot(gx - last_g[0], gy - last_g[1]); last_g = (gx, gy)
        still = still + 1 if moved < 2.0 else 0
        if math.hypot(gx - goal[0], gy - goal[1]) < 80.0 or still > 15:
            goal = env.random_goal(rng, (gx, gy)); path = env.nav_target(*goal); still = 0
            print(f'   k={k} new goal {goal} ({"arrived" if still == 0 and moved >= 2.0 else "stalled"})')
        log.append((tx, ty, yaw, gx, gy, npx, nav.clearance(tx, ty)))
        dts.append(time.time() - t1)
        if k % 50 == 0:
            print(f'   k={k} npx={npx} tgt=({gx:.0f},{gy:.0f}) trk=({tx:.0f},{ty:.0f}) clr={log[-1][6]:.0f} step {np.mean(dts[-50:]) * 1000:.0f} ms')
    L = np.array(log)
    print(f'RESULT steps {STEPS}: mean step {np.mean(dts) * 1000:.0f} ms ({1 / np.mean(dts):.1f} Hz), occluded frames {occl}, vetoed moves {hits}, '
          f'min clearance {L[:, 6].min():.0f} cm, final range {math.hypot(L[-1, 3] - L[-1, 0], L[-1, 4] - L[-1, 1]):.0f} cm')
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(nav.free, cmap='gray', origin='lower', extent=[nav.x0, nav.x0 + nav.free.shape[1] * nav.cell, nav.y0, nav.y0 + nav.free.shape[0] * nav.cell])
    ax.plot(L[:, 3], L[:, 4], '-', color='#E07B00', lw=1.5, label='target'); ax.plot(L[:, 0], L[:, 1], '-', color='#1668C7', lw=1.5, label='tracker')
    ax.plot(L[0, 3], L[0, 4], 'o', color='#E07B00'); ax.plot(L[-1, 0], L[-1, 1], 's', color='#1668C7')
    ax.set_aspect('equal'); ax.legend(); ax.set_title(f'{MAP}: navmesh free space (white), {STEPS} steps'); ax.grid(alpha=.2)
    p = os.path.join(HERE, '..', 'results', 'figures', f'uz_map_{MAP}.png'); fig.savefig(p, dpi=110); print('saved', p)
finally:
    env.close()
