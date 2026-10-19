# -*- coding: utf-8 -*-
"""Collect diffusion-planner training data in generated mazes with the maze A* expert
(same sample format as OA-VAT collect.py / unreal_replay.zarr).

Per sample: random target + UAV positions with the line of sight BLOCKED by a wall,
UAV facing the target (yaw0, as the original expert).  Expert = uz_astar.MazePlanner:
path from the UAV to a free cell 150-330 cm from the target WITH line of sight
(goal re-defined for corners; the original goal was straight ahead on the old line of
sight).  8 waypoints uniformly by arc length (first = start), rotated into the
starting body frame (x forward, y right, z up; cm).  Image = first-person lit frame
at that pose, resized to 360x240 RGB.  bbox = where the occluded target projects
(normalised x, y, w, h), from geometry.
Output: results/dp_maze/maze_<layout>.npz (img, bbox, action)
usage: python uz_collect_maze.py <n_layouts> <samples_per_layout>"""
import os, sys, math, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_maze as Z, uz_astar as A, uz_run as R

NL = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 350
OUT = os.path.join(HERE, '..', 'results', 'dp_maze'); os.makedirs(OUT, exist_ok=True)
H_TGT = 168.0


def resample(path, n=8):
    P = np.array(path, float)
    if len(P) == 1:
        return np.repeat(P, n, axis=0)
    seg = np.hypot(*np.diff(P, axis=0).T); cum = np.r_[0, np.cumsum(seg)]
    out = []
    for s in np.linspace(0, cum[-1], n):
        i = min(int(np.searchsorted(cum, s, side='right') - 1), len(seg) - 1)
        f = (s - cum[i]) / max(seg[i], 1e-6); out.append(P[i] + f * (P[i + 1] - P[i]))
    return np.array(out)


env = U.UZEnv('FlexibleRoom'); env.verbose = False
rng = np.random.default_rng(123)
try:
    first = True
    for li in range(NL):
        name = f'fr10x10l35t24_s{100 + li}'
        rects, info = Z.make_maze(10, 10, 200.0, 100 + li, 0.35, wall_t=24.0)
        Z.save_layout(name, rects, info)
        if first:
            R.setup_maze(env, name); first = False
        else:
            fb = U._floats(env.c.request('vget /object/FLOOR/bounds'))
            Z.build_in_engine(env, rects, fb[5]); R.MAZE = Z.MazeGeo(rects, info); env.nav = R.MAZE
        geo = R.MAZE; pl = A.MazePlanner(geo, margin=30.0)
        floor_top = U._floats(env.c.request('vget /object/FLOOR/bounds'))[5]
        P = geo.pts[np.array([geo.clearance(x, y) > 60.0 for x, y in geo.pts])]
        imgs, boxes, acts = [], [], []; tries = 0; t0 = time.time()
        while len(imgs) < NS and tries < NS * 40:
            tries += 1
            g = P[rng.integers(len(P))]
            d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
            cand = np.where((d > 180) & (d < 450))[0]
            if len(cand) == 0:
                continue
            u = P[rng.choice(cand)]
            if not Z.los_blocked(geo, tuple(u), tuple(g)):
                continue                                         # only occluded configurations
            path, goal = pl.plan(tuple(u), tuple(g))
            if not path:
                continue
            wp = resample(path, 8)
            yaw0 = math.degrees(math.atan2(g[1] - u[1], g[0] - u[0]))
            c, s = math.cos(math.radians(-yaw0)), math.sin(math.radians(-yaw0))
            rel = wp - u
            body = np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], np.zeros(len(rel))]
            # render: target facing a random direction, UAV camera at u facing the target
            R._teleport_target(env, g[0], g[1], float(rng.uniform(-180, 180)))
            env.set_tracker(u[0], u[1], yaw0)
            time.sleep(0.05)
            img = env.api.get_image(env.cam, 'lit')
            cam = U._floats(env.c.request(f'vget /camera/{env.cam}/location'))
            camh = cam[2] - floor_top
            rho = float(np.hypot(g[0] - u[0], g[1] - u[1]))
            top = U.CY0 - U.FX * (H_TGT - camh) / rho; bot = U.CY0 + U.FX * camh / rho
            hpx = bot - top; wpx = 0.33 * hpx
            bb = np.array([(U.CX0 - wpx / 2) / U.W, top / U.H, wpx / U.W, hpx / U.H], np.float32)
            im = cv2.cvtColor(cv2.resize(img, (360, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            imgs.append(im); boxes.append(bb); acts.append(body.astype(np.float32))
            if len(imgs) % 100 == 0:
                print(f'  layout {name}: {len(imgs)}/{NS} samples ({tries} tries, {time.time() - t0:.0f}s)', flush=True)
        np.savez_compressed(os.path.join(OUT, f'maze_{name}.npz'), img=np.array(imgs, np.uint8), bbox=np.array(boxes), action=np.array(acts))
        print(f'saved {name}: {len(imgs)} samples', flush=True)
finally:
    env.close()
print('COLLECT DONE', flush=True)
