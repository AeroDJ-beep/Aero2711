# -*- coding: utf-8 -*-
"""UnrealZoo environment wrapper for the occlusion-tracking pipeline.

Replaces the FlexibleRoom-specific plumbing in run_metrics_test.py:
  * UZEnv(map).spawn()      launch UnrealZoo on a scene, spawn target + tracker humans
  * build_navgrid()         free-space grid from navmesh reachable points (25 cm)
                            -> clearance (distance transform), outward normal,
                               occluder edge column, phantom clamp to free space
  * frame()                 tracker first-person object mask -> (box, npx, mask)
  * set_tracker(x,y,yaw)    teleport the tracker (physics off, camera follows)
  * nav_target / speed      engine moves the target along the navmesh
Geometry helpers mirror the L-map functions (dist_to_walls, wall_push_dir,
edge_u, map_clamp) so the estimator / safety code can be reused unchanged.
"""
import os, sys, json, time, math, re
import numpy as np
import cv2
from unrealcv.launcher import RunUnreal
from unrealcv.api import UnrealCv_API

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = r'E:\UnrealZoo\UnrealZoo_UE5_7_Win64_v3.1.0\Windows\UnrealZoo_UE5_7\Binaries\Win64\UnrealZoo_UE5_7.exe'
CFG_DIR = os.path.join(HERE, '..', 'unrealzoo-gym', 'gym_unrealcv', 'envs', 'setting', 'Track')
DATA_DIR = os.path.join(HERE, '..', 'env_config', 'data')
W, H = 640, 480
FOV = 82.6                         # Tello-class camera, same as the L-map setup
FX = (W / 2.0) / math.tan(math.radians(FOV / 2.0))   # 364.3
CX0, CY0 = W / 2.0, H / 2.0
H_PERSON = 168.0                   # cm, character height used for range from box height
CAM_OFF = (-19.0, 0.0, -40.0)      # camera offset from the tracker actor origin (x fwd, y, z up): ~100 cm above floor.
                                   # x=-19 cancels the character's built-in +19 cm -> camera at the capsule centre, so a body
                                   # near a wall is only pushed AWAY from it (old 40 put the body 59 cm behind the camera;
                                   # depenetration could push it through a wall and leave the camera inside it: blank mask)
PITCH = 0.0                        # level camera: head row well above the horizon -> well-conditioned head-row range
GRID = 25.0                        # navmesh sampling / grid cell (cm)


def _floats(s):
    return [float(v) for v in re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', s)]


class NavGrid:
    """Free-space grid from navmesh points; occupied = everything else."""
    def __init__(self, pts, cell=GRID, pad=8):
        self.cell = cell
        self.x0 = pts[:, 0].min() - pad * cell; self.y0 = pts[:, 1].min() - pad * cell
        nx = int((pts[:, 0].max() - self.x0) / cell) + pad + 1
        ny = int((pts[:, 1].max() - self.y0) / cell) + pad + 1
        free = np.zeros((ny, nx), np.uint8)
        ix = np.round((pts[:, 0] - self.x0) / cell).astype(int); iy = np.round((pts[:, 1] - self.y0) / cell).astype(int)
        free[iy, ix] = 1
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # fill gaps / smooth kinks (<= 2 cells)
        # navmesh points already keep the character capsule (~34 cm) off the walls, so
        # one cell of dilation (25 cm) is still wall-free for the camera point
        free = cv2.dilate(free, np.ones((3, 3), np.uint8))
        self.free = free.astype(bool)
        # signed clearance: +distance to the nearest occupied cell inside free space,
        # -distance to the nearest free cell inside occupied space (so "gaining
        # clearance" is well defined even when the tracker is pushed into a wall cell)
        d_in = cv2.distanceTransform(free, cv2.DIST_L2, 5) * cell
        d_out = cv2.distanceTransform((1 - free).astype(np.uint8), cv2.DIST_L2, 5) * cell
        self.dist = np.where(self.free, d_in, -d_out)
        gy, gx = np.gradient(self.dist)
        n = np.hypot(gx, gy) + 1e-9
        self.gx, self.gy = gx / n, gy / n
        # occupied boundary cells (occluder candidates)
        occ = (~self.free).astype(np.uint8)
        edge = occ & cv2.dilate(free, np.ones((3, 3), np.uint8))
        ys, xs = np.where(edge > 0)
        self.boundary = np.column_stack([self.x0 + xs * cell, self.y0 + ys * cell]).astype(float)
        self.pts = pts[:, :2].astype(float)

    def cell_of(self, x, y):
        ix = int(round((x - self.x0) / self.cell)); iy = int(round((y - self.y0) / self.cell))
        h, w = self.free.shape
        return min(max(iy, 0), h - 1), min(max(ix, 0), w - 1)

    def clearance(self, x, y):
        iy, ix = self.cell_of(x, y); return float(self.dist[iy, ix])

    def push_dir(self, x, y):
        iy, ix = self.cell_of(x, y)
        if self.dist[iy, ix] <= 0.0 or (abs(self.gx[iy, ix]) + abs(self.gy[iy, ix])) < 0.5:
            # inside / on an occupied cell: point at the nearest free navmesh point
            fx_, fy_, d = self.nearest_free(x, y)
            if d > 1e-6:
                return (fx_ - x) / d, (fy_ - y) / d
        return float(self.gx[iy, ix]), float(self.gy[iy, ix])

    def is_free(self, x, y):
        iy, ix = self.cell_of(x, y); return bool(self.free[iy, ix])

    def nearest_free(self, x, y):
        d2 = (self.pts[:, 0] - x) ** 2 + (self.pts[:, 1] - y) ** 2
        k = int(np.argmin(d2)); return float(self.pts[k, 0]), float(self.pts[k, 1]), math.sqrt(float(d2[k]))


class UZEnv:
    def __init__(self, map_name='Hotel_Corridor', resolution=(W, H), sleep_time=30, verbose=True):
        self.map = map_name
        self.cfg = json.load(open(os.path.join(CFG_DIR, map_name + '.json')))
        self.char_path = self.cfg['agents']['player']['asset_path'][0]
        self.starts = [list(map(float, s)) for s in self.cfg['safe_start']]
        self.runner = RunUnreal(ENV_BIN=EXE, ENV_MAP=map_name)
        if not os.path.isfile(self.runner.path2unrealcv):
            open(self.runner.path2unrealcv, 'w').write('[UnrealCV.Core]\nPort=9000\nWidth=%d\nHeight=%d\n' % resolution)
        ip, port = self.runner.start(resolution=resolution, sleep_time=sleep_time)
        self.api = UnrealCv_API(port=port, ip=ip, resolution=resolution)
        self.c = self.api.client
        self.verbose = verbose
        self.tgt, self.trk = 'tgt', 'trk'
        self.cam = None; self.tgt_rgb = None; self.nav = None; self.z = None; self.z_floor = None

    # ---------------- setup ----------------
    def calibrate_height(self, spot, dist=450.0):
        """Measure the target's effective height (cm) from its full-body box at a
        known range on open floor; returns H such that box_h = FX * H / range."""
        x, y = spot
        self.c.request([f'vset /object/{self.tgt}/location {x} {y} {self.z}', f'vset /object/{self.tgt}/rotation 0 90 0'], -1)
        self.set_tracker(x, y - dist, 90.0)
        time.sleep(0.8)
        box, npx, _, _ = self.frame()
        if box is None or box[1] <= 1 or box[1] + box[3] >= H - 1:
            return None
        return box[3] * dist / FX
    def spawn(self, tgt_xy=None, trk_xy=None, yaw=90.0, tgt_asset=None, tgt_app=None, tgt_scale=None):
        s0 = self.starts[0]
        tx, ty = (tgt_xy if tgt_xy is not None else s0[:2])
        kx, ky = (trk_xy if trk_xy is not None else (tx - 250 * math.cos(math.radians(yaw)), ty - 250 * math.sin(math.radians(yaw))))
        z = s0[2]
        for name, (x, y), asset in [(self.tgt, (tx, ty), tgt_asset or self.char_path), (self.trk, (kx, ky), self.char_path)]:
            self.c.request([f'vset /objects/spawn_from_path {asset} {name}',
                            f'vset /object/{name}/location {x} {y} {z}',
                            f'vset /object/{name}/rotation 0 {yaw} 0', f'vbp {name} set_phy 0'], -1)
        if tgt_app is not None:
            self.c.request(f'vbp {self.tgt} set_app {tgt_app}')
        if tgt_scale is not None:
            self.c.request(f'vset /object/{self.tgt}/scale {tgt_scale} {tgt_scale} {tgt_scale}')
        time.sleep(1.5)
        # the default (invisible) UnrealcvPawn stands in the scene with a collision
        # capsule and blocks the navigating target -> park it far away
        for o in self.api.get_objects():
            if o.startswith('UnrealcvPawn'):
                self.c.request(f'vset /object/{o}/location 100000 100000 {z + 500}')
        cams = self.c.request('vget /cameras').split()
        self.cam = next(i for i, n in enumerate(cams) if f'CID-{self.trk}-' in n)
        self.c.request(f'vset /object/{self.trk}/hide')      # camera sits inside the tracker capsule: hide its own mesh
        self.c.request(f'vbp {self.trk} set_cam {CAM_OFF[0]} {CAM_OFF[1]} {CAM_OFF[2]} 0 {PITCH} 0')
        self.c.request(f'vset /camera/{self.cam}/fov {FOV}')
        time.sleep(0.5)
        # measure the camera's horizontal offset from the tracker actor (actor at yaw 0):
        # the camera is ~60 cm ahead of the actor origin.  set_tracker() then treats its
        # (x, y) as the CAMERA position ("camera = vehicle", as in the L map), so the
        # safety layer, ranges and clearances all refer to the camera point.
        self.cam_off = (0.0, 0.0)
        self.c.request([f'vset /object/{self.trk}/location {kx} {ky} {z}', f'vset /object/{self.trk}/rotation 0 0 0'], -1)
        time.sleep(0.6)
        cl = _floats(self.c.request(f'vget /camera/{self.cam}/location'))
        al = _floats(self.c.request(f'vget /object/{self.trk}/location'))
        if len(cl) >= 2 and len(al) >= 2:
            self.cam_off = (cl[0] - al[0], cl[1] - al[1])
        self.cam_dz = (cl[2] - al[2]) if len(cl) >= 3 and len(al) >= 3 else 0.0      # camera height above the actor origin
        raw = self.c.request(f'vget /object/{self.tgt}/color')
        self.tgt_rgb = [int(v) for v in re.findall(r'[RGB]=(\d+)', raw)]
        self.z_floor = _floats(self.c.request(f'vget /object/{self.tgt}/location'))[2]
        # teleport height = the settled actor origin, not the config start height
        # (FlexibleRoom: config 159 vs settled 178 -> the capsule sank and bobbed every frame)
        self.z = self.z_floor
        self.cam_z = self.z + self.cam_dz                    # world height of the camera (fixed altitude)
        if self.verbose:
            print(f'[uz] spawned on {self.map}: tgt colour {self.tgt_rgb}, tracker cam {self.cam}, fov {self.c.request(f"vget /camera/{self.cam}/fov")}')

    def place_tracker_behind(self, dist=250.0, min_clear=60.0, heading=None):
        """Put the tracker on free space `dist` cm behind the target (along the
        target's heading, or `heading` deg if given), facing it; falls back to the
        nearest clear navmesh point."""
        gx, gy, gyaw = self.target_pose()
        if heading is not None:
            gyaw = heading
        cands = []
        for ang in [0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180]:
            a = math.radians(gyaw + 180.0 + ang)
            x, y = gx + dist * math.cos(a), gy + dist * math.sin(a)
            if self.nav.clearance(x, y) >= min_clear:
                cands.append((abs(ang), x, y))
        cands.sort()
        # fallback pool: navmesh points 150-320 cm from the target, preferring "behind"
        P = self.nav.pts
        d = np.hypot(P[:, 0] - gx, P[:, 1] - gy)
        back = math.radians(gyaw + 180.0)
        ang = np.abs((np.degrees(np.arctan2(P[:, 1] - gy, P[:, 0] - gx)) - math.degrees(back) + 180.0) % 360.0 - 180.0)
        pool = np.where((d > 150.0) & (d < 320.0))[0]
        pool = pool[np.argsort(ang[pool])][:12]
        cands += [(180 + float(ang[i]), float(P[i, 0]), float(P[i, 1])) for i in pool]
        if not cands:
            x, y, _ = self.nav.nearest_free(gx - dist * math.cos(math.radians(gyaw)), gy - dist * math.sin(math.radians(gyaw)))
            cands = [(0, x, y)]
        # keep the first candidate from which the target is actually visible (walls, door frames)
        for _, x, y in cands:
            yaw = math.degrees(math.atan2(gy - y, gx - x))
            self.set_tracker(x, y, yaw)
            time.sleep(0.25)
            box, npx, _, _ = self.frame()
            if self.verbose:
                print(f'[uz] placement candidate ({x:.0f},{y:.0f}) yaw {yaw:.0f}: target px {npx}')
            if npx >= 400:
                return x, y, yaw, npx
        _, x, y = cands[0]
        yaw = math.degrees(math.atan2(gy - y, gx - x))
        self.set_tracker(x, y, yaw)
        return x, y, yaw, 0

    def build_navgrid(self, cache=True):
        p = os.path.join(DATA_DIR, f'navpts_{self.map}.npy')
        pts = None
        if cache and os.path.isfile(p):
            pts = np.load(p)
        if pts is None or len(pts) < 100:
            r = self.c.request(f'vget /reachablepoints {GRID:g}', timeout=120)
            j = json.loads(r); pts = np.array([[q[0], q[1], q[2]] for q in j['points']], float)
            np.save(p, pts)
        self.nav = NavGrid(pts)
        if self.verbose:
            print(f'[uz] navgrid: {len(pts)} points, grid {self.nav.free.shape}, boundary cells {len(self.nav.boundary)}')
        return self.nav

    # ---------------- geometry (mirror of the L-map helpers) ----------------
    def dist_to_walls(self, x, y):
        return self.nav.clearance(x, y)

    def wall_push_dir(self, x, y):
        return self.nav.push_dir(x, y)

    def edge_u(self, x, y, yaw, tgt_rel=None, tgt_range=None, half_fov=55.0):
        """Image column of the occluding edge: boundary cell nearer than the target
        whose bearing is closest to the target's last relative bearing."""
        B = self.nav.boundary
        dx, dy = B[:, 0] - x, B[:, 1] - y
        rng = np.hypot(dx, dy)
        rel = (np.degrees(np.arctan2(dy, dx)) - yaw + 180.0) % 360.0 - 180.0
        ok = (np.abs(rel) < half_fov) & (rng > 30.0)
        if tgt_range is not None:
            ok &= rng < tgt_range
        if not ok.any():
            return None
        score = np.abs(rel - (tgt_rel if tgt_rel is not None else 0.0))
        score[~ok] = 1e9
        k = int(np.argmin(score))
        return CX0 + FX * math.tan(math.radians(rel[k]))

    def map_clamp_xy(self, x, y, tol=40.0):
        """Snap a phantom world point into free space (nearest navmesh point)."""
        if self.nav.is_free(x, y):
            return x, y, False
        fx_, fy_, d = self.nav.nearest_free(x, y)
        return fx_, fy_, True

    # ---------------- actors ----------------
    def set_tracker(self, x, y, yaw):
        """(x, y) is the desired CAMERA position; the actor is placed behind it."""
        ox, oy = getattr(self, 'cam_off', (0.0, 0.0))
        c_, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ax_, ay_ = x - (c_ * ox - s_ * oy), y - (s_ * ox + c_ * oy)
        self.c.request([f'vset /object/{self.trk}/location {ax_} {ay_} {self.z}',
                        f'vset /object/{self.trk}/rotation 0 {yaw} 0'], -1)

    def target_pose(self):
        loc = _floats(self.c.request(f'vget /object/{self.tgt}/location'))
        rot = _floats(self.c.request(f'vget /object/{self.tgt}/rotation'))
        return loc[0], loc[1], (rot[1] if len(rot) > 1 else 0.0)

    def set_target_speed(self, v, acc=None):
        self.c.request(f'vbp {self.tgt} set_speed {v}')
        if acc is not None:
            self.c.request(f'vbp {self.tgt} set_acc {acc}')

    def nav_target(self, x, y):
        r = self.c.request(f'vbp {self.tgt} nav_to_goal_bypath {x} {y} {self.z_floor}')
        try:
            path = re.findall(r'X=(-?\d+\.?\d*) Y=(-?\d+\.?\d*) Z=(-?\d+\.?\d*)', json.loads(r)['Path'])
            P = np.array([[float(a), float(b)] for a, b, _ in path])
            # the engine echoes the path twice; keep the first pass
            n = len(P)
            if n >= 4 and np.allclose(P[0], P[n // 2]):
                P = P[:n // 2]
            return P
        except Exception:
            return np.zeros((0, 2))

    def nav_status(self):
        return self.c.request(f'vget /agent/{self.tgt}/nav/status')

    def random_goal(self, rng, from_xy, dmin=1200.0):
        P = self.nav.pts
        d = np.hypot(P[:, 0] - from_xy[0], P[:, 1] - from_xy[1])
        cand = np.where((d >= dmin) & (np.array([self.nav.clearance(px, py) for px, py in P]) > 60.0))[0] if len(P) < 5000 else np.where(d >= dmin)[0]
        if len(cand) == 0:
            cand = np.argsort(d)[-20:]
        k = int(rng.choice(cand))
        return float(P[k, 0]), float(P[k, 1])

    # ---------------- perception ----------------
    def frame(self, lit=False):
        msk = self.api.get_image(self.cam, 'object_mask')
        d = np.abs(msk.astype(np.int16) - np.array(self.tgt_rgb[::-1], np.int16)[None, None, :]).sum(axis=2)
        m = d < 9
        npx = int(m.sum())
        box = None
        if npx > 0:
            ys, xs = np.where(m)
            box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        img = self.api.get_image(self.cam, 'lit') if lit else None
        return box, npx, m, img

    def close(self):
        try:
            self.runner.close()
        except Exception:
            pass


def true_cx_h(tx, ty, yaw, gx, gy):
    """Analytic image column / box height of the target centre (for metrics)."""
    bear = math.degrees(math.atan2(gy - ty, gx - tx))
    rel = (bear - yaw + 180.0) % 360.0 - 180.0
    rng = math.hypot(gx - tx, gy - ty)
    return CX0 + FX * math.tan(math.radians(rel)), FX * H_PERSON / max(rng, 1.0), rng, rel
