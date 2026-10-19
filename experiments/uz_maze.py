# -*- coding: utf-8 -*-
"""Procedural maze inside the UnrealZoo FlexibleRoom (an empty room).

make_maze(nx, ny, cell, seed, loops)  -> list of wall rectangles (x0, x1, y0, y1) in cm
MazeGeo(rects)                        -> analytic clearance / push_dir / edge_u / clamp
build_in_engine(env, rects)           -> spawns /Engine/BasicShapes/Cube.Cube walls
Walls are thin (WALL_T = 20 cm) and WALL_H tall; corridors are cell - WALL_T wide.
Run as a script: build a maze, save top view + first-person view + layout json."""
import os, sys, json, math, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

CUBE = '/Engine/BasicShapes/Cube.Cube'      # 100 cm cube, pivot at centre
WALL_T, WALL_H = 20.0, 250.0
ORIGIN = (-450.0, -50.0)                    # centre of the FlexibleRoom floor
DATA = os.path.join(HERE, '..', 'env_config', 'data')


# ---------------- maze generation ----------------
def make_maze(nx=8, ny=8, cell=200.0, seed=0, loops=0.15, origin=ORIGIN, wall_t=50.0, gates=False):
    """Recursive-backtracker perfect maze on an nx x ny grid, then knock out a
    fraction `loops` of the remaining interior walls to create cycles.
    Returns merged wall rectangles and cell-centre helper info."""
    rng = np.random.default_rng(seed)
    # walls: vertical v[i][j] between cell (i-1,j) and (i,j), i=0..nx ; horizontal h[i][j] between (i,j-1),(i,j), j=0..ny
    v = np.ones((nx + 1, ny), bool); h = np.ones((nx, ny + 1), bool)
    seen = np.zeros((nx, ny), bool); stack = [(0, 0)]; seen[0, 0] = True
    while stack:
        i, j = stack[-1]
        nb = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= i + di < nx and 0 <= j + dj < ny and not seen[i + di, j + dj]]
        if not nb:
            stack.pop(); continue
        a, b = nb[rng.integers(len(nb))]
        if a != i:
            v[max(a, i), j] = False
        else:
            h[i, max(b, j)] = False
        seen[a, b] = True; stack.append((a, b))
    interior = [('v', i, j) for i in range(1, nx) for j in range(ny) if v[i, j]] + \
               [('h', i, j) for i in range(nx) for j in range(1, ny) if h[i, j]]
    for k in rng.permutation(len(interior))[:int(loops * len(interior))]:
        t, i, j = interior[k]
        (v if t == 'v' else h)[i, j] = False
    if gates:
        h[0, ny] = False          # entrance: top wall of the top-left cell (min x, max y)
        h[nx - 1, 0] = False      # exit: bottom wall of the bottom-right cell (max x, min y)
    W, H = nx * cell, ny * cell
    x0, y0 = origin[0] - W / 2.0, origin[1] - H / 2.0
    rects = []
    WALL_T = wall_t                                     # local override of the module default
    # merge collinear runs of wall segments
    for i in range(nx + 1):
        j = 0
        while j < ny:
            if v[i, j]:
                k = j
                while k + 1 < ny and v[i, k + 1]:
                    k += 1
                x = x0 + i * cell
                rects.append((x - WALL_T / 2, x + WALL_T / 2, y0 + j * cell - WALL_T / 2, y0 + (k + 1) * cell + WALL_T / 2))
                j = k + 1
            else:
                j += 1
    for j in range(ny + 1):
        i = 0
        while i < nx:
            if h[i, j]:
                k = i
                while k + 1 < nx and h[k + 1, j]:
                    k += 1
                y = y0 + j * cell
                rects.append((x0 + i * cell - WALL_T / 2, x0 + (k + 1) * cell + WALL_T / 2, y - WALL_T / 2, y + WALL_T / 2))
                i = k + 1
            else:
                i += 1
    info = dict(nx=nx, ny=ny, cell=cell, x0=x0, y0=y0, seed=seed, loops=loops, wall_t=wall_t, gates=gates,
                v=v.tolist(), h=h.tolist())
    return rects, info


def cell_center(info, i, j):
    return info['x0'] + (i + 0.5) * info['cell'], info['y0'] + (j + 0.5) * info['cell']


def cell_graph(info):
    """Adjacency of open cells (for target routes)."""
    nx, ny = info['nx'], info['ny']; v = np.array(info['v']); h = np.array(info['h'])
    adj = {(i, j): [] for i in range(nx) for j in range(ny)}
    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx and not v[i + 1, j]: adj[(i, j)].append((i + 1, j)); adj[(i + 1, j)].append((i, j))
            if j + 1 < ny and not h[i, j + 1]: adj[(i, j)].append((i, j + 1)); adj[(i, j + 1)].append((i, j))
    return adj


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
OCC_INIT_W = 0.1         # CorridorPredictor(occ_init=True): weight factor of start directions that are in view
INIT_BRANCH_W = 0.25     # CorridorPredictor(init_branch=True): weight of the other directions at the start


class CorridorPredictor:
    """Blind prediction that follows the maze corridors (people walk along the cell
    centre lines), with a few discrete hypotheses.  Each hypothesis moves at the
    estimated speed along an axis direction and turns at a cell centre where that
    direction is walled off.  At a branching point it splits into one hypothesis per
    open branch; a branch the UAV can see into gets a low weight (the target is NOT
    seen there).  Every step a hypothesis that lies inside the UAV's view without the
    target being detected is down-weighted (negative information).  Output = the
    heaviest hypothesis (with hysteresis), so the planner looks at it and, if the
    target is not there, the next one takes over."""
    W_SEEN, W_MIN, N_MAX = 0.3, 0.03, 8
    MASS_MIN = 0.05          # --reset: restart when <5% of the belief mass survived the negative information

    def __init__(self, geo, p, v, mults=(1.0,), uturn=False, reset=False, init_branch=False, cam=None, occ_init=False, reexpand=False,
                 came_from=None, reach=False):
        self.geo, self.info = geo, geo.info
        self.uturn, self.reset = uturn, reset          # dead ends: turn back; all ruled out: restart from p0
        self.p0, self.mults = np.array(p, float), tuple(mults)
        self.adj = cell_graph(geo.info)
        self.trav = 0.0; self.mass = 1.0; self.n_reset = 0; self.n_reexp = 0; self.reexpand = reexpand
        self.came_from = tuple(came_from) if came_from is not None else None   # direction of the cell the walker entered from
        self.reach = reach; self.n_reach = 0
        self.adj = cell_graph(geo.info)
        self.v = np.array(v, float)
        d = max(DIRS, key=lambda e: e[0] * v[0] + e[1] * v[1])
        # speed hypotheses: the target may speed up or slow down while hidden; each hypothesis
        # moves at m x the estimated speed (m = 1 first: preferred on ties)
        mults = sorted(mults, key=lambda m: abs(m - 1.0))
        self.H = [dict(p=np.array(p, float), d=d, w=1.0, m=float(m)) for m in mults]
        if init_branch:
            # the target is often hidden right after turning, while the smoothed velocity still points
            # the old way.  Start a hypothesis in EVERY open direction of the current cell (not back).
            # Weight by position: a walker stands on the centre line of the corridor it walks along, so
            # a direction whose centre line passes within 40 cm of the last seen point gets weight 1,
            # the others INIT_BRANCH_W.  Each starts where the walker already is along that direction.
            c = self.cell(self.p0)
            if c is not None:
                cc = np.array(cell_center(geo.info, *c)); off = self.p0 - cc
                def on_line(e):
                    return abs(off[0] * e[1] - off[1] * e[0]) < 40.0          # lateral distance to e's centre line
                self.H = []
                back = self.came_from if self.came_from is not None else (-d[0], -d[1])
                for e in DIRS:
                    # 'back' = the cell the walker came from (seen while visible); without it, the opposite of the
                    # velocity axis -- which is wrong when the velocity is diagonal mid-turn
                    if e == back or not (e == d or self._open(c, e)):
                        continue
                    along = e[0] * v[0] + e[1] * v[1]                       # velocity component along e
                    w = (1.0 if along >= 0 else 0.5) if on_line(e) else INIT_BRANCH_W
                    start = self.p0.copy() if e == d else cc + np.array(e, float) * max(0.0, float(np.dot(off, e)))
                    self.H += [dict(p=start.copy(), d=e, w=w, m=float(m)) for m in mults]
                if not self.H:
                    self.H = [dict(p=self.p0.copy(), d=d, w=1.0, m=float(m)) for m in mults]
                if len(self.H) > len(mults):
                    self.N_MAX = max(self.N_MAX, 16)
        if len(mults) > 1:
            self.N_MAX = 16
        if occ_init and cam is not None:
            # occlusion consistency (the "which side was cut" cue, done with the map): the person has
            # just disappeared, so a direction whose next 1.3-1.7 m (past the cell edge, into the next cell) is in clear view RIGHT NOW is unlikely
            # (walking that way they would still be seen); hidden directions keep their weight
            for h in self.H:
                dv = np.array(h['d'], float); c0 = self.cell(h['p'])
                walled = c0 is not None and not self._open(c0, h['d'])          # cannot go that way at all
                if walled or any(self._visible(tuple(h["p"] + dv * a), cam) for a in (130.0, 170.0)):
                    h['w'] *= OCC_INIT_W
        self.best = 0
        self.n_split = 0

    def cell(self, p):
        inf = self.info
        i, j = int(math.floor((p[0] - inf['x0']) / inf['cell'])), int(math.floor((p[1] - inf['y0']) / inf['cell']))
        return (i, j) if 0 <= i < inf['nx'] and 0 <= j < inf['ny'] else None

    def _open(self, c, e):
        n = (c[0] + e[0], c[1] + e[1])
        if n in self.adj[c]:
            return True
        cc = np.array(cell_center(self.info, *c))
        if self.cell(cc + np.array(e) * self.info['cell']) is None:       # leaving the grid (entrance / exit gate)
            return not los_blocked(self.geo, tuple(cc), tuple(cc + np.array(e) * 0.75 * self.info['cell']))
        return False

    def _visible(self, q, cam):
        tx, ty, yaw = cam
        rel = (math.degrees(math.atan2(q[1] - ty, q[0] - tx)) - yaw + 180.0) % 360.0 - 180.0
        return abs(rel) < 38.0 and 50.0 < math.hypot(q[0] - tx, q[1] - ty) < 800.0 and not los_blocked(self.geo, (tx, ty), tuple(q))

    def _split(self, h, c, opts, cam, ds):
        """Branch hypothesis h at the centre of cell c into the open directions."""
        cc = np.array(cell_center(self.info, *c))
        opts = sorted(opts, key=lambda e: (e != h['d'], -(e[0] * self.v[0] + e[1] * self.v[1])))   # straight first
        out = []
        for e in opts:
            seen = self._visible(cc + np.array(e) * 0.7 * self.info['cell'], cam)
            out += self._advance(dict(p=cc.copy(), d=e, w=h['w'] * (self.W_SEEN if seen else 1.0), m=h.get('m', 1.0)), ds, cam, depth=1)
        if len(opts) > 1:
            self.n_split += 1
        return out

    def _advance(self, h, ds, cam, depth=0):
        p, d = h['p'], np.array(h['d'], float)
        c = self.cell(p)
        if c is None or ds <= 1e-6:                                         # outside the maze: straight on
            h['p'] = p + d * ds; return [h]
        cc = np.array(cell_center(self.info, *c))
        to_c = float(np.dot(cc - p, d))                                     # > 0: the cell centre is still ahead
        side = [e for e in DIRS if e != h['d'] and e != (-h['d'][0], -h['d'][1]) and self._open(c, e)]
        straight = self._open(c, h['d'])
        if depth == 0 and (not straight or side) and ((to_c > 1e-3 and ds >= to_c) or (not straight and to_c <= 1e-3)):
            rest = ds - max(to_c, 0.0)
            opts = side + ([h['d']] if straight else [])
            if not opts:                                                    # dead end
                if self.uturn:                                              # a walker turns back
                    return self._advance(dict(h, p=cc.copy(), d=(-h['d'][0], -h['d'][1])), rest, cam, depth=1)
                h['p'] = cc.copy(); return [h]                              # (else wait at the centre)
            return self._split(h, c, opts, cam, rest)
        perp = np.array([-d[1], d[0]])
        h['p'] = p + d * ds + perp * 0.3 * float(np.dot(cc - p, perp))     # drift back to the centre line
        return [h]

    def _restart(self, cam):
        """Every hypothesis has been looked at without the target: start again from the last
        seen point in ALL open directions (also back), each moved by the distance travelled."""
        c = self.cell(self.p0); H = []
        dirs = [e for e in DIRS if c is not None and self._open(c, e)] or list(DIRS)
        for e in dirs:
            for m in self.mults:
                H += self._advance(dict(p=self.p0.copy(), d=e, w=1.0, m=float(m)), self.trav * m, cam, depth=1)
        W = sum(h['w'] for h in H) or 1.0
        for h in H:
            h['w'] /= W
        self.n_reset += 1; self.mass = 1.0
        return sorted(H, key=lambda h: -h['w'])[:self.N_MAX]

    def _reach_expand(self, cam):
        """All hypotheses were looked at without the target.  Restart from the LAST SEEN point
        (not from the spot just ruled out): every cell the walker could have reached along the
        corridors in the distance travelled so far (BFS on the cell graph), minus the cells in view
        right now; one hypothesis per frontier cell, heading away from the last seen cell."""
        c0 = self.cell(self.p0)
        if c0 is None:
            return []
        cs = self.info['cell']; R = max(self.trav * max(self.mults), cs)
        dist = {c0: 0.0}; came = {c0: None}; q = [c0]
        while q:
            c = q.pop(0)
            for n in self.adj[c]:
                if n not in dist and dist[c] + cs <= R + 1e-6:
                    dist[n] = dist[c] + cs; came[n] = (n[0] - c[0], n[1] - c[1]); q.append(n)
        cand = [c for c in dist if c != c0 and not self._visible(cell_center(self.info, *c), cam)]
        front = [c for c in cand if dist[c] > R - cs]
        pick = sorted(front or cand, key=lambda c: -dist[c])[:self.N_MAX]
        return [dict(p=np.array(cell_center(self.info, *c), float), d=came[c], w=1.0, m=1.0) for c in pick]

    def peek(self, dist, cam):
        """Where the current best hypothesis will be after `dist` cm along the corridors
        (no weights: at a branch the preferred child is followed).  Used as the planner's goal."""
        h = dict(self.H[self.best]); h['p'] = h['p'].copy(); left = dist
        while left > 1e-3:
            stp = min(20.0, left); left -= stp
            hs = self._advance(h, stp, cam)
            h = max(hs, key=lambda q: q['w']); h = dict(h, p=h['p'].copy())
        return tuple(h['p'])

    def step(self, ds, cam):
        cur = self.H[self.best]
        H = []
        for h in self.H:
            H += self._advance(h, ds * h.get('m', 1.0), cam)
        self.trav += ds
        W0 = sum(h['w'] for h in H) or 1.0
        for h in H:                                                         # negative information
            if self._visible(h['p'], cam):
                h['w'] *= self.W_SEEN
        W = sum(h['w'] for h in H) or 1.0
        self.mass *= W / W0                                                 # how much belief survived the looks
        if self.reset and self.mass < self.MASS_MIN:
            self.H = self._restart(cam); self.best = 0
            return tuple(self.H[0]['p'])
        if self.reach and self.mass < self.MASS_MIN:
            Hr = self._reach_expand(cam)
            if Hr:
                H = Hr; self.mass = 1.0; self.n_reach += 1
                W = sum(h['w'] for h in H) or 1.0
        if self.reexpand and self.mass < self.MASS_MIN:
            # all surviving hypotheses were looked at without the target (e.g. a single lagging one that
            # normalisation keeps at weight 1): re-expand from the CURRENT best point in every open
            # direction and every speed multiple -- the walker has moved on from where we looked
            pb = H[max(range(len(H)), key=lambda i: H[i]['w'])]['p'] if H else self.p0
            c = self.cell(pb); dirs = [e for e in DIRS if c is not None and self._open(c, e)] or list(DIRS)
            cc = np.array(cell_center(self.info, *c)) if c is not None else np.array(pb, float)
            H = [dict(p=cc.copy(), d=e, w=1.0, m=float(m)) for e in dirs for m in self.mults if m > 0] or H
            self.mass = 1.0; self.n_reexp += 1
            W = sum(h['w'] for h in H) or 1.0
        for h in H:
            h['w'] /= W
        H = sorted([h for h in H if h['w'] >= self.W_MIN] or H[:1], key=lambda h: -h['w'])[:self.N_MAX]
        # hysteresis: stay with the successor of the current best unless another is clearly heavier
        k_cur = min(range(len(H)), key=lambda i: float(np.hypot(*(H[i]['p'] - cur['p']))))
        self.best = 0 if H[0]['w'] > 1.2 * H[k_cur]['w'] else k_cur
        self.H = H
        return tuple(H[self.best]['p'])


# ---------------- analytic geometry ----------------
class MazeGeo:
    """Same interface as NavGrid (clearance, push_dir, is_free, nearest_free, boundary)
    but exact, from the wall rectangles."""
    def __init__(self, rects, info):
        self.rects = [tuple(r) for r in rects]; self.info = info
        R = np.array(self.rects, float)
        self.R = R
        # boundary points for edge_u: rectangle corners (vertical occluding edges)
        C = []
        for (x0, x1, y0, y1) in self.rects:
            C += [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
        self.boundary = np.array(C, float)
        # free sample points (cell grid at 25 cm) for clamp / random goals; with gates the
        # room floor around the maze (pad) is free too: the target starts and ends out there
        cell = info['cell']
        self.pad = 3 * cell if info.get('gates') else 0.0
        xs = np.arange(info['x0'] - self.pad + 12.5, info['x0'] + info['nx'] * cell + self.pad, 25.0)
        ys = np.arange(info['y0'] - self.pad + 12.5, info['y0'] + info['ny'] * cell + self.pad, 25.0)
        g = np.array([(x, y) for x in xs for y in ys], float)
        self.pts = g[np.array([self.clearance(x, y) > 35.0 for x, y in g])]

    def clearance(self, x, y):
        R = self.R
        dx = np.maximum(np.maximum(R[:, 0] - x, 0.0), x - R[:, 1])
        dy = np.maximum(np.maximum(R[:, 2] - y, 0.0), y - R[:, 3])
        d = np.hypot(dx, dy)
        inside = (dx == 0) & (dy == 0)
        if inside.any():
            k = np.where(inside)[0][0]; x0, x1, y0, y1 = R[k]
            return -min(x - x0, x1 - x, y - y0, y1 - y)
        return float(d.min())

    def push_dir(self, x, y):
        R = self.R
        cx = np.clip(x, R[:, 0], R[:, 1]); cy = np.clip(y, R[:, 2], R[:, 3])
        d = np.hypot(x - cx, y - cy); k = int(np.argmin(d))
        if d[k] > 1e-6:
            return (x - cx[k]) / d[k], (y - cy[k]) / d[k]
        x0, x1, y0, y1 = R[k]
        m = [x - x0, x1 - x, y - y0, y1 - y]; a = int(np.argmin(m))
        return [(-1, 0), (1, 0), (0, -1), (0, 1)][a]

    def is_free(self, x, y):
        inf = self.info; p = getattr(self, 'pad', 0.0)
        inside = (inf['x0'] - p < x < inf['x0'] + inf['nx'] * inf['cell'] + p) and (inf['y0'] - p < y < inf['y0'] + inf['ny'] * inf['cell'] + p)
        return inside and self.clearance(x, y) > 0.0

    def nearest_free(self, x, y):
        d2 = (self.pts[:, 0] - x) ** 2 + (self.pts[:, 1] - y) ** 2
        k = int(np.argmin(d2)); return float(self.pts[k, 0]), float(self.pts[k, 1]), math.sqrt(float(d2[k]))


def los_blocked(geo, a, b):
    """Exact segment-vs-rectangle test."""
    ax, ay = a; bx, by = b
    for (x0, x1, y0, y1) in geo.rects:
        t0, t1 = 0.0, 1.0; dx, dy = bx - ax, by - ay; ok = True
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if abs(p) < 1e-12:
                if q < 0: ok = False; break
            else:
                t = q / p
                if p < 0: t0 = max(t0, t)
                else: t1 = min(t1, t)
                if t0 > t1: ok = False; break
        if ok:
            return True
    return False


# ---------------- scripted target ----------------
class MazeTarget:
    """Target driven by us (teleported every step) along a cell-graph route.
    Route = random walk over open cells (no immediate backtrack unless at a dead
    end), polyline through cell centres, collinear points merged.  The route is
    accepted only if it has an OCCLUDING corner at 4-12 m (line of sight between
    2.5 m before and 2.5 m after the corner crosses a wall).
    Speed: v0 ~ U(40, 60) cm/s walk; from the corner on, constant acceleration
    `sa` (p2: +80 cm/s^2 up to 99; p2a: U(-80, 80), floor 20) -- same as the L map."""
    V_MAX, V_FLOOR, A_MAX = 99.0, 20.0, 80.0
    MIN_AFTER = 0.0          # >0: require a straight leg of this length after the corner (L-map-like)

    def __init__(self, geo, rng, profile='p2', n_cells=18):
        self.geo, self.rng, self.profile = geo, rng, profile
        self.adj = cell_graph(geo.info); self.n_cells = n_cells

    def _walk(self):
        info = self.geo.info
        cur = (int(self.rng.integers(info['nx'])), int(self.rng.integers(info['ny'])))
        cells = [cur]; prev = None
        for _ in range(self.n_cells):
            nb = [c for c in self.adj[cur] if c != prev] or self.adj[cur]
            if not nb:
                break
            nxt = nb[int(self.rng.integers(len(nb)))]
            prev, cur = cur, nxt; cells.append(cur)
        P = np.array([cell_center(info, i, j) for i, j in cells], float)
        keep = [0]
        for k in range(1, len(P) - 1):
            d1 = P[k] - P[keep[-1]]; d2 = P[k + 1] - P[k]
            if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-6 or np.dot(d1, d2) < 0:
                keep.append(k)
        keep.append(len(P) - 1)
        return P[keep]

    def _setup_path(self, P):
        self.P = P
        seg = np.hypot(*np.diff(P, axis=0).T)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])

    def point_at(self, s):
        s = min(max(s, 0.0), self.cum[-1])
        i = int(np.searchsorted(self.cum, s, side='right') - 1); i = min(i, len(self.P) - 2)
        f = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-6)
        p = self.P[i] + f * (self.P[i + 1] - self.P[i])
        d = self.P[i + 1] - self.P[i]
        return float(p[0]), float(p[1]), math.degrees(math.atan2(d[1], d[0]))

    def plan(self, dmin=400.0, dmax=1400.0, look=280.0, behind=250.0, tries=4000, min_after=None):
        """Accept a route only if (a) its FIRST turn is an occluding corner at
        dmin..dmax, and (b) the corridor continues straight >= `behind` cm back from
        the start with line of sight, so the tracker starts lined up (like the L map)."""
        for _ in range(tries):
            P = self._walk()
            if len(P) < 3:
                continue
            self._setup_path(P)
            s_c = self.cum[1]
            if not (dmin <= s_c <= dmax and s_c + look <= self.cum[-1]):
                continue
            a = self.point_at(s_c - look)[:2]; b = self.point_at(s_c + look)[:2]
            if not los_blocked(self.geo, a, b):
                continue
            ma = self.MIN_AFTER if min_after is None else min_after
            if ma and (len(P) < 3 or self.cum[2] - self.cum[1] < ma):
                continue                     # single-corner episode: straight leg after the corner
            d = (P[1] - P[0]) / max(np.hypot(*(P[1] - P[0])), 1e-6)
            bx, by = P[0][0] - behind * d[0], P[0][1] - behind * d[1]
            if self.geo.clearance(bx, by) < 60.0 or not self.geo.is_free(bx, by) or los_blocked(self.geo, (bx, by), tuple(P[0])):
                continue
            if ma:
                self._setup_path(P[:3])      # episode = approach leg + corner + straight leg, then stop
            self.s_corner = float(s_c); self.corner = (float(P[1][0]), float(P[1][1]))
            self.tracker_start = (float(bx), float(by))
            self.v = float(self.rng.uniform(40.0, 60.0)); self.s = 0.0; self.sprinting = False
            self.sa = self.A_MAX if self.profile == 'p2' else float(self.rng.uniform(-self.A_MAX, self.A_MAX))
            return True
        return False

    def plan_gates(self, behind=250.0, outside=1.5):
        """Fixed route: from outside the top-left entrance, along the shortest path
        through the maze, out of the bottom-right exit.  A speed manoeuvre happens
        after EVERY corner (p2: +80 cm/s^2; p2a: a fresh U(-80, 80) per corner); the
        target relaxes back to its walking speed v0 (at 80 cm/s^2) over the last
        150 cm before the next corner, as a person slows down to turn."""
        info = self.geo.info; nx, ny, cell = info['nx'], info['ny'], info['cell']
        a, b = (0, ny - 1), (nx - 1, 0)
        prev = {a: None}; q = [a]
        while q:
            cur = q.pop(0)
            if cur == b:
                break
            for nb in self.adj[cur]:
                if nb not in prev:
                    prev[nb] = cur; q.append(nb)
        cells = [b]
        while prev[cells[-1]] is not None:
            cells.append(prev[cells[-1]])
        cells = cells[::-1]
        top = info['y0'] + ny * cell; bot = info['y0']
        start = (cell_center(info, 0, ny - 1)[0], top + outside * cell)
        end = (cell_center(info, nx - 1, 0)[0], bot - outside * cell)
        P = np.array([start] + [cell_center(info, i, j) for i, j in cells] + [end], float)
        keep = [0]
        for k in range(1, len(P) - 1):
            d1 = P[k] - P[keep[-1]]; d2 = P[k + 1] - P[k]
            if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-6 or np.dot(d1, d2) < 0:
                keep.append(k)
        keep.append(len(P) - 1)
        P = P[keep]; self._setup_path(P)
        self.corners_s = [float(self.cum[k]) for k in range(1, len(P) - 1)]
        self.s_corner = self.corners_s[0]; self.corner = (float(P[1][0]), float(P[1][1]))
        self.tracker_start = (float(start[0]), float(start[1] + behind))
        self.v0 = float(self.rng.uniform(40.0, 60.0)); self.v = self.v0; self.s = 0.0; self.sprinting = False
        self.sa_list = [self.A_MAX if self.profile == 'p2' else float(self.rng.uniform(-self.A_MAX, self.A_MAX)) for _ in self.corners_s]
        self.sa = self.sa_list[0]
        self.gates = True; self.cells = cells
        return True

    def plan_random(self, length=4800.0, behind=250.0, outside=1.5, to_exit=False, beta=0.7):
        """Random walk ('rw' profile): enter at the top-left gate, then at every junction pick
        a random open neighbour (no immediate backtrack; turn back only in a dead end), until
        the route is `length` cm long.  The target wanders: no goal, no fixed route.
        Speed (speed_update, 'rw'): random acceleration held 1-3 s, speed continuous (inertia)."""
        info = self.geo.info; ny, cell = info['ny'], info['cell']
        cur = (0, ny - 1); prev = None; cells = [cur]
        top = info['y0'] + ny * cell; bot = info['y0']; nx = info['nx']
        start = (cell_center(info, 0, ny - 1)[0], top + outside * cell)
        if to_exit:
            # random walk that still ends at the exit (bottom-right gate): at a junction each open
            # neighbour (no immediate backtrack) is chosen with weight exp(-beta * change of the
            # distance to the exit) -- detours and side branches happen, but the walk arrives
            goal = (nx - 1, 0); dist = {goal: 0}; q = [goal]
            while q:
                c = q.pop(0)
                for n2 in self.adj[c]:
                    if n2 not in dist:
                        dist[n2] = dist[c] + 1; q.append(n2)
            while cur != goal and len(cells) < 400:
                nb = [c for c in self.adj[cur] if c != prev] or self.adj[cur]
                w = np.array([math.exp(-beta * (dist[c] - dist[cur])) for c in nb]); w /= w.sum()
                nxt = nb[int(self.rng.choice(len(nb), p=w))]
                prev, cur = cur, nxt; cells.append(cur)
            end = (cell_center(info, nx - 1, 0)[0], bot - outside * cell)
            P = np.array([start] + [cell_center(info, i, j) for i, j in cells] + [end], float)
            length = 1e9                                                    # no cut: the walk ends outside the exit
        else:
            n_need = int(length / cell) + 2
            while len(cells) < n_need:
                nb = [c for c in self.adj[cur] if c != prev] or self.adj[cur]
                nxt = nb[int(self.rng.integers(len(nb)))]
                prev, cur = cur, nxt; cells.append(cur)
            P = np.array([start] + [cell_center(info, i, j) for i, j in cells], float)
        keep = [0]
        for k in range(1, len(P) - 1):
            d1 = P[k] - P[keep[-1]]; d2 = P[k + 1] - P[k]
            if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-6 or np.dot(d1, d2) < 0:
                keep.append(k)
        keep.append(len(P) - 1)
        P = P[keep]
        seg = np.hypot(*np.diff(P, axis=0).T); cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] > length:                                            # cut to the requested length
            i = int(np.searchsorted(cum, length) - 1); f = (length - cum[i]) / max(cum[i + 1] - cum[i], 1e-6)
            P = np.vstack([P[:i + 1], P[i] + f * (P[i + 1] - P[i])])
        self._setup_path(P)
        self.corners_s = [float(self.cum[k]) for k in range(1, len(P) - 1)]
        self.s_corner = self.corners_s[0] if self.corners_s else 0.0
        self.corner = tuple(map(float, P[1])) if len(P) > 2 else tuple(map(float, P[-1]))
        self.tracker_start = (float(start[0]), float(start[1] + behind))
        self.v0 = float(self.rng.uniform(40.0, 80.0)); self.v = self.v0; self.s = 0.0; self.sprinting = False
        self.t_next = float(self.rng.uniform(1.0, 3.0)); self.a_rw = float(self.rng.uniform(-self.A_MAX, self.A_MAX))
        self.sa_list = [0.0 for _ in self.corners_s]; self.sa = 0.0
        self.gates = True; self.cells = cells
        return True

    def done(self):
        return self.s >= self.cum[-1] - 1e-6

    def speed_update(self, dt):
        """Gated-route speed profile at the current arc length s (manoeuvre after every
        corner, relax to v0 over the last 150 cm before the next one)."""
        if self.profile == 'rw':
            # inertia: the ACCELERATION is random, the speed is its integral (continuous).  Every
            # 1-3 s a new acceleration U(-A_MAX, A_MAX) is drawn and held; at the speed limits
            # (V_FLOOR..V_MAX) it is reflected, as a walker settles back from too fast / too slow
            self.t_next -= dt
            if self.t_next <= 0.0:
                self.t_next = float(self.rng.uniform(1.0, 3.0)); self.a_rw = float(self.rng.uniform(-self.A_MAX, self.A_MAX))
            if (self.v >= self.V_MAX and self.a_rw > 0) or (self.v <= self.V_FLOOR and self.a_rw < 0):
                self.a_rw = -self.a_rw
            self.v = float(np.clip(self.v + self.a_rw * dt, self.V_FLOOR, self.V_MAX))
            return
        passed = [k for k, sc in enumerate(self.corners_s) if self.s >= sc]
        if passed:
            k = passed[-1]
            nxt = self.corners_s[k + 1] if k + 1 < len(self.corners_s) else None
            if nxt is not None and nxt - self.s < 150.0:                # slow down for the next turn
                self.v += float(np.clip(self.v0 - self.v, -self.A_MAX * dt, self.A_MAX * dt))
            else:
                self.v = float(np.clip(self.v + self.sa_list[k] * dt, self.V_FLOOR, self.V_MAX))

    def project(self, x, y, s_lo, s_hi):
        """Arc length of the route point closest to (x, y), searched in [s_lo, s_hi]."""
        best, sb = 1e18, self.s
        for i in range(len(self.P) - 1):
            a, b = self.cum[i], self.cum[i + 1]
            if b < s_lo or a > s_hi:
                continue
            p0, d = self.P[i], self.P[i + 1] - self.P[i]; L = max(b - a, 1e-6)
            t = float(np.clip(np.dot(np.array([x, y]) - p0, d) / L ** 2, 0.0, 1.0))
            t = float(np.clip(t, (s_lo - a) / L, (s_hi - a) / L))
            q = p0 + t * d; e = float(np.hypot(x - q[0], y - q[1]))
            if e < best:
                best, sb = e, a + t * L
        return sb, best

    def step(self, dt):
        if getattr(self, 'gates', False):
            self.speed_update(dt)
            self.s = min(self.s + self.v * dt, self.cum[-1])
            return self.point_at(self.s)
        if not self.sprinting and self.s >= self.s_corner:
            self.sprinting = True
        if self.sprinting:
            self.v = float(np.clip(self.v + self.sa * dt, self.V_FLOOR, self.V_MAX))
        self.s = min(self.s + self.v * dt, self.cum[-1])
        return self.point_at(self.s)


class AnimatedDriver:
    """Drive the maze target with its own movement component (walk/run animation plays)
    instead of teleporting it.  Each step: progress s = projection of the real pose on the
    route; speed from the same scripted profile (MazeTarget.speed_update); heading turned
    (rate-limited) towards a look-ahead point on the route; then `set_move 0 v` (one
    command lasts ~0.5 s, so it is re-sent every step; its yaw argument is unreliable,
    so the heading is set through the actor rotation).  Corners are rounded naturally."""
    def __init__(self, mt, env, look=70.0, yaw_rate=240.0):
        self.mt, self.env, self.look, self.yaw_rate = mt, env, look, yaw_rate
        x, y, yaw = mt.point_at(0.0); self.yaw = yaw
        self.dev = []                                                   # distance from the route centre line

    def step(self, dt, pose):
        mt = self.mt; x, y = pose[0], pose[1]
        s_new, e = mt.project(x, y, mt.s - 30.0, mt.s + 200.0)
        mt.s = min(max(mt.s, s_new), mt.cum[-1]); self.dev.append(e)
        mt.speed_update(dt)
        lx, ly, _ = mt.point_at(mt.s + self.look)
        want = math.degrees(math.atan2(ly - y, lx - x)) if math.hypot(lx - x, ly - y) > 5.0 else self.yaw
        d = (want - self.yaw + 180.0) % 360.0 - 180.0
        self.yaw += float(np.clip(d, -self.yaw_rate * dt, self.yaw_rate * dt))
        v = 0.0 if mt.done() else mt.v
        self.env.c.request([f'vset /object/{self.env.tgt}/rotation 0 {self.yaw:.2f} 0', f'vbp {self.env.tgt} set_move 0 {v:.1f}'], -1)


# ---------------- engine ----------------
def build_in_engine(env, rects, z_floor, prefix='mz'):
    c = env.c
    old = [o for o in env.api.get_objects() if o.startswith(prefix)]
    for o in old:
        c.request(f'vset /object/{o}/location 100000 100000 -5000')
    names = []
    for k, (x0, x1, y0, y1) in enumerate(rects):
        n = f'{prefix}{k}'
        if n not in old:
            c.request(f'vset /objects/spawn_from_path {CUBE} {n}')
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        c.request([f'vset /object/{n}/location {cx} {cy} {z_floor + WALL_H / 2.0}',
                   f'vset /object/{n}/rotation 0 0 0',
                   f'vset /object/{n}/scale {(x1 - x0) / 100.0} {(y1 - y0) / 100.0} {WALL_H / 100.0}'], -1)
        names.append(n)
    time.sleep(0.5)
    return names


def save_layout(name, rects, info):
    p = os.path.join(DATA, f'maze_{name}.json'); json.dump(dict(rects=rects, info=info), open(p, 'w'), indent=1); return p


def load_layout(name):
    d = json.load(open(os.path.join(DATA, f'maze_{name}.json'))); return [tuple(r) for r in d['rects']], d['info']


if __name__ == '__main__':
    import cv2
    import uz_env as U
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.patches as mp
    NX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rects, info = make_maze(NX, NX, 300.0, SEED)
    name = f'fr{NX}x{NX}_s{SEED}'
    print(f'maze {name}: {len(rects)} wall rectangles'); print('saved', save_layout(name, rects, info))
    OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_room')
    env = U.UZEnv('FlexibleRoom', sleep_time=30)
    try:
        # place the target in cell (0,0) first, to get the floor height
        cx0, cy0 = cell_center(info, 0, 0)
        env.spawn(tgt_xy=(cx0, cy0), trk_xy=(cx0, cy0 - 100), yaw=90.0)
        zf = env.z_floor; print('floor z (target origin):', zf)
        for o in ['Cube10', 'Cube9', 'Cube7_13', 'Cube8', 'FLOOR', 'Floor']:
            r = env.c.request(f'vget /object/{o}/bounds'); print('bounds', o, str(r)[:90])
        t0 = time.time(); build_in_engine(env, rects, zf - 90.0); print(f'built in {time.time() - t0:.1f}s')
        # top view
        c = env.c; W = info['nx'] * info['cell']
        c.request(f'vset /camera/0/location {ORIGIN[0]} {ORIGIN[1]} {zf + W * 0.62}'); c.request('vset /camera/0/rotation -90 0 0')
        time.sleep(0.8); cv2.imwrite(os.path.join(OUT, f'{name}_top.png'), env.api.get_image(0, 'lit'))
        # first-person from the tracker in cell (0,0) looking along +y
        geo = MazeGeo(rects, info)
        env.set_tracker(cx0, cy0 - 100, 90.0); time.sleep(0.6)
        cv2.imwrite(os.path.join(OUT, f'{name}_fp.png'), env.api.get_image(env.cam, 'lit'))
        box, npx, _, _ = env.frame(); print('target in first-person mask:', npx, box)
        fig, ax = plt.subplots(figsize=(7, 7))
        for (x0, x1, y0, y1) in rects:
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, color='k'))
        ax.plot(*cell_center(info, 0, 0), 'go'); ax.set_aspect('equal'); ax.autoscale()
        ax.set_title(f'{name}: {len(rects)} walls, cell {info["cell"]:.0f} cm, wall {WALL_T:.0f} cm')
        fig.savefig(os.path.join(HERE, '..', 'results', 'figures', f'maze_{name}.png'), dpi=100)
    finally:
        env.close()
