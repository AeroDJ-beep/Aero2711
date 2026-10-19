# -*- coding: utf-8 -*-
"""OA-VAT expert planner (A* on an inflated occupancy grid, receding horizon),
run online with the maze map the UAV already has.

Original expert (collect.py): occupancy grid (50 cm cells, walls inflated 80 cm),
A* from the UAV to a goal at follow_dist (250 cm) in front of the target along the
UAV->target line, path shortcut, then 8 waypoints.  In a maze that goal is often
inside a wall or out of sight after the target turns a corner, so here the goal is
re-defined: the free cell 150-330 cm from the predicted target position with a
clear line of sight to it, minimising  path length + 0.5 * |distance - 250|.
Grid: 10 cm cells, walls inflated by the safety margin (30 cm)."""
import math, heapq
import numpy as np
import uz_maze as Z

CELL = 20.0


class MazePlanner:
    def __init__(self, geo, margin=30.0, cell=CELL):
        self.geo, self.cell, self.margin = geo, cell, margin
        inf = geo.info; p = geo.pad
        self.x0, self.y0 = inf['x0'] - p, inf['y0'] - p
        self.nx = int((inf['nx'] * inf['cell'] + 2 * p) / cell); self.ny = int((inf['ny'] * inf['cell'] + 2 * p) / cell)
        xs = self.x0 + (np.arange(self.nx) + 0.5) * cell; ys = self.y0 + (np.arange(self.ny) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys)                                  # [iy, ix]
        R = geo.R
        dx = np.maximum(np.maximum(R[:, 0][:, None, None] - X, 0), X - R[:, 1][:, None, None])
        dy = np.maximum(np.maximum(R[:, 2][:, None, None] - Y, 0), Y - R[:, 3][:, None, None])
        clr = np.hypot(dx, dy).min(axis=0)
        self.free = clr >= margin                                   # inflated occupancy (False = blocked)
        self.X, self.Y = X, Y

    def cell_of(self, x, y):
        return (min(max(int((y - self.y0) / self.cell), 0), self.ny - 1), min(max(int((x - self.x0) / self.cell), 0), self.nx - 1))

    def xy(self, c):
        return self.x0 + (c[1] + 0.5) * self.cell, self.y0 + (c[0] + 0.5) * self.cell

    def _nearest_free(self, c):
        if self.free[c]:
            return c
        ys, xs = np.where(self.free)
        k = int(np.argmin((ys - c[0]) ** 2 + (xs - c[1]) ** 2)); return (int(ys[k]), int(xs[k]))

    def dijkstra(self, start_xy, max_cost=1200.0):
        """Path cost (cm) from start to every free cell, 8-connected; returns cost, parent."""
        s = self._nearest_free(self.cell_of(*start_xy))
        cost = np.full(self.free.shape, np.inf); cost[s] = 0.0
        parent = {}; pq = [(0.0, s)]
        nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        while pq:
            d, c = heapq.heappop(pq)
            if d > cost[c] or d * self.cell > max_cost:
                continue
            for dy, dx, w in nb:
                n = (c[0] + dy, c[1] + dx)
                if 0 <= n[0] < self.ny and 0 <= n[1] < self.nx and self.free[n]:
                    if dy and dx and not (self.free[c[0] + dy, c[1]] and self.free[c[0], c[1] + dx]):
                        continue                                    # no corner cutting
                    nd = d + w
                    if nd < cost[n]:
                        cost[n] = nd; parent[n] = c; heapq.heappush(pq, (nd, n))
        return cost * self.cell, parent, s

    def plan(self, uav_xy, tgt_xy, d_set=250.0, dmin=150.0, dmax=330.0):
        """Goal = visible viewpoint near the predicted target; returns (path [(x,y)...], goal) or (None, None)."""
        cost, parent, s = self.dijkstra(uav_xy)
        D = np.hypot(self.X - tgt_xy[0], self.Y - tgt_xy[1])
        cand = np.where(self.free & np.isfinite(cost) & (D >= dmin) & (D <= dmax))
        if len(cand[0]) == 0:
            return None, None
        score = cost[cand] + 0.5 * np.abs(D[cand] - d_set)
        order = np.argsort(score)
        goal = None
        for k in order[:400]:                                       # best-scoring candidates with line of sight
            c = (int(cand[0][k]), int(cand[1][k]))
            if not Z.los_blocked(self.geo, self.xy(c), tgt_xy):
                goal = c; break
        if goal is None:
            return None, None
        path = [goal]
        while path[-1] != s:
            path.append(parent[path[-1]])
        P = [self.xy(c) for c in path[::-1]]
        return self._shortcut(P), self.xy(goal)

    def _shortcut(self, P):
        """Greedy line-of-sight shortcut on the inflated grid (as collect.shortcut_path)."""
        if len(P) <= 2:
            return P
        out = [P[0]]; i = 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1 and not self._seg_free(P[i], P[j]):
                j -= 1
            out.append(P[j]); i = j
        return out

    def _seg_free(self, a, b):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.5)))
        for t in np.linspace(0, 1, n):
            if not self.free[self.cell_of(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]:
                return False
        return True


def pursue(path, pos, lookahead=60.0):
    """Pure-pursuit carrot on a polyline: the point `lookahead` cm ahead of the closest point."""
    P = np.array(path, float)
    if len(P) == 1:
        return tuple(P[0])
    seg = np.diff(P, axis=0); L = np.hypot(seg[:, 0], seg[:, 1]); cum = np.r_[0, np.cumsum(L)]
    best, s_best = 1e18, 0.0
    for i in range(len(seg)):
        if L[i] < 1e-6:
            continue
        t = np.clip(np.dot(np.array(pos) - P[i], seg[i]) / L[i] ** 2, 0, 1)
        q = P[i] + t * seg[i]; d = np.hypot(*(np.array(pos) - q))
        if d < best:
            best, s_best = d, cum[i] + t * L[i]
    s = min(s_best + lookahead, cum[-1])
    i = min(int(np.searchsorted(cum, s, side='right') - 1), len(seg) - 1)
    f = (s - cum[i]) / max(L[i], 1e-6)
    return tuple(P[i] + f * seg[i])
