# -*- coding: utf-8 -*-
"""2-D occupancy map from UnrealCV+ mesh occupancy (shared-memory region).

grid, meta = occupancy_2d(client, (xmin, xmax, ymin, ymax), z_floor, voxel_cm=10,
                          z_band=(30, 150))
grid[iy, ix] is True where any voxel in the body-height slab is occupied.
World -> cell: ix = (x - xmin) / voxel_cm, iy = (y - ymin) / voxel_cm.
Also provides dist_to_occ() via a distance transform (cm) and push_dir() from
its gradient, mirroring the L-map helpers used by the safety layer.
Run as a script to build and plot the map of a scene."""
import os, sys, json, math, mmap, time
import numpy as np
import cv2


def occupancy_2d(client, xy_bounds, z_floor, voxel_cm=10.0, z_band=(30.0, 150.0), method='mesh', include_dynamic=0):
    xmin, xmax, ymin, ymax = xy_bounds
    zmin, zmax = z_floor + z_band[0], z_floor + z_band[1]
    ox, oy, oz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    v = voxel_cm / 100.0
    b = [(xmin - ox) / 100.0, (xmax - ox) / 100.0, (ymin - oy) / 100.0, (ymax - oy) / 100.0, (zmin - oz) / 100.0, (zmax - oz) / 100.0]
    cmd = 'vget /scene/occupancy_shared_region %s %s %g %g %g %g 0 %d' % (method, ' '.join('%g' % x for x in b), v, ox, oy, oz, include_dynamic)
    r = client.request(cmd, timeout=120)
    try:
        meta = json.loads(r)
    except Exception:
        print('occupancy request failed:', cmd, '->', str(r)[:300])
        raise
    size = int(meta['offset_bytes']) + int(meta['num_bytes'])
    shared = mmap.mmap(-1, size, tagname=meta['name'], access=mmap.ACCESS_READ)
    try:
        shape = tuple(int(s) for s in meta['shape'])
        g3 = np.ndarray(shape, dtype=np.bool_, buffer=shared, offset=int(meta['offset_bytes']), order='C').copy()
    finally:
        shared.close()
    # axes: shape = (nx, ny, nz) in region-local metres (yaw 0 => world-aligned)
    occ = g3.any(axis=2)            # (nx, ny)
    grid = occ.T.copy()             # (ny, nx) -> grid[iy, ix]
    mn = meta.get('min_meters'); mx = meta.get('max_meters')
    info = dict(xmin=ox + mn[0] * 100.0, ymin=oy + mn[1] * 100.0, xmax=ox + mx[0] * 100.0, ymax=oy + mx[1] * 100.0,
                voxel=voxel_cm, shape=grid.shape, cmd=cmd, meta_keys=list(meta.keys()))
    return grid, info


class OccMap:
    def __init__(self, grid, info):
        self.grid, self.info = grid, info
        self.v = info['voxel']; self.x0, self.y0 = info['xmin'], info['ymin']
        free = (~grid).astype(np.uint8)
        self.dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.v      # cm to nearest occupied cell
        gy, gx = np.gradient(self.dist)
        n = np.hypot(gx, gy) + 1e-9
        self.nx, self.ny = gx / n, gy / n

    def cell(self, x, y):
        ix = int(round((x - self.x0) / self.v)); iy = int(round((y - self.y0) / self.v))
        h, w = self.grid.shape
        return min(max(iy, 0), h - 1), min(max(ix, 0), w - 1)

    def dist_to_occ(self, x, y):
        iy, ix = self.cell(x, y); return float(self.dist[iy, ix])

    def push_dir(self, x, y):
        iy, ix = self.cell(x, y); return float(self.nx[iy, ix]), float(self.ny[iy, ix])

    def is_free(self, x, y, margin=0.0):
        return self.dist_to_occ(x, y) > margin

    def save(self, path):
        np.savez_compressed(path, grid=self.grid, **{k: v for k, v in self.info.items() if k not in ('cmd', 'meta_keys', 'shape')})

    @staticmethod
    def load(path):
        d = np.load(path); info = {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files if k != 'grid'}
        info['shape'] = d['grid'].shape
        return OccMap(d['grid'], info)


if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
    import uz_env as U
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    MAP = sys.argv[1] if len(sys.argv) > 1 else 'Hotel_Corridor'
    env = U.UZEnv(MAP)
    try:
        env.spawn()
        gx, gy, _ = env.target_pose()
        # scene extent from safe starts, padded
        S = np.array(env.starts); pad = 800.0
        xy = (S[:, 0].min() - pad, S[:, 0].max() + pad, S[:, 1].min() - pad, S[:, 1].max() + pad)
        t0 = time.time(); grid, info = occupancy_2d(env.c, xy, env.z_floor, voxel_cm=10.0)
        print('occupancy built in %.1fs shape %s occupied %.1f%%  keys %s' % (time.time() - t0, grid.shape, 100 * grid.mean(), info['meta_keys']))
        om = OccMap(grid, info)
        om.save(os.path.join(HERE, '..', 'env_config', 'data', f'occ_{MAP}.npz'))
        r = env.c.request('vget /reachablepoints')
        try:
            pts = np.array(json.loads(r).get('points', json.loads(r)), float)[:, :2]
        except Exception:
            pts = np.zeros((0, 2)); print('reachablepoints raw', str(r)[:100])
        print('cached reachable points:', len(pts))
        print('clearance at tgt spawn %.0f cm, at trk spawn %.0f cm' % (om.dist_to_occ(gx, gy), om.dist_to_occ(gx, gy - 250)))
        fig, ax = plt.subplots(figsize=(9, 12))
        ax.imshow(~grid, cmap='gray', origin='lower', extent=[info['xmin'], info['xmax'], info['ymin'], info['ymax']])
        if len(pts): ax.plot(pts[:, 0], pts[:, 1], '.', ms=1.5, color='#E07B00', label=f'navmesh points ({len(pts)})')
        ax.plot(S[:, 0], S[:, 1], 'b^', ms=6, label='safe starts')
        ax.set_title(f'{MAP}: mesh occupancy at body height (black), 10 cm cells'); ax.legend(loc='upper right'); ax.set_aspect('equal')
        p = os.path.join(HERE, '..', 'results', 'figures', f'uz_occ_{MAP}.png'); fig.savefig(p, dpi=110); print('saved', p)
    finally:
        env.close()
