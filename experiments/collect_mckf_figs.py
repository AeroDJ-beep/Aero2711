# -*- coding: utf-8 -*-
"""
Collect real data for the two MC-KF illustrations (2026-09-30):
  - first-person view of a fade-out frame (camera image, cut-off box,
    occluder edge column, repaired centre, true centre);
  - top view of a blind frame (camera pose, KF prediction before and after
    the map projection, true target position).
Full system with CV, random target motion, 30 cm margin. Does not modify
run_metrics_test.py; it only wraps functions at run time.
"""
import os
import sys
import json
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics_test as m     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'logs', 'mckf_figs_dyn')
os.makedirs(OUT, exist_ok=True)
SEEDS = [int(a) for a in sys.argv[1:]] or [2000, 2001, 2003, 2004, 2005, 2006, 2007]

state = {'k': -1, 'save': set(), 'seed': None}
CLAMP = []


def wrap_read(ucv):
    orig = ucv.read_image

    def read_image(cam, kind, mode='fast'):
        img = orig(cam, kind, mode)
        if kind == 'object_mask':
            state['k'] += 1
            if state['k'] in state['save']:
                lit = orig(1, 'lit', 'fast')
                cv2.imwrite(os.path.join(OUT, f"s{state['seed']}_k{state['k']:03d}_lit.png"), lit)
                cv2.imwrite(os.path.join(OUT, f"s{state['seed']}_k{state['k']:03d}_mask.png"), img)
        return img
    ucv.read_image = read_image


orig_clamp = m.map_clamp


def backproject(mean, x, y, yaw):
    ang = math.degrees(math.atan2(mean[0] - m.CX0, m.FX))
    rel = math.radians(yaw + ang)
    rho = m.FX * m.H_PERSON / max(mean[3], 1.0)
    return x + rho * math.cos(rel), y + rho * math.sin(rel)


def map_clamp(mean, cam_x, cam_y, yaw_deg):
    pre = backproject(mean, cam_x, cam_y, yaw_deg)
    mean, clamped = orig_clamp(mean, cam_x, cam_y, yaw_deg)
    post = backproject(mean, cam_x, cam_y, yaw_deg)
    CLAMP.append(dict(k=state['k'], cam=(cam_x, cam_y, yaw_deg), pre=pre, post=post, clamped=bool(clamped)))
    return mean, clamped


m.map_clamp = map_clamp


def run(ucv, z_trk, seed, save=()):
    state.update(k=-1, save=set(save), seed=seed)
    CLAMP.clear()
    m.DEBUG_EDGE = []
    log = []
    res = m.run_episode(ucv, z_trk, 'cv', seed, 'rand', steplog=log)
    return res, list(m.DEBUG_EDGE), list(CLAMP), log


def main():
    runner, ucv, z_trk = m.boot()
    wrap_read(ucv)
    for k, v in dict(DYN=True, VCAP=False, EDGEMEAS=True, EDGEAIM=True, BLINDSPEED=True, MAPCLAMP=True).items():
        setattr(m, k, v)
    summary = []
    try:
        for seed in SEEDS:
            res, edge, clamp, log = run(ucv, z_trk, seed)
            fades = [e for e in edge if e[3] >= 0]
            moved = [c for c in clamp if c['clamped']]
            dmax = max([math.dist(c['pre'], c['post']) for c in moved], default=0.0)
            summary.append(dict(seed=seed, reacq=res['reacq'], n_fade=len(fades), n_clamp=len(moved), clamp_max_cm=dmax))
            print(summary[-1]); sys.stdout.flush()
            if fades and moved:
                ks = [int(e[0]) for e in fades]
                res2, edge2, clamp2, log2 = run(ucv, z_trk, seed, save=ks)
                dump = dict(seed=seed, res=res2,
                            edge=[[float(x) for x in e] for e in edge2],
                            clamp=clamp2,
                            steplog=[[float(v) for v in r] for r in log2])
                with open(os.path.join(OUT, f's{seed}.json'), 'w') as f:
                    json.dump(dump, f)
    finally:
        with open(os.path.join(OUT, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        runner.close()


if __name__ == '__main__':
    main()
