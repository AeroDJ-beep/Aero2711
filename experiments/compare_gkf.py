# -*- coding: utf-8 -*-
"""Ground-plane KF vs. current best (image KF + clean-frame velocity), same seeds.
Visible phase: position / velocity error of the floor estimate; occlusion start: velocity
error and whether the corridor predictor starts on the right axis; re-acquisition error.
usage: python compare_gkf.py results/metrics/uz_sweep_groundkf.jsonl"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon

R = [json.loads(l) for l in open(sys.argv[1])]
G = {}
for r in R:
    G.setdefault(r['setting'], {})[r['seed']] = r
names = sorted(G)
print(f"{'setting':20s} {'n':>2s} {'exit':>5s} {'vis pos cm':>10s} {'vis vel cm/s':>12s} {'start verr':>10s} {'axis ok':>8s} {'reacq cm':>9s}")
for g in names:
    S = G[g]; ev = [e for r in S.values() for e in r.get('init_ev') or []]
    print(f"{g:20s} {len(S):2d} {sum(r.get('finished', 0) for r in S.values()):3d}/{len(S):<2d}"
          f" {np.mean([r['vis_pos_cm'] for r in S.values()]):10.1f} {np.mean([r['vis_vel_cms'] for r in S.values()]):12.1f}"
          f" {np.mean([e['verr'] for e in ev]):10.1f} {np.mean([e['axis_ok'] for e in ev]):8.2f}"
          f" {np.mean([r['reacq_all_cm'] for r in S.values() if r['reacq_all_cm'] is not None]):6.1f}±{np.std([r['reacq_all_cm'] for r in S.values() if r['reacq_all_cm'] is not None]):.0f}")
REF = 'astar_T1_corr_cv2'
for a in [n for n in names if n != REF]:
    b = REF
    common = sorted(set(G[a]) & set(G[b]))
    for key in ('reacq_all_cm', 'vis_pos_cm', 'vis_vel_cms'):
        d = [G[a][s][key] - G[b][s][key] for s in common if G[a][s][key] is not None and G[b][s][key] is not None]
        p = wilcoxon(d).pvalue if len(d) >= 6 and any(d) else float('nan')
        print(f'{key:13s} {a} - {b}: mean {np.mean(d):+.1f}, {a} lower in {sum(x < 0 for x in d)}/{len(d)}, p={p:.3g}')
