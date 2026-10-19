# -*- coding: utf-8 -*-
"""Final 20-episode comparison table with paired statistics (same seeds = same target
speed profile).  Re-acquisition error = mean belief error at the frame the target is
seen again, over every occlusion of the episode.  AR per step = AR / EL.
usage: python final_stats.py <jsonl> [<jsonl> ...]"""
import os, json, sys, statistics as st
from scipy.stats import wilcoxon

rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p)]
groups = {}
for r in rows:
    groups.setdefault(r['setting'], {})[r['seed']] = r
REF = os.environ.get('REF', 'ego_wp')
order = ['ref_full', 'ego_wp', 'astar_T1', 'astar_T6', 'dp_orig_T6', 'dp_pad0_T6', 'dp_maze_T6', 'ego_wp_corr', 'astar_T1_corr', 'ego_wp_corr_ou', 'astar_T1_corr_ou', 'ego_wp_corr_cv', 'astar_T1_corr_cv']
print(f"{'setting':12s} {'n':>2s} {'exit':>6s} {'occl':>9s} {'AR/step':>13s} {'reacq cm':>11s} {'median':>7s}  vs {REF}: paired diff, better/n, p")
for g in order:
    if g not in groups:
        continue
    G = groups[g]; seeds = sorted(G)
    fin = sum(G[s].get('finished', 0) for s in seeds)
    occl = [G[s]['occl'] for s in seeds]; arps = [G[s]['AR'] / G[s]['EL'] for s in seeds]
    rq = [G[s]['reacq_all_cm'] for s in seeds if G[s].get('reacq_all_cm') is not None]
    line = (f"{g:12s} {len(seeds):2d} {fin:>3d}/{len(seeds):<2d} {st.mean(occl):5.0f}±{st.pstdev(occl):3.0f} "
            f"{st.mean(arps):6.3f}±{st.pstdev(arps):.3f} {st.mean(rq):5.0f}±{st.pstdev(rq):3.0f} {st.median(rq):7.0f}")
    if g != REF and REF in groups:
        common = [s for s in seeds if s in groups[REF] and G[s].get('reacq_all_cm') is not None and groups[REF][s].get('reacq_all_cm') is not None]
        d = [G[s]['reacq_all_cm'] - groups[REF][s]['reacq_all_cm'] for s in common]
        better = sum(x < 0 for x in d)
        p = wilcoxon(d).pvalue if len(d) >= 6 and any(d) else float('nan')
        line += f"   {st.mean(d):+6.0f} cm, {better}/{len(d)}, p={p:.3g}"
    print(line)
