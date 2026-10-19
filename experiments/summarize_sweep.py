# -*- coding: utf-8 -*-
"""Summarize results/metrics/uz_sweep_astar.jsonl per setting (mean +- spread over seeds)."""
import json, os, sys, statistics as st
from collections import OrderedDict

p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics', 'uz_sweep_astar.jsonl')
rows = [json.loads(l) for l in open(p)]
by = OrderedDict()
for r in rows:
    by.setdefault(r['setting'], []).append(r)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return '   -   '
    return f'{st.mean(v):6.0f}' + (f' ±{st.pstdev(v):4.0f}' if len(v) > 1 else '      ')


print(f"{'setting':10s} {'n':>2s} {'finished':>8s} {'AR':>12s} {'occl frames':>13s} {'reacq err cm':>13s} {'plans':>6s}  per-seed reacq")
for k, R in by.items():
    fin = sum(r.get('finished', 0) for r in R)
    print(f"{k:10s} {len(R):2d} {fin:>4d}/{len(R):<3d} {ms([r['AR'] for r in R]):>12s} {ms([r['occl'] for r in R]):>13s} "
          f"{ms([r.get('reacq_all_cm') for r in R]):>13s} {ms([r.get('dp_plans') for r in R]):>6s}  "
          f"{[r.get('reacq_all_cm') for r in sorted(R, key=lambda x: x['seed'])]}")
