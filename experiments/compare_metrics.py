# -*- coding: utf-8 -*-
"""Compare two or more metrics JSON files side by side (per profile x mode):
AR, pixel reacq, world reacq (cm), range error (%), hits.
usage: python compare_metrics.py label1=path1.json label2=path2.json ..."""
import json, sys, statistics as st


def load(path):
    return json.load(open(path))


def cell(rows, key, absval=False):
    v = [r[key] for r in rows if r.get(key) is not None]
    if absval:
        v = [abs(x) for x in v]
    return (st.mean(v), len(v)) if v else (float('nan'), 0)


runs = [(a.split('=', 1)[0], load(a.split('=', 1)[1])) for a in sys.argv[1:]]
profs = sorted({r['profile'] for _, d in runs for r in d})
modes = [m for m in ['none', 'cv', 'ca', 'cv_r01', 'ca_r01'] if any(r['mode'] == m for _, d in runs for r in d)]
hdr = f"{'prof':5}{'mode':7}" + ''.join(f"{lab:>34}" for lab, _ in runs)
print(hdr)
print(f"{'':12}" + ''.join(f"{'AR   px  cm  dr%  hits':>34}" for _ in runs))
for p in profs:
    for m in modes:
        line = f"{p:5}{m:7}"
        for lab, d in runs:
            rows = [r for r in d if r['profile'] == p and r['mode'] == m]
            if not rows:
                line += f"{'-':>34}"; continue
            ar = st.mean(r['AR'] for r in rows)
            px, n = cell(rows, 'reacq'); cm, _ = cell(rows, 'reacq_cm'); dr, _ = cell(rows, 'reacq_dr', True)
            hits = sum(r['hits'] for r in rows)
            line += f"   {ar:6.1f} {px:5.1f} {cm:5.1f} {dr:5.1f} {hits:4d}(n{n:2d})"
        print(line)
