# -*- coding: utf-8 -*-
"""Per-re-acquisition breakdown from reacq_ev (uz_run): how long the target was hidden,
how big the belief error was, and whether it is along the target's motion (speed /
timing: + = belief ahead) or across it (wrong corridor / direction).
usage: python analyze_reacq_ev.py <jsonl>"""
import json, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
by = {}
for r in rows:
    for e in r.get('reacq_ev') or []:
        by.setdefault(r['setting'], []).append(dict(e, seed=r['seed']))
for g, E in by.items():
    b = np.array([e['blind'] for e in E]); err = np.array([e['e'] for e in E])
    al = np.array([e['along'] for e in E]); cr = np.array([e['cross'] for e in E])
    vt = np.array([e['v_true'] for e in E]); ve = np.array([e['v_est'] for e in E])
    print(f'== {g}: {len(E)} events')
    print(f'   blind steps  mean {b.mean():.1f}  median {np.median(b):.0f}  max {b.max()}')
    print(f'   error cm     mean {err.mean():.0f}  median {np.median(err):.0f}')
    print(f'   along (+ahead) mean {np.nanmean(al):+.0f}  |along| {np.nanmean(np.abs(al)):.0f}   |cross| {np.nanmean(np.abs(cr)):.0f}')
    print(f'   speed true {vt.mean():.0f}  est {ve.mean():.0f} cm/s')
    for lo, hi in ((5, 10), (10, 20), (20, 40), (40, 999)):
        m = (b >= lo) & (b < hi)
        if m.any():
            print(f'   blind {lo:3d}-{hi:<3d}: n {m.sum():2d}  err {err[m].mean():4.0f}  along {np.nanmean(al[m]):+5.0f}  |cross| {np.nanmean(np.abs(cr[m])):4.0f}')
    big = sorted(E, key=lambda e: -e['e'])[:5]
    print('   worst:', [(e['seed'], e['k'], e['blind'], e['e'], e['along'], e['cross'], e['s']) for e in big])
