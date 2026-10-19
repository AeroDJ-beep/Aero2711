# -*- coding: utf-8 -*-
"""Summarize a uz_run metrics JSON per mode (and per seed with -v)."""
import json, sys, statistics as st

d = json.load(open(sys.argv[1]))
verbose = '-v' in sys.argv
print('episodes', len(d))
for m in ['none', 'cv', 'ca', 'cv_r01', 'ca_r01']:
    R = [r for r in d if r['mode'] == m]
    if not R:
        continue
    rc = [r['reacq_cm'] for r in R if r.get('reacq_cm') is not None]
    print(f"{m:6s} AR {st.mean(r['AR'] for r in R):6.1f}  SR {st.mean(r['SR'] for r in R):.2f}  EL {st.mean(r['EL'] for r in R):5.0f}  "
          f"occl {st.mean(r['occl'] for r in R):4.0f}  hits {sum(r['hits'] for r in R):3d}  "
          f"reacq_cm {(st.mean(rc) if rc else float('nan')):6.1f} (n={len(rc)})")
    if verbose:
        for r in R:
            print(f"     seed {r['seed']} c {r.get('c')} AR {r['AR']} EL {r['EL']} SR {r['SR']} occl {r['occl']} reacq_cm {r.get('reacq_cm')}")
