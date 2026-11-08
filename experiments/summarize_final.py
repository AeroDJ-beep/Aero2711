"""Summary table of the final evaluation (14 configurations x 20 episodes).

Reads the per-episode results of the presets `rwfinal` and `rwfinalkpf` and prints, per
configuration: success with a Wilson 95% interval, mean progress, time out of view with a
bootstrap 95% interval, score per step, and (real perception only) missed / false detections.
usage: python summarize_final.py [out.csv]"""

import os, sys, json, math, random

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, '..', 'results', 'metrics')
FILES = ['uz_sweep_rwfinal.jsonl', 'uz_sweep_rwfinalkpf.jsonl']


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def bootstrap_mean(x, reps=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


groups = {}
for f in FILES:
    for line in open(os.path.join(M, f)):
        r = json.loads(line)
        groups.setdefault(r['setting'].rsplit('_r', 1)[0], []).append(r)  # merge repeats r1 / r2

rows = []
for name, E in groups.items():
    n = len(E)
    k = sum(e.get('finished', 0) for e in E)
    lo, hi = wilson(k, n)
    hidden = [100 * e['occl'] / e['EL'] for e in E]
    hlo, hhi = bootstrap_mean(hidden)
    gt_vis = sum(e.get('gt_vis') or 0 for e in E)
    miss = f'{100 * sum(e["det_miss"] for e in E) / gt_vis:.1f}%' if gt_vis else '-'
    false = str(sum(e['det_false'] for e in E)) if gt_vis else '-'
    rows.append(
        [
            name,
            n,
            f'{k}/{n}',
            f'{100 * lo:.0f}%-{100 * hi:.0f}%',
            f'{100 * sum(e["progress"] for e in E) / n:.0f}%',
            f'{sum(hidden) / n:.1f}%',
            f'{hlo:.1f}%-{hhi:.1f}%',
            f'{sum(e["AR"] / e["EL"] for e in E) / n:.3f}',
            miss,
            false,
        ]
    )

head = [
    'config',
    'episodes',
    'success',
    'success_ci',
    'progress',
    'hidden',
    'hidden_ci',
    'score',
    'det_miss',
    'det_false',
]
for r in [head] + rows:
    print(f'{r[0]:24s}' + ''.join(f'{str(c):>14s}' for c in r[1:]))
if len(sys.argv) > 1:
    with open(sys.argv[1], 'w') as fo:
        fo.write('\n'.join(','.join(map(str, r)) for r in [head] + rows) + '\n')
