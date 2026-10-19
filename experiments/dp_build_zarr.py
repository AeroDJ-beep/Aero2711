# -*- coding: utf-8 -*-
"""Build a replay zarr = original OA-VAT data (FlexibleRoom, 1464 samples) + maze A*
expert samples (results/dp_maze/*.npz), in the exact unreal_replay.zarr format:
data/img (N,240,360,3) uint8 RGB, data/bbox (N,4) normalised xywh, data/action (N,3) cm;
each sample = 8 rows (image/bbox duplicated, one waypoint per row); meta/episode_ends.
Runs in the `follow` env.  usage: python dp_build_zarr.py <out.zarr> [--maze-only]"""
import os, sys, glob
import numpy as np, zarr
from numcodecs import Blosc

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'OA-VAT', 'data', 'unreal_replay.zarr')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'OA-VAT', 'data', 'unreal_replay_maze.zarr')
MAZE_ONLY = '--maze-only' in sys.argv
comp = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

root = zarr.open(OUT, mode='w')
d = root.create_group('data'); m = root.create_group('meta')
img = d.zeros('img', shape=(0, 240, 360, 3), chunks=(8, 240, 360, 3), dtype='u1', compressor=comp)
bbox = d.zeros('bbox', shape=(0, 4), chunks=(4096, 4), dtype='f4')
act = d.zeros('action', shape=(0, 3), chunks=(4096, 3), dtype='f4')
ends = []
n_rows = 0
if not MAZE_ONLY:
    src = zarr.open(SRC, 'r')
    e_src = src['meta/episode_ends'][:]
    step = 800
    for a in range(0, len(src['data/img']), step):
        b = min(a + step, len(src['data/img']))
        img.append(src['data/img'][a:b]); bbox.append(src['data/bbox'][a:b]); act.append(src['data/action'][a:b])
    ends += list(e_src); n_rows = int(e_src[-1])
    print('original samples', len(e_src))
n_maze = 0
for f in sorted(glob.glob(os.path.join(HERE, '..', 'results', 'dp_maze', 'maze_*.npz'))):
    z = np.load(f)
    I, B, A = z['img'], z['bbox'], z['action']          # (n,240,360,3), (n,4), (n,8,3)
    img.append(np.repeat(I, 8, axis=0)); bbox.append(np.repeat(B, 8, axis=0)); act.append(A.reshape(-1, 3))
    for _ in range(len(I)):
        n_rows += 8; ends.append(n_rows)
    n_maze += len(I)
    print(os.path.basename(f), len(I))
m.array('episode_ends', np.array(ends, np.int64))
m.array('horizon', np.array(8, np.int64)); m.array('image_shape', np.array([240, 360, 3], np.int64))
print(f'wrote {OUT}: {len(ends)} samples ({n_maze} maze), {n_rows} rows')
