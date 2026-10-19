# -*- coding: utf-8 -*-
"""Is the diffusion planner out of distribution in the maze?
Runs the policy (follow env) on (a) training samples from the replay zarr and
(b) maze frames, and compares the structure of the outputs with the training
labels (first waypoint = (0,0), lateral detour, final point straight ahead)."""
import os, sys, glob
import numpy as np, cv2, torch, zarr
HERE = os.path.dirname(os.path.abspath(__file__))
OAVAT = os.path.join(HERE, '..', 'OA-VAT'); sys.path.insert(0, OAVAT)
from dp.dp_inference import _load_dp_policy, build_dp_obs_from_frame

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OAVAT, 'data', 'dp_run', 'checkpoints', 'epoch=0025-val_loss=0.0212.ckpt')
print('checkpoint:', os.path.basename(os.path.dirname(os.path.dirname(CKPT))), os.path.basename(CKPT))
policy, cfg, dev = _load_dp_policy(CKPT, os.path.join(OAVAT, 'gym-unrealcv', 'diffusion_policy-main'))
z = zarr.open(os.path.join(OAVAT, 'data', 'unreal_replay.zarr'), 'r')
ends = z['meta/episode_ends'][:]; starts = np.r_[0, ends[:-1]]
rng = np.random.default_rng(0)


def run(img_rgb_or_bgr, bbox_xywh, is_rgb=False):
    img = cv2.cvtColor(img_rgb_or_bgr, cv2.COLOR_RGB2BGR) if is_rgb else img_rgb_or_bgr
    with torch.no_grad():
        a = policy.predict_action(build_dp_obs_from_frame(img, bbox_xywh, cfg, dev))['action'][0].cpu().numpy()
    return a[:, :2]


def summary(name, rows):
    R = np.array(rows)       # (N, 8, 2)
    first = np.hypot(R[:, 0, 0], R[:, 0, 1]); maxlat = np.abs(R[:, :, 1]).max(1); last_fwd = R[:, -1, 0]
    print(f'{name:34s} n={len(R):3d}  first-point |xy| p50 {np.median(first):5.0f} cm | max lateral detour p50 {np.median(maxlat):5.0f} '
          f'p90 {np.percentile(maxlat, 90):5.0f} cm | final forward p50 {np.median(last_fwd):5.0f} cm')


# (a) training samples: image + bbox + label
pred, lab = [], []
for e in rng.choice(len(ends), 40, replace=False):
    s = starts[e]
    img = z['data/img'][s]; bb = z['data/bbox'][s]; H, W = img.shape[:2]
    bbox = bb * np.array([W, H, W, H]) if bb.max() <= 1.5 else bb
    pred.append(run(img, bbox, is_rgb=True)); lab.append(z['data/action'][s:ends[e], :2])
summary('training labels (A* expert)', lab)
summary('policy on training images', pred)
# (b) maze frames (raw first-person lit images) with a box at the occluder edge
maze_imgs = [p for p in glob.glob(os.path.join(HERE, '..', 'results', 'logs', 'uz_maze', '**', 'k*.png'), recursive=True)][:30]
maze_imgs += glob.glob(os.path.join(HERE, '..', 'results', 'logs', 'uz_room', 'fr10x10*_fp.png'))
rows = []
for p in maze_imgs:
    im = cv2.imread(p)
    if im is None:
        continue
    im = cv2.resize(im, (640, 480))
    for bb in [(250, 140, 90, 300), (430, 140, 90, 300), (110, 140, 90, 300)]:
        rows.append(run(im, bb))
summary(f'policy on maze frames ({len(maze_imgs)} imgs x 3 boxes)', rows)
print('example maze output:', np.round(rows[0]).tolist())
print('example train label:', np.round(lab[0]).tolist())
print('example train pred :', np.round(pred[0]).tolist())
