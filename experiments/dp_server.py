# -*- coding: utf-8 -*-
"""OA-VAT diffusion-policy planner (module 3) as a local service.

Runs in the `follow` conda env (torch / hydra / diffusion_policy).  The maze
pipeline (`unrealzoo` env, different numpy) connects over a local socket and
sends {'img': bytes, 'shape': (H, W, 3), 'bbox': (x, y, w, h)}; the server returns
the predicted action sequence as a plain list of [dx, dy, dz] (cm, body frame of
the UAV at planning time: x forward, y right, z up) -- exactly what the policy was
trained on (A* expert paths, collect.py world_abs_to_body0).
usage: python dp_server.py [port]"""
import os, sys, time
import numpy as np
from multiprocessing.connection import Listener

HERE = os.path.dirname(os.path.abspath(__file__))
OAVAT = os.path.join(HERE, '..', 'OA-VAT')
sys.path.insert(0, OAVAT)
from dp.dp_inference import _load_dp_policy, build_dp_obs_from_frame   # noqa: E402
import torch                                                              # noqa: E402

CKPT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OAVAT, 'data', 'dp_run', 'checkpoints', 'epoch=0025-val_loss=0.0212.ckpt')
REPO = os.path.join(OAVAT, 'gym-unrealcv', 'diffusion_policy-main')
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 6010
print('[dp_server] checkpoint', CKPT, flush=True)

policy, cfg, device = _load_dp_policy(CKPT, REPO)
if policy is None:
    print('[dp_server] failed to load policy', flush=True); sys.exit(1)
dummy = np.zeros((240, 360, 3), np.uint8)
with torch.no_grad():
    policy.predict_action(build_dp_obs_from_frame(dummy, (144, 96, 72, 48), cfg, device))
print(f'[dp_server] ready on 127.0.0.1:{PORT} (horizon {policy.horizon}, n_action_steps {policy.n_action_steps})', flush=True)

with Listener(('127.0.0.1', PORT), authkey=b'oavat') as lst:
    while True:
        conn = lst.accept()
        try:
            while True:
                msg = conn.recv()
                if msg == 'quit':
                    conn.close(); sys.exit(0)
                img = np.frombuffer(msg['img'], np.uint8).reshape(msg['shape'])
                t0 = time.time()
                with torch.no_grad():
                    out = policy.predict_action(build_dp_obs_from_frame(img, tuple(msg['bbox']), cfg, device))
                act = out['action'][0].detach().cpu().numpy().astype(float).tolist()
                conn.send({'action': act, 'ms': (time.time() - t0) * 1000.0})
        except (EOFError, ConnectionResetError):
            conn.close()
