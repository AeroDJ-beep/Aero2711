# -*- coding: utf-8 -*-
"""Phone-playable copy of a recorded episode: first-person and 2-D top view side by side,
H.264 (Windows Media Foundation encoder), scaled so the file stays under the 30 MB limit.
usage: python make_phone_video.py <video base path without _fp.mp4/_2d.mp4> <out.mp4> [height=360]"""
import os, sys
import cv2, numpy as np

base, out = sys.argv[1], sys.argv[2]
H = int(sys.argv[3]) if len(sys.argv) > 3 else 360
a, b = cv2.VideoCapture(base + '_fp.mp4'), cv2.VideoCapture(base + '_2d.mp4')
fps = a.get(cv2.CAP_PROP_FPS) or 8
vw = None; n = 0
while True:
    ok1, f1 = a.read(); ok2, f2 = b.read()
    if not (ok1 and ok2):
        break
    f1 = cv2.resize(f1, (int(f1.shape[1] * H / f1.shape[0]), H), interpolation=cv2.INTER_AREA)
    f2 = cv2.resize(f2, (int(f2.shape[1] * H / f2.shape[0]), H), interpolation=cv2.INTER_AREA)
    fr = np.hstack([f1, f2]); fr = fr[:, : fr.shape[1] // 2 * 2]
    if vw is None:
        vw = cv2.VideoWriter(out, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'avc1'), fps, (fr.shape[1], fr.shape[0]))
    vw.write(fr); n += 1
vw.release()
print(f'{out}: {n} frames, {fr.shape[1]}x{fr.shape[0]}, {os.path.getsize(out) / 1048576:.1f} MB')
