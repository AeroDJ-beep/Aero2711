"""Detector server for the tracking loop: YOLOE-11l-seg (OA-VAT's detector, open-vocabulary prompt
'person') runs here in the `follow` conda env; the simulation loop (`unrealzoo` env) sends each lit camera
image over a local socket and gets back the best person box and its segmentation-mask pixel count.
Run from D:\\Aero2711\\OA-VAT (yoloe-11l-seg.pt and mobileclip_blt.ts live there, nothing is downloaded).
With 'pose' as 3rd argument, YOLO11s-pose also runs and the person matching the YOLOE box (best IoU) gives
torso keypoints: neck = mid-shoulders, pelvis = mid-hips (u, v, confidence = min of the pair).
usage: python perception_server.py [port=6020] [conf=0.3] [pose]"""

import sys, time
import numpy as np
from multiprocessing.connection import Listener

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 6020
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

POSE = len(sys.argv) > 3 and sys.argv[3] == 'pose'
from ultralytics import YOLOE, YOLO  # noqa: E402

pose_model = YOLO('yolo11s-pose.pt') if POSE else None
model = YOLOE('yoloe-11l-seg.pt')
NAMES = ['person']
model.set_classes(NAMES, model.get_text_pe(NAMES))
model.predict(np.zeros((480, 640, 3), np.uint8), conf=CONF, imgsz=640, verbose=False)  # warm-up
if POSE:
    pose_model.predict(np.zeros((480, 640, 3), np.uint8), conf=CONF, imgsz=640, verbose=False)
print(f'[perception] YOLOE ready on port {PORT}, conf {CONF}, pose {POSE}', flush=True)


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


with Listener(('127.0.0.1', PORT), authkey=b'aero2711') as lst:
    while True:
        conn = lst.accept()
        print('[perception] client connected', flush=True)
        try:
            while True:
                shape, raw = conn.recv()  # raw bytes + shape: the two envs have different numpy versions
                img = np.frombuffer(raw, np.uint8).reshape(shape)
                t = time.perf_counter()
                r = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0]
                out = dict(box=None, conf=0.0, npx=0, ms=0.0, kp=None)
                if r.boxes is not None and len(r.boxes):
                    i = int(r.boxes.conf.argmax())
                    x0, y0, x1, y1 = r.boxes.xyxy[i].cpu().numpy().tolist()
                    out['box'] = [int(round(x0)), int(round(y0)), int(round(x1 - x0)) + 1, int(round(y1 - y0)) + 1]
                    out['conf'] = float(r.boxes.conf[i])
                    if r.masks is not None:
                        m = r.masks.data[i].cpu().numpy()
                        # mask comes at model resolution; scale its area to the 640x480 image
                        out['npx'] = int(m.sum() * (img.shape[0] * img.shape[1]) / (m.shape[0] * m.shape[1]))
                if POSE and out['box'] is not None:
                    rp = pose_model.predict(img, conf=CONF, imgsz=640, verbose=False)[0]
                    if rp.keypoints is not None and len(rp.boxes):
                        bx = out['box']
                        yb = [bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]]
                        ious = [_iou(yb, b) for b in rp.boxes.xyxy.cpu().numpy().tolist()]
                        j = int(np.argmax(ious))
                        if ious[j] > 0.3:
                            k = rp.keypoints.data[j].cpu().numpy()  # 17 x (u, v, conf), COCO order
                            nk, pv = (k[5] + k[6]) / 2.0, (k[11] + k[12]) / 2.0
                            out['kp'] = [
                                float(nk[0]),
                                float(nk[1]),
                                float(min(k[5, 2], k[6, 2])),
                                float(pv[0]),
                                float(pv[1]),
                                float(min(k[11, 2], k[12, 2])),
                            ]
                out['ms'] = (time.perf_counter() - t) * 1000.0
                conn.send(out)
        except (EOFError, ConnectionResetError, OSError):
            print('[perception] client disconnected', flush=True)
