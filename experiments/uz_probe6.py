# -*- coding: utf-8 -*-
"""Probe 6: why does the target stall near y=-777 on Hotel_Corridor?
Drive it north with both nav APIs, poll status, capture its own camera when it
stalls, try opening doors, and check the tracker placement helper."""
import os, sys, time, math, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_smoke')
env = U.UZEnv('Hotel_Corridor', sleep_time=25)
try:
    env.spawn(); nav = env.build_navgrid(); c = env.c
    cams = c.request('vget /cameras').split(); cam_tgt = next(i for i, n in enumerate(cams) if 'CID-tgt-' in n)
    x, y, yaw = env.place_tracker_behind(); print('tracker placed at (%.0f,%.0f) yaw %.0f clearance %.0f' % (x, y, yaw, nav.clearance(x, y)))
    env.set_target_speed(80)
    goal = (0.0, 500.0)
    print('bypath ->', str(c.request(f'vbp tgt nav_to_goal_bypath {goal[0]} {goal[1]} {env.z_floor}'))[:80].replace('\n', ' '))
    last = None; stall = 0
    for i in range(40):
        time.sleep(0.5)
        gx, gy, gyaw = env.target_pose(); st = env.nav_status().replace('\n', ' ')[:90]
        if last is not None and math.hypot(gx - last[0], gy - last[1]) < 1.0:
            stall += 1
        else:
            stall = 0
        last = (gx, gy)
        if i % 4 == 0 or stall == 3:
            print(f'  t={i * 0.5:4.1f}s tgt=({gx:.0f},{gy:.0f}) yaw {gyaw:.0f} clr {nav.clearance(gx, gy):.0f} | {st}')
        if stall == 3:
            cv2.imwrite(os.path.join(OUT, 'stall_tgtcam_lit.png'), env.api.get_image(cam_tgt, 'lit'))
            cv2.imwrite(os.path.join(OUT, 'stall_tgtcam_mask.png'), env.api.get_image(cam_tgt, 'object_mask'))
            print('  stalled: saved target-camera frames; trying set_open_door + agent nav/goto')
            print('   open_door ->', str(c.request('vbp tgt set_open_door 1'))[:60].replace('\n', ' '))
            print('   nav/goto ->', str(c.request(f'vset /agent/tgt/nav/goto {goal[0]} {goal[1]} {env.z_floor}'))[:80])
        if stall >= 10:
            print('  still stalled after retries'); break
        if math.hypot(gx - goal[0], gy - goal[1]) < 100:
            print('  reached goal'); break
    # what objects are near the stall point?
    gx, gy, _ = env.target_pose()
    near = []
    for o in env.api.get_objects():
        r = c.request(f'vget /object/{o}/bounds'); v = U._floats(r) if isinstance(r, str) else []
        if len(v) == 6 and v[0] - 60 <= gx <= v[3] + 60 and v[1] - 200 <= gy + 150 <= v[4] + 200 and (v[3] - v[0]) < 400 and (v[4] - v[1]) < 400:
            near.append((o, [round(t) for t in v]))
    print('small objects around/ahead of the stall point:', near[:12])
finally:
    env.close()
