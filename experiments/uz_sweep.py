# -*- coding: utf-8 -*-
"""One-factor-at-a-time sweep of the A* planner parameters on the gated maze.
Boots the engine once, builds the maze once, runs every (setting, seed) and appends
one JSON line per episode to results/metrics/uz_sweep_astar.jsonl (resumable:
finished (setting, seed) pairs are skipped).
usage: python uz_sweep.py [seeds=3]"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U, uz_run as R

MAZE = 'fr10x10l35t24g_s0'
SEEDS = ([int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 and ',' in sys.argv[1]
         else list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)))   # '5' -> 0..4, '0,3,4' -> those seeds
PRESET = sys.argv[2] if len(sys.argv) > 2 else 'oat'
OUT = os.path.join(HERE, '..', 'results', 'metrics', 'uz_sweep_astar.jsonl' if PRESET == 'oat' else f'uz_sweep_{PRESET}.jsonl')
BASE = dict(astar=True, T=3, E=5, D=250.0, L=60.0)
SETTINGS = [('base', {}), ('ego_only', dict(astar=False)),
            ('T1', dict(T=1)), ('T6', dict(T=6)),
            ('E2', dict(E=2)), ('E10', dict(E=10)),
            ('D200', dict(D=200.0)), ('D300', dict(D=300.0)),
            ('L30', dict(L=30.0)), ('L120', dict(L=120.0))]
if PRESET == 'confirm':                       # best one-factor values combined vs. no planner, more seeds
    SETTINGS = [('best_T6_L30', dict(T=6, L=30.0)), ('ego_only', dict(astar=False))]
if PRESET == 'dp':                            # learned planners in the maze: original vs pad_after=0 retrain
    SETTINGS = [('dp_pad0_T6', dict(astar=False, dp='pad0', T=6)), ('dp_orig_T6', dict(astar=False, dp='orig', T=6))]
if PRESET == 'final':                         # 20-episode comparison, all with ego-motion comp. + world prediction
    WP = dict(ego=True, wp=True)
    SETTINGS = [('ego_wp', dict(WP, astar=False)), ('astar_T1', dict(WP, T=1, L=30.0)), ('astar_T6', dict(WP, T=6, L=30.0)),
                ('dp_pad0_T6', dict(WP, astar=False, dp='pad0', T=6)), ('dp_orig_T6', dict(WP, astar=False, dp='orig', T=6)),
                ('ref_full', dict(astar=False, ego=False, wp=False))]
if PRESET == 'final_maze':                    # the maze-trained diffusion planner, added once training is done
    SETTINGS = [('dp_maze_T6', dict(ego=True, wp=True, astar=False, dp='maze', T=6))]
if PRESET == 'diag':                          # per-re-acquisition events (reacq_ev) for the best setting and its baseline
    WP = dict(ego=True, wp=True)
    SETTINGS = [('astar_T1', dict(WP, T=1, L=30.0)), ('ego_wp', dict(WP, astar=False))]
if PRESET == 'corridor':                      # corridor-following multi-hypothesis blind prediction (20 seeds, same as 'final')
    WP = dict(ego=True, wp=True, corr=True)
    SETTINGS = [('astar_T1_corr', dict(WP, T=1, L=30.0)), ('ego_wp_corr', dict(WP, astar=False))]
if PRESET == 'corr_ou':                       # corridor prediction + mean-reverting blind speed
    WP = dict(ego=True, wp=True, corr=True, ou=True)
    SETTINGS = [('astar_T1_corr_ou', dict(WP, T=1, L=30.0)), ('ego_wp_corr_ou', dict(WP, astar=False))]
if PRESET == 'cleanvel':                      # corridor prediction + velocity from whole-body frames only
    WP = dict(ego=True, wp=True, corr=True, cv=True)
    SETTINGS = [('astar_T1_corr_cv', dict(WP, T=1, L=30.0)), ('ego_wp_corr_cv', dict(WP, astar=False))]
if PRESET == 'groundkf':                      # ground-plane KF with projected measurement noise vs. the current best
    WP = dict(ego=True, wp=True, corr=True, T=1, L=30.0)
    SETTINGS = [('astar_T1_corr_gkf', dict(WP, gkf=True)), ('astar_T1_corr_cv2', dict(WP, cv=True)),
                ('astar_T1_corr_gkfskip', dict(WP, gkf=True, gks=True))]
if PRESET == 'anim':                          # animated (walking) target instead of teleported, current best pipeline
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True)
    SETTINGS = [('anim_astar_T1_corr_cv', WP), ('anim_ego_wp_corr_cv', dict(WP, astar=False))]
if PRESET in ('rw', 'rwspd', 'rw3', 'rwchase', 'rwring', 'rwarc', 'rwflow', 'rwstop', 'rwff', 'rwfix', 'rwyaw', 'rwexit', 'rwkp', 'rwkp20', 'rwfix2', 'rwrex', 'rwbody', 'rwbody20', 'rwbody1', 'rwfix3', 'rwyawdyn', 'rwet', 'rwkino', 'rwkinob_stress', 'rwkinob', 'rwkinobg_stress'):                            # random walk + inertial random acceleration, walking (animated) target
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True)
    SETTINGS = [('rw_astar_T1_corr_cv', WP), ('rw_ego_wp_corr_cv', dict(WP, astar=False)),
                ('rw_astar_T1_corr_cv_spd', dict(WP, cs=(0.6, 1.0, 1.5)))]
if PRESET == 'rwspd':                         # speed hypotheses only (run first: base rw lost 3/3)
    SETTINGS = [s for s in SETTINGS if s[0].endswith('_spd')]
if PRESET == 'rw3':                           # look-ahead planner + U-turns + restart, on top of speed hypotheses
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.6, 1.0, 1.5))
    SETTINGS = [('rw_full', dict(WP, ah=1.0, ut=True, rs=True))]
if PRESET == 'rwchase':                       # user idea: while blind, close in on the prediction (40 / 100 cm)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.6, 1.0, 1.5))
    SETTINGS = [('rw_chase100', dict(WP, chase=100.0)),                 # user picked 100 cm (40 cm cuts the body out of view)
                ('rw_chase100_ib', dict(WP, chase=100.0, ib=True)), ('rw_spd_ib', dict(WP, ib=True))]
if PRESET == 'rwring':                        # only the user's ring rule (run after the rwchase preset)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.6, 1.0, 1.5))
    SETTINGS = [('rw_ring100_ib', dict(WP, ring=100.0, ib=True))]
if PRESET == 'rwarc':                         # ring rule without stopping: full speed in, then along the circle (next best view)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.6, 1.0, 1.5))
    SETTINGS = [('rw_ringarc100_ib', dict(WP, ring=100.0, arc=True, ib=True))]
if PRESET == 'rwflow':                        # ring rule, never stopping: along the route ~1 m behind the prediction
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.6, 1.0, 1.5))
    SETTINGS = [('rw_ringflow100_ib', dict(WP, ring=100.0, flow=True, ib=True))]
if PRESET == 'rwstop':                        # + a 'standing still' speed hypothesis (m = 0); end of walk counts as success
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True)
    SETTINGS = [('rw_spd_v2', dict(WP, cs=(0.6, 1.0, 1.5))), ('rw_spd_stop_ib2', dict(WP, cs=(0.0, 0.6, 1.0, 1.5), ib=True))]
if PRESET == 'rwff':                          # current best + feedforward tracking controller (reverse allowed)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True)
    SETTINGS = [('rw_best_v3', WP), ('rw_best_v3_ff', dict(WP, ff=True))]     # after the world-belief fix
if PRESET == 'rwfix':                         # best config after the frozen-belief fix (no wall-crossing hold with the corridor predictor)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True)
    SETTINGS = [('rw_best_v4_ff', WP)]
if PRESET == 'rwyaw':                         # best + yaw low-pass / yaw-acceleration limit (camera shake)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True)
    SETTINGS = [('rw_best_v4_ff_ys', dict(WP, ys=True))]
if PRESET == 'rwexit':                        # random walk from the entrance TO THE EXIT (detours), best config +/- yaw smoothing
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True)
    SETTINGS = [('rwx_best', WP), ('rwx_best_ys', dict(WP, ys=True))]
if PRESET == 'rwkp':                          # torso keypoints (column + neck-pelvis range) vs. box, exit-bound random walk
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True)
    SETTINGS = [('rwk_box', WP), ('rwk_kp', dict(WP, kpp=True, kpr=True))]
if PRESET == 'rwkp20':                        # success-rate estimate of the current best (keypoints): seeds x 2 repeats
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True)
    SETTINGS = [('best_kp_rep1', WP), ('best_kp_rep2', dict(WP))]
if PRESET == 'rwfix2':                        # after the sliver guard: the hard seeds again
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True)
    SETTINGS = [('best_kp_sliver_rep1', WP), ('best_kp_sliver_rep2', dict(WP))]
if PRESET == 'rwrex':                         # + re-expand when every hypothesis was seen empty (hard seeds)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True)
    SETTINGS = [('best_kp_rex_rep1', WP), ('best_kp_rex_rep2', dict(WP))]
if PRESET in ('rwbody', 'rwbody20'):          # same as rwrex, after the tracker-body fix (camera at capsule centre, mesh hidden, pose read back)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1)
    SETTINGS = [('best_body_rep1', WP), ('best_body_rep2', dict(WP))]
if PRESET == 'rwbody1':                       # one run per seed (new seeds: feasibility check)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1)
    SETTINGS = [('best_body', WP)]
if PRESET == 'rwkinobg_stress':               # recovery test with the A* guide in the kinodynamic search (KINO_GUIDE_W)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1, cf=True, rch=True, yi=True, kinob=True)
    SETTINGS = [('fix_yi_kinob_guide', WP)]
if PRESET == 'rwkinob_stress':                # recovery test: old visible controller (frequent losses) + kinodynamic search while hidden; compare with rwyawdyn
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1, cf=True, rch=True, yi=True, kinob=True)
    SETTINGS = [('fix_yi_kinob', WP)]
if PRESET == 'rwkinob':                       # full system: kinodynamic look-ahead while visible AND while hidden; compare with rwkino
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1, cf=True, rch=True, yi=True, et=True, kino=True, kinob=True)
    SETTINGS = [('fix_yi_et_kino_kinob', WP)]
if PRESET == 'rwkino':                        # look-ahead planner with the kinodynamic search; compare with rwet
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1, cf=True, rch=True, yi=True, et=True, kino=True)
    SETTINGS = [('fix_yi_et_kino', WP)]
if PRESET == 'rwet':                          # + Elastic-Tracker-style look-ahead planner while visible; compare with rwyawdyn
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1, cf=True, rch=True, yi=True, et=True)
    SETTINGS = [('fix_yi_et', WP)]
if PRESET == 'rwyawdyn':                      # + turning inertia (yaw rate first-order lag, tau 0.3 s); compare with rwfix3 fix_cf_reach
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1, cf=True, rch=True, yi=True)
    SETTINGS = [('fix_yawinertia', WP)]
if PRESET == 'rwfix3':                        # predictor fixes: came-from 'back' + reachable-set restart, vs. without (same seeds)
    WP = dict(ego=True, wp=True, corr=True, cv=True, T=1, L=30.0, anim=True, cs=(0.0, 0.6, 1.0, 1.5), ib=True, ff=True, kpp=True, kpr=True, rex=True, E=1)
    SETTINGS = [('base', WP), ('fix_cf_reach', dict(WP, cf=True, rch=True))]
CKPTS = {'pad0': r'D:\Aero2711\OA-VAT\data\dp_run_pad0\checkpoints\epoch=0150-val_loss=0.0052.ckpt', 'orig': None,
         'maze': os.environ.get('DP_MAZE_CKPT', '')}
PORTS = {'pad0': 6011, 'orig': 6010, 'maze': 6012}

done = set()
if os.path.isfile(OUT):
    for line in open(OUT):
        r = json.loads(line); done.add((r['setting'], r['seed']))
if PRESET in ('rw', 'rwspd', 'rw3', 'rwchase', 'rwring', 'rwarc', 'rwflow', 'rwstop', 'rwff', 'rwfix', 'rwyaw', 'rwexit', 'rwkp', 'rwkp20', 'rwfix2', 'rwrex', 'rwbody', 'rwbody20', 'rwbody1', 'rwfix3', 'rwyawdyn', 'rwet', 'rwkino', 'rwkinob_stress', 'rwkinob', 'rwkinobg_stress'):                            # random walk: UAV limits 1.5x the target (99 cm/s): 150 cm/s, 90 deg/s
    R.C.V_CAM_MAX, R.C.W_CAM_MAX, R.DP_YAW_MAX = 150.0, 90.0, 90.0
for f in R.FLAGS:
    R.FLAGS[f] = f in ('edgemeas', 'edgeaim', 'blindspeed', 'mapclamp', 'egocomp')
env = U.UZEnv('FlexibleRoom'); env.verbose = False
try:
    R.setup_maze(env, MAZE)
    for name, over in SETTINGS:
        p = dict(BASE, **over)
        R.FLAGS['astar'] = p['astar']; R.DP_TRIGGER = p['T']; R.ASTAR_EVERY = p['E']; R.ASTAR_DSET = p['D']; R.ASTAR_LOOK = p['L']
        R.FLAGS['dp'] = bool(p.get('dp'))
        R.FLAGS['egocomp'] = p.get('ego', True); R.FLAGS['worldpred'] = p.get('wp', False); R.FLAGS['corridor'] = p.get('corr', False); R.FLAGS['ouspeed'] = p.get('ou', False); R.FLAGS['cleanvel'] = p.get('cv', False); R.FLAGS['groundkf'] = p.get('gkf', False); R.GK_SKIP_UNCLEAN = p.get('gks', False); R.ANIM_TARGET = p.get('anim', False); R.CORR_SPEEDS = tuple(p.get('cs', (1.0,))); R.CORR_UTURN = p.get('ut', False); R.CORR_RESET = p.get('rs', False); R.ASTAR_AHEAD = p.get('ah', 0.0); R.CHASE_D = p.get('chase', 0.0); R.CORR_INIT_BRANCH = p.get('ib', False); R.RING_R = p.get('ring', 0.0); R.RING_ARC = p.get('arc', False); R.RING_FLOW = p.get('flow', False); R.FF_CTRL = p.get('ff', False); R.KP_POS = p.get('kpp', False); R.KP_RANGE = p.get('kpr', False); R.CORR_REEXPAND = p.get('rex', False); R.YAW_SMOOTH = p.get('ys', False); R.CORR_CAME_FROM = p.get('cf', False); R.CORR_REACH = p.get('rch', False); R.YAW_INERTIA = p.get('yi', False); R.ET_PLANNER = p.get('et', False); R.ET_KINO = p.get('kino', False); R.KINO_BLIND = p.get('kinob', False)
        if p.get('dp'):
            R.DP_PORT = PORTS[p['dp']]; R.DP_CKPT = CKPTS[p['dp']]; R.dp_connect()
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.time()
            steplog = []; R.CORR_LOG = []
            res = R.run_episode(env, 'ca', seed, 'rw' if PRESET in ('rw', 'rwspd', 'rw3', 'rwchase', 'rwring', 'rwarc', 'rwflow', 'rwstop', 'rwff', 'rwfix', 'rwyaw', 'rwexit', 'rwkp', 'rwkp20', 'rwfix2', 'rwrex', 'rwbody', 'rwbody20', 'rwbody1', 'rwfix3', 'rwyawdyn', 'rwet', 'rwkino', 'rwkinob_stress', 'rwkinob', 'rwkinobg_stress') else 'p2', steps=2000 if PRESET.startswith('rw') else 1200, log=steplog)
            res.update(setting=name, params=p, wall_s=round(time.time() - t0, 1))
            if PRESET in ('rwfix3', 'rwyawdyn', 'rwet', 'rwkino', 'rwkinob_stress', 'rwkinob', 'rwkinobg_stress'):      # per-step logs, so every loss can be traced afterwards
                import numpy as np
                LD = os.path.join(HERE, '..', 'results', 'logs', 'sweep_steps'); os.makedirs(LD, exist_ok=True)
                np.save(os.path.join(LD, f'{PRESET}_{name}_s{seed}.npy'), np.array(steplog, float))
                json.dump(R.CORR_LOG, open(os.path.join(LD, f'{PRESET}_{name}_s{seed}_corr.json'), 'w'))
            with open(OUT, 'a') as f:
                f.write(json.dumps(res) + '\n')
            print(f'[{name} seed {seed}] finished {res.get("finished", 0)} EL {res["EL"]} AR {res["AR"]} occl {res["occl"]} '
                  f'reacq_all {res.get("reacq_all_cm")} plans {res.get("dp_plans")}', flush=True)
finally:
    env.close()
print('SWEEP DONE', flush=True)
