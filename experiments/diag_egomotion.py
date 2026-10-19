# -*- coding: utf-8 -*-
"""Diagnose whether camera yaw motion contaminates the KF's pixel-space
acceleration estimate (ax) before occlusion.  Wraps the CA filter to record
its state every step; no algorithm change.  Output: figure + numbers."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_metrics_test as M
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SEEDS = [(2005, 'p2a'), (2000, 'p2'), (2003, 'p2')]
OUT = os.path.join(HERE, '..', 'results', 'figures')
M.DYN = M.EDGEMEAS = M.EDGEAIM = M.BLINDSPEED = M.MAPCLAMP = True
M.EDGEGEO = '--edgegeo' in sys.argv
M.EDGEGEO_BEHAV = '--geobehav' in sys.argv
for a in sys.argv:
    if a.startswith('--fader=auto'): M.FADE_RAUTO = True
    elif a.startswith('--fader='): M.FADE_RSCALE = float(a.split('=')[1])
    if a.startswith('--tmplmin='): M.TMPL_MINFRAC = float(a.split('=')[1])
    if a.startswith('--qvel='): M.QVEL = float(a.split('=')[1])
    if a.startswith('--wmodel='): M.WMODEL = a.split('=')[1]
    if a.startswith('--fademeas='): M.FADEMEAS = a.split('=')[1]
TAG = ('_geo' if M.EDGEGEO else '') + ('B' if M.EDGEGEO_BEHAV else '') + (f'_r{M.FADE_RSCALE:g}' if M.FADE_RSCALE != 1 else '') + (f'_w{M.WMODEL}' if M.WMODEL else '') + (f'_{M.FADEMEAS}' if M.FADEMEAS != 'edge' else '') + ('_rauto' if M.FADE_RAUTO else '') + (f'_tm{M.TMPL_MINFRAC:g}' if M.TMPL_MINFRAC != 0.4 else '') + (f'_qv{M.QVEL:g}' if M.QVEL else '')
SEEDS = [(2005, 'p2a')] if '--one' in sys.argv else SEEDS

rec = []
KF = M.KFCATightR
_p, _u, _i = KF.predict, KF.update, KF.initiate
def wp(self, mean, cov, coasting=False):
    m, c = _p(self, mean, cov, coasting); rec.append(('p', m.copy())); return m, c
def wu(self, mean, cov, meas):
    m, c = _u(self, mean, cov, meas); rec.append(('u', m.copy())); return m, c
def wi(self, meas, init_velocity=None):
    m, c = _i(self, meas, init_velocity); rec.append(('i', m.copy())); return m, c
KF.predict, KF.update, KF.initiate = wp, wu, wi


def analyse(log, seed, prof):
    n = len(log)
    # per-step final KF mean: walk rec; a step = one 'i' or one 'p' (+ optional 'u')
    means = [None] * n; k = -1; j = 0
    # steps before first KF call have mean None; find first step with have=True
    first = next(i for i, r in enumerate(log) if r[8])
    k = first - 1
    for kind, m in rec:
        if kind in ('i', 'p'):
            k += 1
        if k < n:
            means[k] = m
    yaw = np.array([r[2] for r in log]); det = np.array([r[5] for r in log], bool)
    tx = np.array([r[0] for r in log]); ty = np.array([r[1] for r in log])
    Px = np.array([r[3] for r in log]); Py = np.array([r[4] for r in log])
    bear = np.degrees(np.arctan2(Py - ty, Px - tx))
    rel = (bear - yaw + 180) % 360 - 180
    cx_true = M.CX0 + M.FX * np.tan(np.radians(rel))
    kf_cx = np.array([m[0] if m is not None else np.nan for m in means])
    kf_vx = np.array([m[4] if m is not None else np.nan for m in means])
    kf_ax = np.array([m[8] if m is not None and len(m) > 8 else np.nan for m in means])
    w = np.gradient(yaw) / M.DT                     # deg/s
    # ego-induced pixel shift per frame: du = -fx*sec^2(rel)*dyaw
    ego_du = -M.FX * (1 + np.tan(np.radians(rel)) ** 2) * np.radians(np.gradient(yaw))
    cx_comp = cx_true - np.cumsum(ego_du)           # what cx would be with a fixed camera
    # first blind frame
    blind = np.where(~det)[0]
    b0 = int(blind[0]) if len(blind) else n
    w0, w1 = max(0, b0 - 40), b0
    def acc(x):  # pixel acceleration per frame^2 via 2nd difference
        return np.diff(x, 2)
    a_raw = acc(cx_true[w0:w1]); a_comp = acc(cx_comp[w0:w1]); a_ego = acc(-np.cumsum(ego_du)[w0:w1])
    kfa = kf_ax[w0:w1]
    ok = ~np.isnan(kfa[2:])
    r_kf_ego = np.corrcoef(kfa[2:][ok], a_ego[ok])[0, 1] if ok.sum() > 3 else np.nan
    r_kf_raw = np.corrcoef(kfa[2:][ok], a_raw[ok])[0, 1] if ok.sum() > 3 else np.nan
    print(f'seed {seed} {prof}: blind starts step {b0}; window {w0}-{w1}')
    print(f'   pixel-accel RMS  raw cx {np.sqrt(np.mean(a_raw**2)):.2f}   ego part {np.sqrt(np.mean(a_ego**2)):.2f}   compensated {np.sqrt(np.mean(a_comp**2)):.2f}  (px/frame^2)')
    print(f'   share of raw pixel-accel variance explained by ego-motion: {1 - np.var(a_comp)/np.var(a_raw):.0%}')
    print(f'   corr(KF ax, ego accel) = {r_kf_ego:.2f}   corr(KF ax, raw accel) = {r_kf_raw:.2f}')
    print(f'   yaw rate in window: mean |w| {np.mean(np.abs(w[w0:w1])):.1f} deg/s, max {np.max(np.abs(w[w0:w1])):.1f}')
    print(f'   KF ax at blind onset {kf_ax[b0-1]:.3f} px/frame^2; ego accel at onset {a_ego[-1]:.3f}; compensated target accel {a_comp[-1]:.3f}')
    sys.stdout.flush()
    # --- clipping band: when does the box touch the occluder edge? ---
    half_w = 0.5 * np.array([m[2] * m[3] if m is not None else 60.0 for m in means])
    eu = np.array([(M.edge_u(r[0], r[1], r[2]) or np.nan) for r in log])
    gap = eu - cx_true                     # + : edge is to the right of the target centre
    clip = np.abs(gap) < half_w            # box overlaps the occluder edge column
    clip_start = int(np.where(clip[:b0])[0][0]) if clip[:b0].any() else b0
    err = kf_cx - cx_true
    dev_start = next((i for i in range(w0, b0) if abs(err[i]) > 10), b0)
    print(f'   box starts touching occluder edge at step {clip_start} ({b0-clip_start} frames before blind);'
          f' |KF cx - true| exceeds 10 px from step {dev_start} ({b0-dev_start} before blind);'
          f' error at blind onset {err[b0-1]:+.1f} px')
    if M.DEBUG_EDGE:
        H = [(kk, bl, br, cxc) for (kk, bl, br, u, wg, cxc, npx, rh, g) in M.DEBUG_EDGE if not rh and not g and 40 < kk < b0]
        eb = np.array([br - cx_true[kk] for kk, bl, br, cxc in H]); el = np.array([cx_true[kk] - bl for kk, bl, br, cxc in H])
        ec = np.array([cxc - cx_true[kk] for kk, bl, br, cxc in H])
        print(f'   HEALTHY frames n={len(H)}: right half-width mean {eb.mean():.1f} sd {eb.std():.1f} range [{eb.min():.0f},{eb.max():.0f}];'
              f' left half-width mean {el.mean():.1f} sd {el.std():.1f}; box-centre minus true cx mean {ec.mean():+.1f} sd {ec.std():.1f}')
        last = [(kk, bl, br) for kk, bl, br, cxc in H][-12:]
        print('   last healthy: ' + ' '.join(f'{kk}:R{br - cx_true[kk]:.0f}/L{cx_true[kk] - bl:.0f}' for kk, bl, br in last))
        F = [(kk, bl, br, u, wg, cxc) for (kk, bl, br, u, wg, cxc, npx, rh, g) in M.DEBUG_EDGE if (rh or g) and kk < b0]
        if F:
            print('   FADE frames: clean-edge half-width (br - true cx): ' + ' '.join(f'{kk}:{br - cx_true[kk]:.0f}' for kk, bl, br, u, wg, cxc in F))
            accd = [(kk, cxc - cx_true[kk]) for kk, bl, br, u, wg, cxc in F if not np.isnan(cxc)]
            print(f'   fade meas accepted {len(accd)}/{len(F)}; errors: ' + ' '.join(f'{kk}:{e:+.0f}' for kk, e in accd))
    acc_k = {kk for kk, e in accd} if M.DEBUG_EDGE and F else set()
    print('   KF trace  k: true_cx  kf_cx  err  kf_vx  kf_ax  [M=meas accepted, .=coast]')
    print('   ' + '  '.join(f'{kk}:{cx_true[kk]:.0f}/{kf_cx[kk]:.0f}/{kf_cx[kk]-cx_true[kk]:+.0f}/{kf_vx[kk]:+.2f}/{kf_ax[kk]:+.3f}{"M" if kk in acc_k else "."}' for kk in range(max(0, b0 - 22), min(n, b0 + 2))))
    print(f'   visible fraction at first |err|>10 step: {min(1.0, (half_w[dev_start] + abs(gap[dev_start])) / (2*half_w[dev_start])):.2f}'
          f'   KF vx at onset {kf_vx[b0-1]:+.2f} px/frame, true dcx/dt {np.mean(np.diff(cx_true[b0-6:b0])):+.2f}, compensated {np.mean(np.diff(cx_comp[b0-6:b0])):+.2f}')
    np.savez(os.path.join(HERE, '..', 'results', 'logs', f'diag_seed{seed}_{prof}{TAG}.npz'), log=np.array(log, float),
             means=np.array([m if m is not None else np.full(10, np.nan) for m in means]))
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = np.arange(n)
    ax[0].plot(t, cx_true, 'k', lw=1, label='true cx (pixel)'); ax[0].plot(t, kf_cx, 'r--', lw=1, label='KF cx')
    ax[0].plot(t, cx_comp, 'b', lw=1, alpha=.7, label='ego-compensated cx (camera held fixed)')
    ax[0].plot(t, eu, 'm:', lw=1, label='occluder edge column'); ax[0].axvline(clip_start, color='m', ls='--', lw=.8)
    ax[0].set_ylabel('cx (px)'); ax[0].legend(fontsize=8, loc='upper left')
    ax[1].plot(t, w, 'g'); ax[1].set_ylabel('yaw rate w (deg/s)')
    ax[2].plot(t[2:], acc(cx_true), 'k', lw=.8, label='raw pixel accel'); ax[2].plot(t[2:], acc(cx_comp), 'b', lw=.8, label='compensated')
    ax[2].plot(t[2:], acc(-np.cumsum(ego_du)), 'g', lw=.8, alpha=.7, label='ego (camera) part'); ax[2].set_ylabel('px/frame²'); ax[2].legend(fontsize=8)
    ax[3].plot(t, kf_ax, 'r', lw=1); ax[3].set_ylabel('KF ax (px/frame²)'); ax[3].set_xlabel('step')
    for a in ax:
        for s0 in np.where(np.diff(det.astype(int)) == -1)[0]: a.axvline(s0 + 1, color='0.6', ls=':')
        a.set_xlim(max(0, b0 - 60), min(n, b0 + 40)); a.grid(alpha=.3)
    fig.suptitle(f'seed {seed} {prof} ca_r01 - is KF ax driven by camera yaw?  (dotted = occlusion onset)')
    fig.tight_layout(); p = os.path.join(OUT, f'diag_egomotion_seed{seed}_{prof}{TAG}.png'); fig.savefig(p, dpi=120); print('   saved', p)


if __name__ == '__main__':
    runner, ucv, z_trk = M.boot()
    try:
        for seed, prof in SEEDS:
            rec.clear(); log = []; M.DEBUG_EDGE = []
            res = M.run_episode(ucv, z_trk, 'ca_r01', seed, prof, steplog=log); print('   result', {k: res[k] for k in ('AR','hits','fade','reacq')})
            analyse(log, seed, prof)
    finally:
        try: runner.close()
        except Exception: pass
