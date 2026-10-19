# Simulation Environment Configuration (v2)

Project 2711 — occlusion-aware visual active tracking at indoor corners.
All quantities engine-verified unless marked *nominal*. Units: cm, s, px. Last updated: 2026-08-24.

---

## 1. Simulator stack

| Layer | Component |
|---|---|
| 3D world | FlexibleRoom (packaged UE4 binary, `FlexibleRoom_Win_v0`) |
| Remote control | UnrealCV plugin (TCP, text commands) |
| RL wrapper | gym-unrealcv (editable install, Windows fixes applied) |
| Timing regime | Real-time asynchronous — the engine never waits for computation |

## 2. Arena (measured via `get_obj_bounds`)

- Floor slab 30.0 × 30.0 m, floor top at z = 100.
- Physical interior ≈ 29.1 × 28.3 m (x ∈ [−1925, 985], y ∈ [−1494, 1339]); boundary walls 12 m tall, no ceiling (skylight).
- Design area (reset_area, keep all geometry inside): x ∈ [−1800, 900], y ∈ [−1400, 1200].
- 15 movable blocks (5 cubes, 5 cylinders, 5 cones; note the block named `cube2_7`, not `cube2`). Pivot at bottom center; place on floor top z = 100.
- Runtime-built walls carry collision but are invisible to the pre-baked navmesh → target locomotion must use explicit waypoints, never GoalNav pathfinding.

## 3. Track: L-corner (corridor width 150)

Wall thickness 100, wall height 350. Centerlines: south leg x = −510, east leg y = 320.
Corridor width **W = 150** (interior faces: south leg x ∈ [−585, −435]; east leg y ∈ [245, 395]).
Inner corner point: (−435, 245).

Verified wall bounds (engine readback):

| Block | x range | y range |
|---|---|---|
| cube1 (west wall, south) | −685..−585 | −530..70 |
| cube5 (west wall, north) | −685..−585 | 70..395 |
| cube2_7 (east wall, south leg) | −435..−335 | −530..145 |
| cube4 (east-leg south wall) | −435..40 | 145..245 |
| cube3 (north end wall) | −710..−35 | 395..495 |

**Target route (wall-hugging, clearance c = 40 from inner faces):**
entry (−475, −450) → corner (−475, 285) → exit (60, 285).
Rationale: a fleeing person cuts the corner tight; c = 40 respects the body's collision capsule.

## 4. Target

- Single person (`targetbp2` in the eval pipeline; `targetbp` is the tracker's own body — do not confuse). Distractors = 0 (player_num = 2 is the hard minimum).
- Height **168 cm** (scaled 0.85×; standing z = 183.6 = floor 100 + half-height 83.6). Original model: 198 cm, animated bbox ≈ 98 × 90.
- Speed profiles: P0 constant 80 cm/s; P2 corner attack — visible phase unchanged, occlusion-triggered sprint +4 cm/s per step capped at **99 cm/s** (= camera max; control-variable rule) with sharp turn ±36°/step × 2–4 steps; P3 slow ramp a = 8 cm/s² (offline ½at² studies only).
- Target turning: ~0.10–0.15 s for 90° in attack mode; ~0.3–0.6 s under normal navigation.

## 5. Camera (tracker)

- Mounted on the tracker avatar: body-frame offset (forward 30, lateral 0, up 60); optical center **158 cm above floor** (world z ≈ 258).
- Pitch **−15° fixed** (chosen so the target is near-centered vertically at 250 cm: offset ≈ 24 px low, foot margin 77 px; −5° left only 12 px of foot margin. Justification: quadrotor forward-flight attitude 10–20°.) Roll 0; yaw = body heading (no gimbal).
- hFOV **82.6°** (*nominal* Tello spec, treated as horizontal; real-Tello calibration deferred — one ROS calibration reports fx ≈ 921 @ 960×720 → hFOV ≈ 55°, so the spec value is generous).
- Resolution 640 × 480 (4:3, same aspect as Tello). Zero distortion (synthetic pinhole).
- Intrinsics: fx = fy = 364.3, cx = 320, cy = 240; vFOV 66.8°. Pixel↔cm at 250 cm: 1 px ≈ 0.69 cm.
- Kinematics: unicycle (v, ω); v ≤ **99 cm/s**, ω ≤ **30°/s**. Min turn radius R = v/ω → 189 cm at full speed → cornering requires slowing to ≈ 78 cm/s (emergent, like a car).
- Camera count: cam0 = overhead PawnSensor (quirky, photos only), cam1 = tracker's eye (perception + mask ground truth, same pose), cam2 = target's eye (unused).

## 6. Following distance & control

- Following distance **d = 250 cm** — OUR experiment's setting (final, 2026-08-24). Rationale: on the open L-corner this distance natively guarantees the full occlusion sequence — fade-out (partial, 19 steps) THEN fully-blind (31 steps) — with no props, engine-verified; it also sits exactly on the scoring optimum (zero standing deduction). The 140 option (+ screen occluder) was considered and retired.
- Distance control: bbox-area PID. Pixel-space "satisfied" band with the 1.0-scale legacy calibration: area 30,720 ± 3,072 px² (0.1 ± 10% of frame). For the 0.85 target at 250 cm the derived setpoint is **0.078** of frame area (formula ratio = f²·H·W/(d²·A) × cos(pitch); calibrate empirically, formula for design).
- Yaw control: error_x = bbox center − 320, no deadband, PID output clipped ±99.
- Vertical control disabled; vertical image position is not scored.
- **Tracker anti-collision margin (calibrated 2026-09-05, replaces the 40 cm placeholder):** the constraint point is the **camera** (body + 30 cm forward offset), not the body. Render-breaking mechanism: UE near-clip plane ≈ 10.5 cm — a wall's front face inside it is fully clipped (back faces culled), producing total see-through. Engine sweep: facing-wall leaks at camera clearance ≤ 10 cm, clean at ≥ 12; parallel/oblique clean down to 5. Rule: **camera-point clearance ≥ 15 cm** (any heading) + body point ≥ 5 cm outside walls. Effective corridor lane widens 70 → 120 cm for parallel travel; facing a wall the body implicitly needs 45 cm. Data: `data/margin_calib/margin_calib*.json` + renders.
- Known proxy noise (engine-measured, standing pose, full 360° sweep at 250 cm): silhouette area ±14%; detector-style bbox area −27%/+28% (minima at oblique 120°/300° — NOT at profile; the walking-pose model is widest side-on because limbs spread along the walking direction). The ±10% tolerance band absorbs most pivot-induced false distance signals (worst-case creep ≈ 25 cm; a 90° front-to-profile pivot moves the camera ≈ 3 cm only). Data: `data/aspect_sweep.csv`.

## 7. Timing

- Control period **50 ms (20 Hz)**, enforced by pad-to-period sleep so ctrl_dt = 0.05 is factual; per-step wall-clock timestamps logged. KF's implicit dt = 1 frame ≡ 50 ms.
- Latency is real (asynchronous regime): actions execute under zero-order hold while the next frame is processed.

## 8. Occlusion geometry (engine-verified)

- **Leak rule** (open corner): the wall blocks the line of sight only if **d > 4c**. With c = 40: d = 140 → no occlusion (verified); d = 250 → occluded (verified).
- **At our d = 250 setting** the open L-corner occludes natively (250 > 4c = 160): both occlusion states occur in sequence every episode. A blind-corner / screen variant (`build_Lb`) remains an optional harder configuration for longer windows, not a requirement.
- **Window length formula: L_occ = √(d(d − 4c))** target-travel cm. d = 250, c = 40 → 150 cm ≈ 1.88 s ≈ **38 steps** at 80 cm/s (mask sweep, camera advancing along path). Both formula predictions matched the engine within one 25 cm sample.
- Three measured phases (d = 250, c = 40, W = 150, 80 cm/s):
  1. occlusion onset 0.5 s after the turn (body starts clipping);
  2. fade-out (partial → zero pixels): 0.94 s ≈ 19 steps — bbox center biases toward the visible side here ("deathbed contamination" of the KF's final velocity estimate; log and split in analysis);
  3. fully blind (zero pixels → reappearance): 1.56 s ≈ 31 steps; reappearance is step-like (~10⁴ px at once) so the re-acquisition instant is sharp.
- Closed-loop caution: a kinematic pursuit simulation (reactive steering, wall sliding, no recovery planner) gives first blind windows of **51 steps** (no-prediction, steer to vanishing point) vs **72 steps** (CV-coast steering, gets pinned against the inner wall) — i.e., borderline-lethal vs the 50-step deadline, and *no-prediction beats CV-coast* at this corner (mechanistic reproduction of the ParkingLot corner-battle result). Real closed-loop verdict pending.

## 9. Evaluation protocol (frozen, inherited)

- Episode: 500 steps (25 s). Visibility rule: ρ ≤ 750 cm AND |bearing| ≤ 45° (pure geometry — blind to walls; occlusion ground truth comes from the object mask instead: occluded ⇔ target mask pixels < 20, with hysteresis exit > 50 and an in-FOV check to separate "occluded" from "out-of-view").
- Termination: 50 consecutive invisible steps (2.5 s). Kept frozen; an extended 100-step deadline may be added later as a separately-reported `--fail_streak` parameter for long-window sweeps only.
- Score per step: r = 1 − |ρ − 250|/750 − |θ|/45 (full marks: true distance exactly 250 and target on the heading axis; vertical position and the target's own facing angle are score-irrelevant). AR = per-episode sum (max ≈ 500); EL = steps survived; SR = 1 iff 500 steps reached and visible at the last step. 20 episodes per setting (seed 101), 3 seeds for defended conclusions.
- New mechanism metric: **re-acquisition error** — at the first reappearance step, distance between the predictor's extrapolated bbox center and the true target position projected through K (world → camera → pixel chain).

## 9b. Formal 20-seed result (2026-09-06)

- Setup: seeds 2000–2019, modes {none, cv, ca, cv_r01, ca_r01} × target profiles {random-speed, P2 corner sprint}, paired. r01 = R scaled ×0.1 to match the measured oracle sigma (0.0 px static). Camera platform: inverse mount (camera-point vehicle, zero rotation radius); safety margin camera ≥ 15 cm.
- AR/EL/SR ceiling-saturated throughout (451–455 / 500 / 1.00); the re-acquisition error is the discriminating mechanism metric.
- **Main result — CA is conditionally effective**: with sprint target AND calibrated R, ca_r01 beats cv_r01 by 10.8 px (15/20 paired wins) and beats standard-R ca by 28.6 px (18/20). With the inherited R, ca vs cv = +3.7 px (4/20): nothing. In the random-speed regime ca_r01 vs cv_r01 is a coin flip (no true acceleration → no contribution, as theory requires).
- Universal lever: R matched to measured sigma (cv_r01 − cv ≈ −14 px in both regimes).
- Interpretation: the bottleneck is estimation lag at the disappearance instant, not the motion model; remaining best-case error 169 px ≈ 1.2 m is structural → motivates map-constrained prediction.
- Data: `../results/metrics/metrics_formal_20seeds_2026-09-06.json`, `../results/metrics/metrics_cvr01_control_2026-09-06.json` (+ earlier rounds `metrics_smoke_2026-09-04`, `metrics_round3_calibrated_margin`, `metrics_round4_inverse_mount`, `metrics_tightR_ablation`).

## 10. Open items

- Real-Tello camera calibration (checkerboard) at the sim-to-real stage; sim keeps nominal 82.6°.
- Aspect sweep used the standing pose; walking-swing numbers will come free from closed-loop logs.
- Closed-loop verification of the corner window (implementation queue: env hooks → measurement infra + logging → CA Kalman → 4-mode showdown on the L track).
- Optional harder variant: blind-corner block (c → 0) for 2–3 s windows; wide-corridor maze as the final integrated test (15-block budget).

## Data files (this folder)

- `data/aspect_sweep.csv` — bbox w/h and silhouette area vs body yaw (15° grid, 250 cm).
- `data/L_occlusion_sweep.csv` — naive path-following sweep, W=300 (conflates out-of-view).
- `data/L_occlusion_sweep_w150.csv` — same at W=150.
- `data/L_occlusion_sweep_w150_facing.csv` — camera facing target (pure wall occlusion): none on centerline.
- `data/L_occlusion_sweep_w150_hug.csv` — wall-hugging route, d=140: no occlusion (leak rule).
- `data/L_occlusion_sweep_w150_hug250.csv` — wall-hugging route, d=250: 150 cm window (formula match).

Figures (kept in `OA-VAT/data/maps/`, copies in `D:\Aero2711`): `pipeline_reference.*`, `L_scene_topview.*`, `pursuit_policy_compare.*`, `L_floorplan.png`.
