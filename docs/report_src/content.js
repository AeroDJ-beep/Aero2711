// Shared report content. Used by build_report.js (docx) and build_tex.js (LaTeX).
// Citations: [@key] or [@a,@b]; numbered by first appearance.
// Markup: **bold**, *italic*, ^sup^, ~sub~.

const REFDB = {
  oavat: 'H. Sun, K. Zhou, H. Gao, S. Zhang, J. Hu, X. Wen, Q. Ye, and M. Tan, "Instance-level visual active tracking with occlusion-aware planning," arXiv:2604.21453, 2026.',
  luo18: 'W. Luo, P. Sun, F. Zhong, W. Liu, T. Zhang, and Y. Wang, "End-to-end active object tracking via reinforcement learning," in *Proc. Int. Conf. Machine Learning (ICML)*, 2018, pp. 3286–3295.',
  luo20: 'W. Luo, P. Sun, F. Zhong, W. Liu, T. Zhang, and Y. Wang, "End-to-end active object tracking and its real-world deployment via reinforcement learning," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 6, pp. 1317–1332, 2020.',
  advat: 'F. Zhong, P. Sun, W. Luo, T. Yan, and Y. Wang, "AD-VAT: An asymmetric dueling mechanism for learning visual active tracking," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.',
  zhong21: 'F. Zhong, P. Sun, W. Luo, T. Yan, and Y. Wang, "Towards distraction-robust active visual tracking," in *Proc. Int. Conf. Machine Learning (ICML)*, 2021, pp. 12782–12792.',
  rspt: 'F. Zhong, X. Bi, Y. Zhang, W. Zhang, and Y. Wang, "RSPT: Reconstruct surroundings and predict trajectory for generalizable active object tracking," in *Proc. AAAI Conf. Artificial Intelligence*, 2023.',
  zhong24: 'F. Zhong, K. Wu, H. Ci, C. Wang, and H. Chen, "Empowering embodied visual tracking with visual foundation models and offline RL," in *Proc. European Conf. Computer Vision (ECCV)*, 2024.',
  unrealcv: 'W. Qiu, F. Zhong, Y. Zhang, S. Qiao, Z. Xiao, T. S. Kim, Y. Wang, and A. Yuille, "UnrealCV: Virtual worlds for computer vision," in *Proc. ACM Int. Conf. Multimedia*, 2017, pp. 1221–1224.',
  dinov3: 'O. Siméoni *et al.*, "DINOv3," arXiv:2508.10104, 2025.',
  yoloe: 'A. Wang, L. Liu, H. Chen, Z. Lin, J. Han, and G. Ding, "YOLOE: Real-time seeing anything," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2025.',
  ortrack: 'Y. Wu, X. Wang, X. Yang, M. Liu, D. Zeng, H. Ye, and S. Li, "Learning occlusion-robust vision transformers for real-time UAV tracking," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2025, pp. 17103–17113.',
  diffpol: 'C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, "Diffusion policy: Visuomotor policy learning via action diffusion," in *Proc. Robotics: Science and Systems (RSS)*, 2023.',
  kalman: 'R. E. Kalman, "A new approach to linear filtering and prediction problems," *J. Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.',
  barshalom: 'Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*. New York: Wiley, 2001.',
  singer: 'R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering targets," *IEEE Trans. Aerospace and Electronic Systems*, vol. AES-6, no. 4, pp. 473–483, 1970.',
  li03: 'X. R. Li and V. P. Jilkov, "Survey of maneuvering target tracking. Part I: Dynamic models," *IEEE Trans. Aerospace and Electronic Systems*, vol. 39, no. 4, pp. 1333–1364, 2003.',
  imm: 'H. A. P. Blom and Y. Bar-Shalom, "The interacting multiple model algorithm for systems with Markovian switching coefficients," *IEEE Trans. Automatic Control*, vol. 33, no. 8, pp. 780–783, 1988.',
  sort: 'A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, "Simple online and realtime tracking," in *Proc. IEEE Int. Conf. Image Processing (ICIP)*, 2016, pp. 3464–3468.',
  deepsort: 'N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," in *Proc. IEEE Int. Conf. Image Processing (ICIP)*, 2017, pp. 3645–3649.',
  bytetrack: 'Y. Zhang *et al.*, "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. European Conf. Computer Vision (ECCV)*, 2022.',
  ocsort: 'J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, "Observation-centric SORT: Rethinking SORT for robust multi-object tracking," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 9686–9696.',
  rudenko: 'A. Rudenko, L. Palmieri, M. Herman, K. M. Kitani, D. M. Gavrila, and K. O. Arras, "Human motion trajectory prediction: A survey," *Int. J. Robotics Research*, vol. 39, no. 8, pp. 895–935, 2020.',
  sociallstm: 'A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese, "Social LSTM: Human trajectory prediction in crowded spaces," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 961–971.',
  trajectron: 'T. Salzmann, B. Ivanovic, P. Chakravarty, and M. Pavone, "Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data," in *Proc. European Conf. Computer Vision (ECCV)*, 2020.',
  kitani: 'K. M. Kitani, B. D. Ziebart, J. A. Bagnell, and M. Hebert, "Activity forecasting," in *Proc. European Conf. Computer Vision (ECCV)*, 2012.',
  simon: 'D. Simon, "Kalman filtering with state constraints: A survey of linear and nonlinear algorithms," *IET Control Theory & Applications*, vol. 4, no. 8, pp. 1303–1318, 2010.',
  chung: 'T. H. Chung, G. A. Hollinger, and V. Isler, "Search and pursuit-evasion in mobile robotics: A survey," *Autonomous Robots*, vol. 31, no. 4, pp. 299–316, 2011.',
  nageli: 'T. Nägeli, J. Alonso-Mora, A. Domahidi, D. Rus, and O. Hilliges, "Real-time motion planning for aerial videography with dynamic obstacle avoidance and viewpoint optimization," *IEEE Robotics and Automation Letters*, vol. 2, no. 3, pp. 1696–1703, 2017.',
  penin: 'B. Penin, P. Robuffo Giordano, and F. Chaumette, "Vision-based reactive planning for aggressive target tracking while avoiding collisions and occlusions," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3725–3732, 2018.',
  bonatti: 'R. Bonatti, W. Wang, C. Ho, A. Ahuja, M. Gschwindt, E. Camci, E. Kayacan, S. Choudhury, and S. Scherer, "Autonomous aerial cinematography in unstructured environments with learned artistic decision-making," *J. Field Robotics*, vol. 37, no. 4, pp. 606–641, 2020.',
  fasttracker: 'Z. Han, R. Zhang, N. Pan, C. Xu, and F. Gao, "Fast-Tracker: A robust aerial system for tracking agile target in cluttered environments," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2021, pp. 328–334.',
  elastic: 'J. Ji, N. Pan, C. Xu, and F. Gao, "Elastic Tracker: A spatio-temporal trajectory planner for flexible aerial tracking," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2022.',
  voxblox: 'H. Oleynikova, Z. Taylor, M. Fehr, R. Siegwart, and J. Nieto, "Voxblox: Incremental 3D Euclidean signed distance fields for on-board MAV planning," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2017.',
  cbf: 'A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, "Control barrier functions: Theory and applications," in *Proc. European Control Conf. (ECC)*, 2019, pp. 3420–3431.',
};

module.exports = function build(h) {
  // ---- citation resolver ----
  const order = [];
  const num = (k) => { if (!REFDB[k]) throw new Error('unknown ref ' + k); let i = order.indexOf(k); if (i < 0) { order.push(k); i = order.length - 1; } return i + 1; };
  const R = (t) => t.replace(/\[@([^\]]+)\]/g, (_, keys) => {
    const ns = keys.split(',').map((s) => num(s.trim().replace(/^@/, ''))).sort((a, b) => a - b);
    const out = []; let i = 0;
    while (i < ns.length) { let j = i; while (j + 1 < ns.length && ns[j + 1] === ns[j] + 1) j++;
      out.push(j - i >= 2 ? `[${ns[i]}]–[${ns[j]}]` : ns.slice(i, j + 1).map((n) => `[${n}]`).join(', ')); i = j + 1; }
    return out.join(', ');
  });
  const P = (t, o) => h.P(R(t), o), B = (t) => h.B(R(t)), N = (t, r) => h.N(R(t), r);
  const { H1, H2, EQ, FIGURE, TABLE, TOC } = h;
  const TBL = (w, rows, cap) => TABLE(w, rows.map((r) => r.map((c) => R(String(c)))), cap);
  const C = [];

  // ================= Abstract =================
  C.push(H1(`Abstract`));
  C.push(P(`Visual active tracking (VAT) systems steer a camera to keep a person in view. When the person turns a corner and disappears behind a wall, the tracker has to judge where the person will reappear. Current systems, including the recent OA-VAT framework, use a constant-velocity (CV) Kalman filter for this and are evaluated on targets that walk at near-constant speed. This project studies targets that speed up or slow down while hidden at a corner.`));
  C.push(P(`On a calibrated L-corner benchmark in Unreal Engine, upgrading the motion model from CV to constant acceleration (CA) gave only a small gain (about 3%), and only when the measurement noise was calibrated. Most of the error came from how the filter is used at the corner. Partially occluded measurements corrupt its last estimate, and its blind-phase prediction drifts into walls. A reactive controller that steers toward this prediction then carries the error into the camera motion. The main contribution of this project is a new architecture built around the scene map. It combines a **map-constrained Kalman filter (MC-KF)**, which repairs partially occluded measurements and projects the blind-phase prediction into reachable corridor space, with **belief-independent blind-phase control**, in which the camera stops steering from the filter’s estimate (its *belief*) while the target is hidden. Over 20 paired seeds, the re-acquisition error falls from 165 px to 12 px (about 8 cm), and the new architecture is better on all 20 seeds. Success rate stays at 1.00 and there are no wall collisions. The baseline is the CV Kalman filter released with OA-VAT, run with the same ideal (oracle) detector and controller, so the comparison measures occlusion handling only. Inside the new architecture a CA filter is significantly *worse* than CV, because its acceleration term and the map projection correct the same lag twice.`));

  C.push(TOC());

  // ================= 1 =================
  C.push(H1(`1  Background and Motivation`));
  C.push(P(`Small drones are increasingly used to follow people autonomously, for filming, inspection and search-and-rescue. This task is called *visual active tracking*. A passive tracker only draws a box around a target in a fixed video; an active tracker must also move the camera to keep the target in view at a suitable distance [@luo18,@luo20]. Indoors and in cities, a common cause of failure is occlusion by structure. The target walks around a corner, a wall blocks the view, and the tracker has to decide where to look and where to fly without any measurement.`));
  C.push(P(`Most VAT systems bridge occlusion with a short-term motion model, usually a CV Kalman filter that extrapolates the last bounding box until the target is detected again. The recent OA-VAT framework [@oavat] follows this pattern, and its benchmark targets walk at almost constant speed, which matches the filter’s own assumption. Real people who go out of sight, however, often change their motion: they speed up, stop, or turn. Breaking line of sight before manoeuvring is the standard way to escape a follower [@chung]. This project was motivated by two observations made while reproducing OA-VAT (Section 6.1). In a small pilot, gentle target acceleration alone reduced its tracking reward by 14%, and at corners its predictor performed no better than using no predictor at all. These observations raised the question studied here: what actually goes wrong when a target manoeuvres behind a corner, and how should a tracker handle it?`));
  C.push(P(`The obvious fix is a richer motion model, such as constant acceleration (CA) or an interacting multiple model. This project tested that first and found that it helps only a little. The larger errors come from *how the filter is used* at a corner: what it is fed while the target is partly hidden, and where its prediction is allowed to go. When the camera is also steered toward that prediction, the error goes straight into the flight path. A corner is a known piece of geometry, and the tracker can use it. This motivates the approach of the project: a map-constrained estimation and control architecture that works with any motion model.`));

  // ================= 2 =================
  C.push(H1(`2  Literature Survey`));
  C.push(H2(`2.1  Visual active tracking`));
  C.push(P(`Luo et al. [@luo18,@luo20] formulated active tracking as end-to-end reinforcement learning, with a policy that maps frames directly to camera actions, trained in virtual worlds built with UnrealCV [@unrealcv]. Follow-up work improved robustness with an adversarial escaping target (AD-VAT [@advat]) and distractors [@zhong21], reconstructed the surroundings and predicted trajectories for generalisation (RSPT [@rspt]), and used visual foundation models with offline RL [@zhong24]. In these methods occlusion handling is implicit in a recurrent state, so failures cannot easily be attributed to prediction or to control. OA-VAT [@oavat] instead is modular. It detects and re-identifies the target (YOLOE [@yoloe], DINOv3 prototypes [@dinov3], ORTrack [@ortrack]). When the target is lost, it stops the camera, extrapolates the box with a confidence-aware Kalman filter to decide where to search, and falls back to a diffusion-policy planner [@diffpol] if the target stays lost. OA-VAT motivated this project, and the observations that led to the research question were made while reproducing it. This project reuses its Kalman filter, evaluation protocol and simulator (Section 5.1), but not its tracking pipeline.`));
  C.push(H2(`2.2  Motion models and occlusion in tracking`));
  C.push(P(`The Kalman filter [@kalman] is optimal when the motion model matches the target. For targets that do not follow a fixed model, the tracking literature offers the Singer acceleration model [@singer], a family of dynamic models surveyed by Li and Jilkov [@li03], and the Interacting Multiple Model (IMM) estimator [@imm], the standard tool in radar and aerospace tracking [@barshalom]. All of these assume a steady stream of measurements. During a long occlusion there are no updates, and every model reduces to open-loop extrapolation from the estimate at the moment the target disappeared.`));
  C.push(P(`In computer vision, the SORT family [@sort,@deepsort,@bytetrack] runs a CV Kalman filter on the bounding box to bridge missed detections. OC-SORT [@ocsort] shows that a filter coasting through occlusion builds up error, mainly through its noisy velocity estimate, and corrects the filter *after* the target is found again. ORTrack [@ortrack] makes the appearance model robust to partial occlusion. None of these methods uses scene geometry, and none addresses a subtler problem: a correctly detected but *clipped* box has a biased centre, and this bias goes straight into the velocity estimate just before the target disappears.`));
  C.push(H2(`2.3  Scene-aware prediction and constrained estimation`));
  C.push(P(`Human trajectory prediction has moved to learned, context-aware models [@rudenko] such as Social LSTM [@sociallstm] and Trajectron++ [@trajectron]. Kitani et al. [@kitani] showed that knowing where a person *can* walk strongly constrains long-range forecasts. These predictors are mostly evaluated offline with full observation histories, not inside a closed tracking loop. In estimation theory, known physical limits can be enforced with *state-constrained Kalman filtering* [@simon]. The simplest method, estimate projection, projects the estimate onto the feasible set. The MC-KF in this project applies this idea with the corridor as the feasible set, through the camera’s projection model.`));
  C.push(H2(`2.4  Aerial target following among obstacles`));
  C.push(P(`Model-based aerial trackers plan drone trajectories with explicit collision and visibility terms [@nageli,@penin,@bonatti]. Fast-Tracker [@fasttracker] and Elastic Tracker [@elastic] track agile targets in clutter using a map of the environment [@voxblox] and a kinematic prediction of the target. They are designed to *prevent* occlusion, and they assume a metric map and a 3D target state. Collision safety is often enforced with control barrier functions [@cbf], which change a nominal command as little as possible. The safety layer used here follows the same idea at velocity level, and never changes the heading.`));
  C.push(H2(`2.5  Research gaps`));
  C.push(...TBL([2.2, 1.0, 2.6, 1.2], [
    [`Work`, `Active camera`, `Estimation during occlusion`, `Uses map`],
    [`SORT family, OC-SORT [@sort,@ocsort]`, `No`, `CV-KF coasting; re-update after recovery`, `No`],
    [`Trajectron++ [@trajectron]`, `No`, `Learned multimodal forecast (offline)`, `Semantic`],
    [`AD-VAT, RSPT [@advat,@rspt]`, `Yes`, `Implicit (RNN) / learned trajectory`, `Partly`],
    [`Elastic / Fast-Tracker [@elastic,@fasttracker]`, `Drone`, `Kinematic prediction; avoids occlusion`, `Yes`],
    [`OA-VAT [@oavat]`, `Yes`, `Confidence-aware CV-KF; camera stops while lost`, `Planner only`],
    [`**This project**`, `Yes`, `**Map-constrained KF + belief-independent blind control**`, `Corridor`],
  ], `Positioning of representative work.`));
  C.push(B(`**No manoeuvring-target test.** VAT benchmarks use targets that walk at near-constant speed, so a predictor that fails when the target manoeuvres behind cover is not exposed.`));
  C.push(B(`**No use of the map inside the estimator.** Clipped measurements and blind-phase predictions are not checked against the scene geometry the tracker already knows.`));
  C.push(B(`**Estimation and control during occlusion.** A tracker that steers toward its prediction turns every estimation error into a control error, while one that stops, as OA-VAT does, falls behind the target. How the camera should move while the target is hidden has received little attention.`));

  // ================= 3 =================
  C.push(H1(`3  Problem Statement`));
  C.push(P(`**How can a drone-mounted active tracker use the scene map to predict, and steer toward, the reappearance of a person who changes speed while hidden behind a corner, without compromising collision safety?**`));
  C.push(P(`The design target is a re-acquisition error below about 30 px (about 20 cm at 250 cm range), with success rate 1.00 and zero wall contacts, for both sprinting and randomly accelerating/braking targets.`));
  C.push(H2(`3.1  Contributions to date`));
  C.push(N(`**Map-constrained Kalman filter (MC-KF).** A standard box-space Kalman filter with two map-based steps: repairing partially occluded measurements, and projecting the blind-phase prediction into reachable corridor space. It works with either a CV or a CA motion model.`, `contrib`));
  C.push(N(`**Belief-independent blind-phase control.** While the target is hidden, the camera aims at the occluding edge at a fixed speed instead of chasing the prediction. A velocity-level safety layer keeps it clear of walls.`, `contrib`));
  C.push(N(`**Decoupling finding.** When the camera kept steering toward its prediction, the more accurate estimate produced by measurement repair made the tracker lose the target in 7 of 20 sprint episodes. With belief-independent blind-phase control these failures disappear. During occlusion, estimation and control should therefore be designed separately.`, `contrib`));
  C.push(N(`**Benchmark evidence.** A calibrated corner benchmark, a re-acquisition metric and a full set of ablations. They show that CV → CA gives at most a small, noise-dependent gain on the baseline, and that inside the new architecture the CA acceleration term conflicts with the map projection.`, `contrib`));

  // ================= 4 =================
  C.push(H1(`4  Project Objectives`));
  C.push(P(`✔ = complete, ○ = remaining.`));
  C.push(N(`✔ Reproduce OA-VAT [@oavat] in simulation as the starting point for understanding current methods.`));
  C.push(N(`✔ Build a calibrated L-corner benchmark with engine-verified occlusion timing and a 20-seed paired closed-loop harness.`));
  C.push(N(`✔ Test the motion-model hypothesis (none / CV / CA, standard and calibrated noise).`));
  C.push(N(`✔ Design and ablate the MC-KF and belief-independent blind-phase control.`));
  C.push(N(`○ Generalise the map: compute corridor and occluding edges from wall geometry; test on T-junction and S-bend tracks (20 seeds each, target below 30 px).`));
  C.push(N(`○ Replace the oracle detector with the OA-VAT perception stack; re-calibrate noise and test innovation gating.`));
  C.push(N(`○ Study braking targets (short windows, overshoot); add an IMM baseline and three seed sets with confidence intervals.`));
  C.push(N(`○ Stretch: multi-hypothesis prediction at the T-junction; indoor test on a Tello drone.`));

  // ================= 5 =================
  C.push(H1(`5  Methodology`));
  C.push(H2(`5.1  Simulation setup`));
  C.push(P(`The benchmark runs in the FlexibleRoom scene of gym-unrealcv [@unrealcv]. Five scripted blocks form an L-shaped corridor, and each wall position is checked against the engine. No planner is used: the camera is steered by a simple reactive controller, so any difference in performance comes from the estimation and blind-phase logic (Figure 1). Table 2 lists the main settings.`));
  C.push(...TBL([2.2, 2.6, 3.2], [
    [`Item`, `Value`, `Reason`],
    [`Corridor`, `W = 150 cm, walls 350 cm`, `Narrow indoor corridor`],
    [`Camera`, `640×480, hFOV 82.6°, pitch −15°`, `Tello spec; target centred at 250 cm`],
    [`Camera limits`, `v ≤ 99 cm/s, |ω| ≤ 30°/s, 20 Hz`, `Speed and rate from OA-VAT; turn-rate limit set here`],
    [`Target`, `Person, 168 cm, speed 40–99 cm/s`, `Never faster than the camera`],
    [`Following distance`, `250 cm`, `Optimum of the OA-VAT score`],
    [`Wall margin`, `camera ≥ 30 cm from walls`, `Near-clip limit 15 cm (measured) plus body radius and localisation slack`],
  ], `Frozen benchmark configuration.`));
  C.push(P(`**What is taken from OA-VAT.** Only the elements in Table 3 are inherited. Everything else is designed in this project: the track, camera setup, detector, controller, target behaviours, re-acquisition metric and the proposed architecture.`));
  C.push(...TBL([2.0, 3.6, 2.4], [
    [`Element`, `Inherited content`, `Difference in this project`],
    [`Kalman filter`, `8-D CV filter from the OA-VAT code (structure from DeepSORT [@deepsort]); noise weights 1/20 (position), 1/40 (velocity)`, `Used unchanged as the baseline; the CA filter extends it`],
    [`Evaluation protocol`, `Per-step score r = 1 − |ρ − 250|/750 − |θ|/45; visible if ρ ≤ 750 cm and |θ| ≤ 45°; 500 steps; AR / EL / SR`, `Failure after 50 invisible steps (OA-VAT: after more than 50); r clipped to [−1, 1] before summing`],
    [`Control timing`, `50 ms period (20 Hz), speed limit 99 cm/s, following distance 250 cm`, `None`],
    [`Simulator`, `gym-unrealcv FlexibleRoom scene and person model`, `Windows fixes; walls built from scripted blocks`],
  ], `Elements inherited from OA-VAT.`));
  C.push(P(`**Scope of the comparison.** This test bench deliberately simplifies perception and control, so that occlusion handling can be measured on its own. Detection uses the engine’s object mask instead of OA-VAT’s YOLOE, DINOv3 and ORTrack stack. Steps run in fixed 50 ms increments with no latency, and no planner is used. OA-VAT’s own occlusion procedure is therefore not reproduced here. That procedure stops the camera, searches around the prediction with its detector for at most 15 frames, and then falls back to the planner. All results in this report compare configurations under the same detector and controller.`));
  C.push(...FIGURE(`fig_arch.png`, 14, `Closed-loop architecture. Purple: MC-KF steps; red: belief-independent blind-phase control.`));

  C.push(H2(`5.2  Occlusion geometry`));
  C.push(P(`For a camera following at distance d, and a target passing the inner corner at clearance c, the wall blocks the line of sight only if`));
  C.push(EQ(`d > 4c`, `d > 4c`));
  C.push(P(`and the target stays hidden over a path length`));
  C.push(EQ(`L~occ~ = √( d (d − 4c) )`, `L_{\\mathrm{occ}} = \\sqrt{d\\,(d-4c)}`));
  C.push(P(`Both results match the engine’s ground-truth mask: with d = 250 cm and c = 40 cm the target is hidden for 150 cm, about 38 steps. Each passage has a *fade-out* phase, in which the wall gradually clips the box, followed by a *fully blind* phase (Figure 2). In the experiments c is drawn from [40, 60] cm, so the line of sight is always blocked when the target keeps moving. A target that brakes at the corner may never become fully hidden.`));
  C.push(...FIGURE(`fig_occlusion.png`, 14.5, `Engine-measured target pixels during one corner passage, showing fade-out, fully blind phase and reappearance.`));

  C.push(H2(`5.3  Target behaviour`));
  C.push(P(`Each seed generates a random route through the corner. Two profiles are tested: **P2 (sprint)**, where the target accelerates at 80 cm/s² to 99 cm/s from the corner onward, and **P2a**, where the corner acceleration is drawn from [−80, 80] cm/s², so the target may sprint or brake behind the wall.`));

  C.push(H2(`5.4  Baseline filter and controller`));
  C.push(P(`The filter state is the box (cx, cy, a, h) and its velocities, with a one-frame step and noise scaled by box height h. The **baseline** is the filter released with OA-VAT (Table 3), used with its original noise settings: it predicts every frame, updates whenever the target is detected, and coasts until the target reappears.`));
  C.push(EQ(`x~k+1~ = F x~k~ + w~k~,   z~k~ = H x~k~ + v~k~,   F = [ I  I ; 0  I ],   H = [ I  0 ]`,
    `\\mathbf{x}_{k+1} = F\\mathbf{x}_k + \\mathbf{w}_k,\\quad \\mathbf{z}_k = H\\mathbf{x}_k + \\mathbf{v}_k,\\quad F=\\begin{bmatrix} I & I\\\\ 0 & I\\end{bmatrix},\\quad H=\\begin{bmatrix} I & 0\\end{bmatrix}`));
  C.push(P(`The CA variant adds accelerations for cx and cy, bounded by the pedestrian limit of 80 cm/s². When the target is visible, the camera turns toward the estimated bearing and keeps the estimated range at 250 cm:`));
  C.push(EQ(`θ = atan((cx − 320)/f~x~),   ρ̂ = f~x~ · 168 / h`,
    `\\theta = \\arctan\\frac{c_x-320}{f_x},\\qquad \\hat{\\rho} = \\frac{f_x\\cdot 168}{h}`));
  C.push(EQ(`ω = clip(3θ, ±30°/s),   v = clip(2(ρ̂ − 250), 0, 99 cm/s)`,
    `\\omega = \\operatorname{clip}(3\\theta,\\ \\pm 30^\\circ/\\mathrm{s}),\\qquad v = \\operatorname{clip}\\big(2(\\hat{\\rho}-250),\\ 0,\\ 99\\ \\mathrm{cm/s}\\big)`));
  C.push(P(`In the baseline the same controller is also driven by the filter’s prediction while the target is hidden, so the camera keeps steering toward the extrapolated box.`));

  C.push(H2(`5.5  Map-constrained Kalman filter (MC-KF)`));
  C.push(P(`The MC-KF keeps the standard predict and update equations and adds two steps that use the corridor map.`));
  C.push(P(`**Measurement repair (fade-out frames).** When the target’s pixel count drops below 50% of its peak over the last 8 healthy frames, the box is being clipped by the wall and its centre is biased. The corner apex is projected into the image to decide which side is clipped. The centre is rebuilt from the clean edge using the box width remembered from healthy frames: cx = u~clean~ ± w~good~/2. Only this repaired cx is used in the update.`));
  C.push(P(`**Reachable-set projection (blind frames).** After each prediction, the predicted (cx, h) is back-projected to a floor point using the camera pose ψ:`));
  C.push(EQ(`p = p~cam~ + ρ̂ [cos(ψ + θ), sin(ψ + θ)]`, `\\mathbf{p} = \\mathbf{p}_{\\mathrm{cam}} + \\hat{\\rho}\\,\\big[\\cos(\\psi+\\theta),\\ \\sin(\\psi+\\theta)\\big]^{\\top}`));
  C.push(P(`If p lies outside the corridor free space, it is moved to the nearest reachable point and projected back into (cx, h). The covariance is not changed, so the filter stays cautious and trusts the first new measurement. This is estimate projection in the sense of [@simon]. Two earlier versions, pushing the point out of the nearest wall and ray-marching through walls, failed and were discarded.`));

  C.push(H2(`5.6  Belief-independent blind-phase control and safety`));
  C.push(P(`Here *belief* means the filter’s current estimate of where the target is. While the target is fading or hidden, yaw aims at the image column of the occluding corner edge, where the target must reappear, and speed is fixed at 99 cm/s. The belief is no longer used for steering. It keeps running in the background, is scored when the target reappears, and then continues with the new measurement. A safety layer limits only the wall-approach component of the velocity, based on the distance d to the nearest wall:`));
  C.push(EQ(`v~in~ ≤ max(0, d − 30 cm) / 0.6 s`, `v_{\\mathrm{in}} \\le \\frac{\\max(0,\\ d-30\\,\\mathrm{cm})}{0.6\\,\\mathrm{s}}`));
  C.push(P(`The safety layer never changes heading. A hard veto with wall sliding backs it up.`));

  C.push(H2(`5.7  Evaluation`));
  C.push(P(`On a single corner the OA-VAT metrics (accumulated reward AR, episode length EL, success rate SR) change very little between methods, because every method eventually finds the target again. The main metric is therefore the **re-acquisition error**. When the target is detected again after at least 5 blind frames, it is the horizontal pixel distance between the filter’s predicted box centre and the centre of the newly detected box. At 250 cm, 1 px corresponds to about 0.69 cm; a sprinting target is usually farther away, so the conversion to centimetres is approximate. Every configuration runs on the same 20 seeds (2000–2019) with identical target routes. Results are reported as paired differences, per-seed win counts and bootstrap 95% confidence intervals. Repeating a configuration gives small run-to-run differences (for example 20.9 vs 22.0 px on P2a), so only paired differences with confidence intervals are interpreted.`));

  // ================= 6 =================
  C.push(H1(`6  Progress Report and Revised Gantt Chart`));
  C.push(H2(`6.1  Reproduction and motivating observations`));
  C.push(P(`OA-VAT was reproduced on Windows with all three modules, including training the diffusion planner on 1,464 self-collected episodes. This required several fixes to the released code. One of them was a missing dependency that had silently disabled the confidence-aware smoothing. On UrbanCity the reproduction reached SR 0.95 and EL 487 (paper: 0.95 / 486), and on ParkingLot the planner raised SR from 0.70 to 0.90 (paper: 0.93). AR was 80–115 points lower than reported (270 vs 385 on UrbanCity, 314 vs 392 on ParkingLot), and the matching SR values were obtained before the missing dependency was found. The reproduction is therefore close but not exact. It served to understand the method rather than to benchmark it.`));
  C.push(P(`Two observations from this stage motivated the project. On a four-corner course, target acceleration capped at the camera’s top speed reduced AR from 291 to 249 (10 episodes, one seed). In a ParkingLot variant where the target manoeuvres at corners, the confidence-aware predictor reached SR 0.85 against 0.90 without a predictor (20 episodes). Both samples are small, so these results are treated as motivation, not as evidence.`));
  C.push(H2(`6.2  The motion-model hypothesis`));
  C.push(P(`The original plan was to replace CV with a stronger motion model. This was tested on the calibrated L-corner with 20 paired seeds per configuration. With the inherited noise settings, CA was 6.3 px *worse* than CV on the sprint profile (95% CI [−1.1, 13.0] px; CA better on 6/20 seeds). With the measurement noise reduced to match the near-noiseless detector, CA beat CV by 4.9 px (136 vs 141 px; CI [−8.2, −2.1]; 17/20 seeds). This gain is statistically significant but small, about 3%, and even the best result was still about 0.9 m. The estimate at the moment the target disappeared was already wrong, and a better motion model cannot correct that.`));
  C.push(H2(`6.3  From the decoupling finding to the new architecture`));
  C.push(P(`The next step was to clean the measurements. In early single-seed tests, a more accurate repaired estimate made the camera steer deep behind the wall and miss the target reappearing at the corner edge. A 20-seed control confirms this failure mode: with belief-driven steering and no map projection, adding measurement repair lowered the error on the surviving episodes but made the tracker lose the target in 7 of 20 sprint episodes (SR 1.00 → 0.65, AR 450 → 387). Once blind-phase steering and speed were made independent of the belief, the same repair became the largest single gain, and adding the reachable-set projection completed the MC-KF. Figure 3 shows the sequence.`));
  C.push(...FIGURE(`fig_progress.png`, 14.5, `Re-acquisition error on the sprint profile at each development stage (20 paired seeds per bar). Stages 2–5 use the calibrated measurement noise.`));
  C.push(P(`Measured with CV on the sprint profile, the reduction from 165 px to 14 px breaks down as follows: noise calibration 24 px, velocity-cap safety layer 6 px, measurement repair with edge aiming 97 px, and fixed blind speed with the reachable-set projection 23 px. With the original noise settings the full architecture reaches 12 px, so it does not depend on the noise calibration.`));
  C.push(H2(`6.4  Current results`));
  C.push(...TBL([2.6, 1.3, 1.3, 1.4, 0.8], [
    [`Configuration`, `Re-acq. P2 (px)`, `Re-acq. P2a (px)`, `AR (P2)`, `SR`],
    [`Baseline: OA-VAT CV-KF (no map, no safety layer)`, `164.9`, `137.1`, `449.1`, `1.00`],
    [`New architecture, no predictor`, `131.8`, `125.6`, `456.0`, `1.00`],
    [`**New architecture, MC-KF (CV)**`, `**12.3**`, `**20.9**`, `455.9`, `1.00`],
    [`New architecture, MC-KF (CA)`, `21.4`, `34.8`, `456.0`, `1.00`],
    [`MC-KF (CV) − baseline, paired (95% CI)`, `−152.6 [−173.6, −131.3]`, `−116.2 [−152.4, −79.9]`, ``, ``],
    [`Seeds where MC-KF (CV) is better`, `20 / 20`, `11 / 12`, ``, ``],
  ], `Current results (30 cm wall margin). P2 uses all 20 seeds; P2a uses the 12 seeds in which the target became fully hidden for at least 5 frames (braking targets often did not). Wall contacts: 60 per episode for the baseline, 0 for the new architecture.`));
  C.push(P(`The new architecture reduces the error about thirteenfold, from 165 px to 12 px (about 8 cm), and is better than the baseline on all 20 sprint seeds. The Kalman filter remains essential: with the same architecture but no predictor, the error is 132 px. AR and SR barely change, because on one corner every method eventually finds the target. On normal targets without a corner manoeuvre (random speed changes), the architecture also lowers the error by 80 px (CI [−110, −52]; 17/18 seeds) and removes all wall contacts, so it does not hurt ordinary tracking.`));
  C.push(P(`**Ablations.** Table 5 removes one component at a time from the full architecture. Measurement repair is by far the most important part: without it the error rises to about 265 px, worse than using no predictor, because the clipped box is fed straight into the filter. Aiming at the occluding edge and flying at fixed speed are both needed. Edge aiming lowers AR slightly, since the score rewards keeping the target centred, but improves re-acquisition by 63 px. The velocity cap does not change accuracy; its role is safety.`));
  C.push(...TBL([3.0, 1.8, 1.8, 1.4], [
    [`Removed from full architecture (CV)`, `Δ re-acq. P2 (px, 95% CI)`, `Δ re-acq. P2a (px)`, `Other effect`],
    [`Measurement repair`, `+254 [+233, +275]`, `+284`, `—`],
    [`Edge aiming`, `+63 [+40, +88]`, `+67`, `AR +12`],
    [`Fixed blind speed`, `+24 [+13, +34]`, `+21`, `1 episode lost (SR 0.95)`],
    [`Reachable-set projection`, `+24 [+16, +32]`, `+9 (n.s.)`, `—`],
    [`Velocity-cap safety layer`, `+1 (n.s.)`, `−5`, `38–89 wall contacts per episode`],
  ], `Leave-one-out ablation, 20 paired seeds (n.s. = confidence interval includes zero).`));
  C.push(P(`**Why CA is worse inside the MC-KF.** With CA, the error is 9 px larger on P2 (CI [1.3, 15.9]; CA better on 5/20 seeds) and 8 px larger on P2a over 40 seeds (CI [2.1, 13.5]). The filter’s internal state shows why. During the blind phase a CV filter lags behind the target, and the reachable-set projection removes this lag, leaving no systematic error. The CA filter enters the blind phase with an acceleration learned during fade-out, which also compensates the lag. Combined, the two corrections overshoot by about 20 px in the direction of motion, and the acceleration term explains the CA–CV difference seed by seed (correlation 0.99). Zeroing the acceleration during the blind phase makes CA equal to CV (+0.6 px, CI [−1.4, 2.5]). Removing the projection instead makes CA as good as CV with the projection (+0.9 px on P2; −8.7 px, CI [−18.1, 0.5] on P2a). The lag can therefore be corrected either by the map or by the acceleration term, but not by both. **The improvement comes from the architecture, not from the motion model.**`));
  C.push(P(`**Limitations.** The results use an oracle detector and a single corner with a hand-coded map. They show the benefit of the architecture over the OA-VAT filter under identical conditions, not a head-to-head comparison with the full OA-VAT system, which would require both pipelines to run on the same real detector. P2a has only 12 usable seeds out of 20, because braking targets often never become fully hidden; braking also shortens the occlusion window and causes overshoot, which no predictor can correct.`));
  C.push(H2(`6.5  Revised Gantt chart`));
  C.push(P(`The original plan had five stages: a manoeuvring target, evidence that prediction degrades, CA/IMM baselines, a learned predictor, and a final comparison. The first three stages are done except for IMM. The results showed that the gains come from the map and from separating estimation and control, not from the motion model. The MC-KF architecture therefore became the core contribution, and the learned predictor became a stretch goal (Figure 4).`));
  C.push(...FIGURE(`fig_gantt.png`, 15.5, `Revised Gantt chart (green: complete; orange: in progress; blue: planned; hatched: stretch).`));

  // ================= 7 =================
  C.push(H1(`7  Planned Work`));
  C.push(P(`The final report will add the following sections:`));
  C.push(H2(`7.1  General map representation and multi-corner tracks`));
  C.push(H2(`7.2  Real detector in the loop and innovation gating`));
  C.push(H2(`7.3  Braking targets and blind-phase speed`));
  C.push(H2(`7.4  Statistical validation and IMM baseline`));
  C.push(H2(`7.5  Discussion and conclusion`));

  return { C, refs: order.map((k) => REFDB[k]) };
};
