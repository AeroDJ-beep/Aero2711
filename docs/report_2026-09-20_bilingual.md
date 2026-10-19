# 汇报稿 · 中英逐句对照 · 对应最终配置 2026-09-19

最终配置 / Final configuration (2026-09-28):
`run_metrics_test.py ca cv none '@p2' '@p2a' --dyn --edgemeas --edgeaim --blindspeed --mapclamp`
（A = 200 cm/s²，margin = 30 cm；未启用 --gate / gate not enabled）
结果文件 / results: results/metrics/metrics_dyn2_margin30_none_cv_ca_2026-09-28.json

---

## 开场 / Opening

我们的问题是：目标在拐角被墙遮挡时会加速或减速，卡尔曼滤波按旧速度外推，目标重新出现时预测框已经偏得很远。
Our problem: when the target is occluded at a corner it may speed up or slow down, the Kalman filter extrapolates with the old velocity, and by the time the target reappears the predicted box is far off.

我们没有换掉 KF，而是围绕它加了三件事：让预测受地图约束，让无人机在没有规划器的情况下不撞墙，让测量端在半遮挡时给 KF 正确的输入。
We did not replace the KF. We added three things around it: map constraints on the prediction, wall avoidance without a planner, and correct measurements during partial occlusion.

评测设置：20 个随机种子，两种目标行为（拐角冲刺、拐角随机加减速），三种运动模型（none、匀速 CV、匀加速 CA），共 120 局。
Evaluation: 20 random seeds, two target behaviours (corner sprint, corner random accel/decel), three motion models (none, CV, CA), 120 episodes in total.

---

## 一、KF 如何使用地图 / 1. How the KF uses the map

结论：地图不进 KF 方程，只在盲期把预测均值拉回可达空间。协方差不动。
Claim: the map never enters the KF equations; it only pulls the blind-period predicted mean back into reachable space. The covariance is untouched.

KF 的状态里有 cx 和 h。cx 编码目标的方位角，h 编码距离。
The KF state contains cx and h. cx encodes the target's bearing, h encodes its range.

具体关系是 θ = atan((cx − 320)/fx)，ρ = fx·168/h，fx = 364.3。
Concretely, θ = atan((cx − 320)/fx) and ρ = fx·168/h, with fx = 364.3.

每一个没有测量的帧，预测步之后，我们把 (cx, h) 反投影到地面，得到目标的世界坐标。
On every frame without a measurement, after the predict step, we back-project (cx, h) onto the floor to get the target's world position.

如果这个点落在走廊自由空间之外，就投影到走廊的最近点。走廊是两条矩形条带，投影就是把 x、y 各夹进矩形区间。
If that point lies outside the corridor free space, we project it onto the nearest corridor point. The corridor is two rectangular strips, so the projection is a clamp of x and y into each rectangle.

然后把这个点重新投影回图像，覆盖 KF 预测均值里的 cx 和 h。速度状态和协方差 P 都不动。
Then we re-project that point into the image and overwrite cx and h in the KF's predicted mean. Velocity states and the covariance P are untouched.

这个覆盖只发生在没有测量的帧，和更新步互斥。目标重新出现的那一帧，新息 y 变小，但 S、K 不受影响，因为它们只依赖 P。
This overwrite happens only on frames without a measurement, never in the same frame as an update. On re-acquisition the innovation y is smaller, while S and K are unchanged because they depend only on P.

为什么不把地图写进 Q 或 R：墙是硬约束，协方差表达不了"不能到"。
Why not encode the map in Q or R: a wall is a hard constraint, and a covariance cannot express "unreachable".

所以做成预测后的投影，KF 本身不变。P 因此偏保守，重捕时更信测量，这是安全的一侧。
So we implement it as a post-prediction projection and leave the KF itself unchanged. P is therefore conservative, which makes the filter trust the first new measurement more. That is the safe side.

证据：none 到 cv，所有 20 个 seed 的重捕误差都从 100–215 px 降到 55 px 以下，均值从 158 降到 17 px。这一步贡献最大。
Evidence: from none to cv, all 20 seeds drop from 100–215 px to below 55 px; the mean goes from 158 to 17 px. This step contributes the most.

---

## 二、无人机如何不撞墙 / 2. How the UAV avoids walls

结论：不用路径规划器。用解析距离场加三层安全滤波，只削速度，不改航向。
Claim: no path planner. An analytic distance field plus a three-tier safety filter, which only trims velocity and never changes heading.

场景是 5 个矩形墙，到最近墙的距离 d 和外法线方向可以直接算，不需要栅格地图。
The scene is five wall rectangles, so the distance d to the nearest wall and its outward normal are computed analytically, without an occupancy grid.

安全线取 30 cm：机身半径加定位余量。相机就是无人机本体，机身放在相机后方，转动半径为零。
The safety line is 30 cm: body radius plus localisation slack. The camera is the vehicle; the body sits behind it, so the turning radius is zero.

无人机有惯性：速度是状态，每步变化不超过 A·dt，A = 200 cm/s² 是无人机的最大加减速度。
The UAV has inertia: velocity is a state that changes by at most A·dt per step, where A = 200 cm/s² is the UAV's maximum acceleration and deceleration.

第一层是刹车距离律。把速度分解为朝墙分量 u 和沿墙分量。朝墙分量不得超过以 A 刹车能在安全线前停住的速度：u ≤ √(2·A·(d − 30))。
Tier one is a braking-distance law. Velocity is split into a wall-approach component u and an along-wall component. u may not exceed the speed that braking at A can absorb before the margin: u ≤ √(2·A·(d − 30)).

这条约束加在积分后的实际速度上，不是加在控制器的期望值上，所以实际运动一定满足它。它是前向不变的：这一步满足，下一步最大刹车后仍满足。
The bound is enforced on the integrated actual velocity, not on the controller's wish, so the realised motion always satisfies it. It is forward-invariant: if it holds now, it still holds after one step of maximum braking.

第二层是硬否决。一步的落点若进入安全区，撤销那个轴的位移，另一轴保留，所以是沿墙滑而不是停。
Tier two is a hard veto. If a step would land inside the safety zone, that axis's displacement is cancelled and the other axis is kept, so the UAV slides along the wall instead of stopping.

第三层是推离。卡住时沿墙的外法线推一小步，解死锁。
Tier three is a push-off. When stuck, a small step along the wall's outward normal breaks the deadlock.

三层都不碰 yaw，相机始终对着目标，跟踪行为不受安全层影响。
None of the three tiers touches yaw. The camera always faces the target, so tracking is unaffected by the safety layer.

证据：120 局，撞墙次数为 0，实际加速度峰值恰为 200 cm/s²，AR 在三种模式下完全一致（453 / 444）。
Evidence: zero wall hits over 120 episodes, peak realised acceleration exactly 200 cm/s², and AR identical across the three modes (453 / 444).

---

## 三、测量端的阈值如何帮助模型 / 3. How measurement-side thresholds help the model

结论：目标被墙切掉一半时，检测框中心是错的。我们用阈值找出这些帧，用干净的一条边重建中心，不把坏测量喂给 KF。
Claim: when a wall clips half the target, the box center is wrong. We flag those frames with a threshold and rebuild the center from the clean edge, instead of feeding a bad measurement to the KF.

阈值是像素数。当前帧目标像素数低于最近 8 帧健康帧峰值的一半，判定为半遮挡。
The threshold is on pixel count. If the target's pixel count falls below half the peak of the last 8 healthy frames, the frame is flagged as partially occluded.

用地图算出遮挡边（拐角顶点）在图像里的列，就知道被切的是左边还是右边。
From the map we compute the occluder edge's image column, the corner apex, which tells us whether the left or right side is clipped.

健康帧记住了框宽。从没被切的那条边加减半个宽度，得到真正的中心。
Healthy frames remember the box width. Adding or subtracting half that width from the un-clipped edge gives the true center.

重建的 cx 作为测量送进 KF；框高和长宽比用预测值，所以只有 cx 带新息。
The reconstructed cx goes into the KF as the measurement; box height and aspect ratio use the predicted values, so only cx carries innovation.

为什么这一步重要：不做的话，遮挡前最后几帧的假中心会让 KF 学到"目标在减速或往回走"，盲期就往错方向推。
Why it matters: without it, the false centers in the last frames before occlusion make the KF believe the target is slowing or reversing, so the blind-period prediction goes the wrong way.

做了这一步，进入盲期时的速度是对的，地图约束才有意义。
With it, the velocity entering the blind period is correct, and the map constraint has something correct to work with.

与之配套的两条行为规则：半遮挡和盲期时，航向对准遮挡边而不是预测框；速度直接取最大值而不看预测距离。
Two behaviour rules go with it: during partial occlusion and blind frames, yaw aims at the occluder edge rather than the predicted box, and speed is set to maximum instead of being derived from the predicted range.

这样做的原因是：盲期里估计的准确性和转向的有用性是脱钩的，行为通道不应消费一个正在漂移的 belief。
The reason: during occlusion, estimator accuracy and steering utility are decoupled, so the behaviour channels must not consume a drifting belief.

卡方门也已实现：新息的马氏距离 d² = yᵀ S⁻¹ y 超过 9.49 就拒收，连续三次重置。最终配置没有启用，因为仿真检测器是理想的，它是为真实检测器准备的。
A chi-square gate is also implemented: a measurement is rejected when the innovation's Mahalanobis distance d² = yᵀ S⁻¹ y exceeds 9.49, and three rejections trigger a reset. It is not enabled in the final configuration because the simulated detector is ideal; it is there for a real detector.

证据：单 seed 2005，重捕误差从 48.7 降到 3.7 px。每局约 9 帧被标为半遮挡并修正。
Evidence: on seed 2005 the re-acquisition error drops from 48.7 to 3.7 px. About 9 frames per episode are flagged and corrected.

---

## 结尾 / Closing

全套下来：AR 不变，成功率 1.0，撞墙 0 次，重捕误差从 158 降到 16 px（冲刺）、130 降到 16–26 px（随机加减速）。
Overall: AR unchanged, success rate 1.0, zero wall hits, re-acquisition error down from 158 to 16 px (sprint) and from 130 to 16–26 px (random accel/decel).

匀速和匀加速模型在冲刺目标上只差 1 px，在随机加减速目标上 CV 反而好 10 px；逐 seed 看 CA 只赢不到一半。误差的下降来自地图和测量端，不来自运动模型。
CV and CA differ by 1 px on the sprint target, and CV is 10 px better on the random accel/decel target; per seed, CA wins less than half the time. The improvement comes from the map and the measurement side, not from the motion model.

下一步：把走廊约束和遮挡边从硬编码推广到任意地图；接入真实检测器验证卡方门。
Next steps: generalize the corridor constraint and occluder edge from hard-coded to arbitrary maps, and plug in a real detector to validate the chi-square gate.

---

## 要带的图 / Figures to bring

- results/figures/metrics_compare_dyn2_margin30_none_cv_ca_2026-09-28.png — 结果对比 / results
- results/videos/traj_margin_compare_seed2005.mp4 — 安全线 15 vs 30 / margin comparison
- results/figures/flow_simple.png — 流程 / flow
- results/figures/our_pipeline.png — 管线 / pipeline
- results/videos/traj_vcap_compare_seed2005.mp4 — 不撞墙对比 / wall avoidance
- results/videos/traj_mapclamp_compare_seed2005.mp4 — 地图约束对比 / map clamp

---

## 附录 · 2026-09-28 诊断结论 / Addendum: diagnostics of 2026-09-28

像素 reacq 在拐角场景是退化指标：目标总从遮挡边那一列重现，belief 停在遮挡边即得高分。改用世界坐标误差 reacq_cm（幻影反投影点到真实位置的距离）和距离误差 reacq_dr。
Pixel reacq is degenerate at a corner: the target always re-emerges at the occluder-edge column, so a belief frozen at that column scores best. Use the world-space error reacq_cm and the range error reacq_dr instead.

最终配置（--dyn --edgemeas --edgeaim --blindspeed --mapclamp，margin 30）的世界坐标结果：p2 none/cv/ca = 146 / 41 / 37 cm；p2a = 160 / 51 / 49 cm。距离方向误差 11–22% 占大头。
World-space results of the final configuration: p2 none/cv/ca = 146 / 41 / 37 cm; p2a = 160 / 51 / 49 cm. Range error of 11–22% dominates.

遮挡前 KF 落后 30 px 的根因：目标起跑时身体前倾、后腿拖后，检测框的"干净边"是后脚而非躯干；模板匹配可在前 8–11 帧给出无偏测量，但盲期起点速度更准并没有带来更小的世界坐标误差（54 vs 41 cm），因为盲期内目标继续加速转弯，直线外推本身不成立。
Root cause of the 30 px pre-occlusion lag: the sprinting body leans and trails a leg, so the box's "clean" edge is the trailing foot. Template matching gives unbiased measurements for 8–11 frames, but a better velocity at blind onset did not reduce the world-space error (54 vs 41 cm), because the target keeps accelerating and turning while hidden; straight extrapolation is the wrong model.

排除项：相机自转（占像素加速度方差 <20%）；轮廓抖动（健康帧半宽 ±2 px）；fade 帧 R 放大（丢失加速度信息）。
Ruled out: camera yaw (<20% of pixel-acceleration variance), silhouette jitter (±2 px on healthy frames), R inflation on fade frames (discards acceleration information).
