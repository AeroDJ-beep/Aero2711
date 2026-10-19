# Aero2711 — 拐角遮挡下的视觉主动跟踪（Tello / OA-VAT 仿真）

## 目录布局（2026-09-19 整理）

| 目录 | 内容 |
|---|---|
| `OA-VAT/` | OA-VAT (CVPR 2026) 复现仓库 + gym-unrealcv + FlexibleRoom 引擎二进制；`kalman_filter.py` / `kalman_filter_ca.py` 在此 |
| `env_config/` | **权威环境配置** `ENVIRONMENT_CONFIG.md`（相机/场景/协议/标定，英文）+ `data/`（遮挡扫描 CSV、体态扫描 CSV、`margin_calib/` 贴墙标定截图） |
| `experiments/` | 实验脚本（从临时目录抢救）。主脚本 `run_metrics_test.py`：五模式 × 剖面 × 20 种子闭环评测，所有机制开关见文件头常量（`--dyn --edgemeas --edgeaim --blindspeed --mapclamp` 为当前最强配置） |
| `results/metrics/` | 每轮实验的逐集 JSON（文件名含日期与实验名） |
| `results/figures/` | 管线图（`our_pipeline`、`flow_simple`）、剂量-响应、速度曲线、L 跑道俯视图等 |
| `results/videos/` | 2D 俯视动画（基线 / 冲刺病例 / 刹车病例 / 限速对比 / mapclamp 对比） |
| `results/logs/` | 逐帧状态日志 |
| `docs/` | 参考文献 PDF、KF 一帧推导、PS 源文件 |
| `Tello_*`, `first-project/` | 真机与早期练习工程（未动） |
| `_misc/ps_installer/` | 误解压到根目录的 PS 插件安装包（可删） |

## 运行实验

```
C:\Users\AUFB\miniconda3\envs\follow\python.exe experiments\run_metrics_test.py ca cv none '@p2' '@p2a' --dyn --edgemeas --edgeaim --blindspeed --mapclamp
```
（PowerShell 下 `@p2` 必须加引号；结果写到 `experiments/metrics_smoke.json`，归档时改名复制到 `results/metrics/`。）

## 当前最佳结果（20 种子，全架构）
重现误差 ca 22.1 px（p2 冲刺）/ 27.4 px（p2a 随机机动），none 178 px；AR 452、SR 1.00。
