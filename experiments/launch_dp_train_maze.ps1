# Train the OA-VAT diffusion planner on original + maze A*-expert data, with the windowing fix
# (pad_after=0: one window per sample, starting at waypoint 0).  3564 samples -> 112 it/epoch;
# 120 epochs ~= 13.4k gradient steps (about the original run).  Checkpoints every 20 epochs.
$py   = "C:\Users\AUFB\miniconda3\envs\follow\python.exe"
$repo = "D:\Aero2711\OA-VAT\gym-unrealcv\diffusion_policy-main"
$out  = "D:\Aero2711\OA-VAT\data\dp_run_maze"
New-Item -ItemType Directory -Force $out | Out-Null
$args = @(
  "train.py", "--config-dir=diffusion_policy/config", "--config-name=train_unreal_bbox_image_workspace",
  "task.dataset.zarr_path=D:\Aero2711\OA-VAT\data\unreal_replay_maze.zarr",
  "task.dataset.pad_after=0",
  "training.num_epochs=120", "training.checkpoint_every=20",
  "training.rollout_every=100000", "training.sample_every=100000",
  "dataloader.num_workers=0", "val_dataloader.num_workers=0",
  "dataloader.batch_size=32", "val_dataloader.batch_size=16",
  "logging.mode=offline",
  "multi_run.run_dir=$out", "hydra.run.dir=$out"
)
Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log"
