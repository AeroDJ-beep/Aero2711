# Retrain the OA-VAT diffusion planner with the windowing fixed.
# Data: one observation per sample, its 8 waypoints stored as 8 rows with the image duplicated.
# The original run used horizon=8 + pad_after=7, which turns every sample into 8 windows that
# start at waypoint 0..7 for the SAME input -> the model learns a mixture of shifted tails.
# pad_after=0 keeps exactly one window per sample, starting at waypoint 0 (= the expert path).
# Same hyper-parameters as data/dp_run otherwise; 300 epochs ~= the original number of gradient steps.
$py   = "C:\Users\AUFB\miniconda3\envs\follow\python.exe"
$repo = "D:\Aero2711\OA-VAT\gym-unrealcv\diffusion_policy-main"
$out  = "D:\Aero2711\OA-VAT\data\dp_run_pad0"
New-Item -ItemType Directory -Force $out | Out-Null
$args = @(
  "train.py", "--config-dir=diffusion_policy/config", "--config-name=train_unreal_bbox_image_workspace",
  "task.dataset.zarr_path=D:\Aero2711\OA-VAT\data\unreal_replay.zarr",
  "task.dataset.pad_after=0",
  "training.num_epochs=300", "training.checkpoint_every=25",
  "training.rollout_every=100000", "training.sample_every=100000",
  "dataloader.num_workers=0", "val_dataloader.num_workers=0",
  "dataloader.batch_size=32", "val_dataloader.batch_size=16",
  "logging.mode=offline",
  "multi_run.run_dir=$out", "hydra.run.dir=$out"
)
Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$out\train_stdout.log" -RedirectStandardError "$out\train_stderr.log"
