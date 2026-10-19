# Stop every uz_sweep.py / uz_video_maze.py process and every UnrealZoo engine (used by run_sweep_watchdog.sh).
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uz_sweep.py|uz_video_maze.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
taskkill /F /T /IM UnrealZoo_UE5_7.exe 2>$null | Out-Null       # /T: the launcher spawns child processes
Start-Sleep -Seconds 2
$left = @(Get-Process | Where-Object { $_.ProcessName -match 'UnrealZoo' }).Count
Write-Output "engines left: $left"
