@echo off
REM z4-embed-server scheduled-task action.
REM Runs the embed lifecycle guard in the foreground (so the task stays "running"
REM while the guard loops). The guard manages llama-server: start / cede-by-kill /
REM restart, per E:\z4-coord\gpu-preempt.flag + Revit GPU pressure. Args pass through
REM (e.g. -CarveOut 0 for idle-window-only, -DryRun to validate the cede).
REM Production = idle-only (CarveOut 0): the guard pre-flight refuses to launch unless the
REM console has been idle >=20 min, and cedes if Mats returns (idle <90s). So it NEVER runs
REM while Mats is at his desk. %* still allows an explicit override (e.g. -CarveOut 1).
powershell -NoProfile -ExecutionPolicy Bypass -File E:\llama-embed\embed_guard_local.ps1 -CarveOut 0 %*
