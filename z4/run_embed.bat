@echo off
REM z4-embed-server scheduled-task action.
REM Runs the embed lifecycle guard in the foreground (so the task stays "running"
REM while the guard loops). The guard manages llama-server: start / cede-by-kill /
REM restart, per E:\z4-coord\gpu-preempt.flag + Revit GPU pressure. Args pass through
REM (e.g. -CarveOut 0 for idle-window-only, -DryRun to validate the cede).
REM ALWAYS-ON with CAD-presence cede (CarveOut 1, Fredrik 2026-06-12 evening). Both auto-idle
REM signals are broken on this box: per-process GPU memory reads [N/A], and quser console-idle
REM shows "active" for hours after Mats leaves. So idle-only would never run. Instead: run
REM continuously, and CEDE (kill the server, ~1.5s) when Revit/AutoCAD is process-present (Mats
REM doing local CAD) OR the gpu-preempt.flag is raised. Parsec-ACTIVE (rare; he flags it) is
REM monitored by the 45-min health loop + operator, not auto-detected (parsecd is GPU-present
REM even when idle). A GPU at 100% doesn't lock up the desktop; only CAD/Parsec would feel it.
powershell -NoProfile -ExecutionPolicy Bypass -File E:\llama-embed\embed_guard_local.ps1 -CarveOut 1 -PollSec 1.5 %*
