@echo off
REM z4-embed-server scheduled-task action.
REM Runs the embed lifecycle guard in the foreground (so the task stays "running"
REM while the guard loops). The guard manages llama-server: start / cede-by-kill /
REM restart, per E:\z4-coord\gpu-preempt.flag + Revit GPU pressure. Args pass through
REM (e.g. -CarveOut 0 for idle-window-only, -DryRun to validate the cede).
REM CARVE-OUT (CarveOut 1, Fredrik 2026-06-12): runs whenever the GPU has headroom, INCLUDING
REM Mats's active-but-GPU-idle desk time (the GPU is mostly free even when he's working). It
REM cedes within ~1.5s on a Revit/AutoCAD GPU-memory jump (>600MB) OR the gpu-preempt.flag.
REM Tiny model (~4GB) so no VRAM-exhaustion; worst case is a few seconds of viewport stutter
REM on a util-only Revit spike. Revert to idle-only with -CarveOut 0 if it ever disturbs him.
powershell -NoProfile -ExecutionPolicy Bypass -File E:\llama-embed\embed_guard_local.ps1 -CarveOut 1 -PollSec 1.5 %*
