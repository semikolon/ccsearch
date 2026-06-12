@echo off
REM z4-embed-server scheduled-task action.
REM Runs the embed lifecycle guard in the foreground (so the task stays "running"
REM while the guard loops). The guard manages llama-server: start / cede-by-kill /
REM restart, per E:\z4-coord\gpu-preempt.flag + Revit GPU pressure. Args pass through
REM (e.g. -CarveOut 0 for idle-window-only, -DryRun to validate the cede).
REM IDLE-ONLY (CarveOut 0). The carve-out (run during Mats-active GPU-idle) was tried 2026-06-12
REM but proved UNSAFE on this machine: Mats works via Parsec (remote desktop, uses the GPU's
REM NVENC), and per-process GPU memory reads [N/A] here, so the memory-jump cede is BLIND to his
REM CAD/Parsec GPU use. The only reliable "Mats present" signal is console-idle (quser), which
REM catches both local AND Parsec sessions. So: launch only after console idle >=20 min, cede if
REM he returns (idle <90s). Never runs while he's working (local or remote). Backlog clears in
REM away-windows (overnight + breaks); the bulk-write throughput fix keeps that fast.
powershell -NoProfile -ExecutionPolicy Bypass -File E:\llama-embed\embed_guard_local.ps1 -CarveOut 0 %*
