# Z4 embedder — mannaminne backlog on Mats's RTX A4000

Semantic-embedding backend for mannaminne (and the same-space fallback story for Graphiti).
Runs Qwen3-Embedding-4B on the Z4's A4000 instead of Darwin's contended GTX-1650, clearing
the ~830k-chunk backlog overnight. **Status (2026-06-12): LIVE, idle-only.**

## Architecture
- **Model** — `Qwen3-Embedding-4B-Q4_K_M.gguf`, **byte-identical to Darwin's** (sha256
  `2b0cf8…`). Same GGUF + same llama.cpp pooling ⇒ **same vector space**, so chunks already
  embedded by Darwin stay valid and Darwin remains a true semantic fallback. This is why we
  mirror Darwin's Q4 rather than serve FP16 via Infinity: FP16 is a *different* numeric space
  (dimensionality stays 1024 via the client's MRL truncation; precision is what differs) and
  would force a full re-embed + drop Darwin as a fallback. Throughput gain of FP16 is
  non-load-bearing for a one-time backlog.
- **Server** — llama.cpp `llama-server.exe` (b9610 win-cuda-12.4) at `E:\llama-embed\`,
  OpenAI-compat `/v1/embeddings` on `0.0.0.0:8081`, `--parallel 8 --ctx-size 8192
  --batch-size 2048` (A4000 tensor cores; Darwin's single-stream/batch-2 config throttled it
  to ~3/sec — see "throughput" below).
- **Reach** — the Z4's :8081 is NOT WAN-exposed (only ssh is); the Mac client reaches it via
  an SSH tunnel `ssh -L 8081:127.0.0.1:8081 z4` (self-healing supervisor loop on the Mac).
- **Client** — `mannaminne embed` on the Mac (psycopg → Darwin Postgres `:5440`), batch-of-8,
  `MANNAMINNE_EMBED_URL=http://127.0.0.1:8081/v1/embeddings`, run under `caffeinate`, backs
  off 30s when the server is down/ceded. Pipeline proven (commits land).
- **Guard** (`embed_guard_local.ps1`, scheduled task `z4-embed-server`) — the lifecycle
  manager: starts / cedes / restarts llama-server. **Cede = KILL the server** (fully frees
  VRAM; a paused-but-loaded server would still hold ~4 GB and block higher-priority Z4 jobs).

## Mats-safety (the load-bearing constraint)
The Z4 is Mats's daily workstation — and he often works **remotely via Parsec** (which uses the
GPU's NVENC to stream his session). Production runs **CarveOut=0 (idle-only)** (`run_embed.bat`
passes `-CarveOut 0`):
- **Pre-flight** — launches ONLY after the console is idle ≥ 20 min (`quser` session idle, which
  correctly shows Mats present for BOTH local and Parsec sessions; NOT `GetLastInputInfo`).
- **Cede** (~1.5s, kills the server) if console idle < 90s (Mats returned) OR
  `E:\z4-coord\gpu-preempt.flag` appears (a higher-priority Z4 job preempts; see the council doc).
- Net: **never runs while Mats is at his machine (local or Parsec).** A 45-min health loop verifies this.

**Why NOT the carve-out** (running during his active-but-GPU-idle desk time): tried for ~1h on
2026-06-12, proved unsafe. (a) Mats works via Parsec, which uses the GPU — so "active at the desk"
often means "GPU-busy," and the embedder hit **88% util competing with his stream** before the
health loop caught it. (b) **🚨 Per-process GPU memory reads `[N/A]` on this A4000**
(`nvidia-smi --query-compute-apps=...,used_memory` → `[N/A]`), so the Revit/AutoCAD memory-jump
cede is BLIND and never fires — it can't detect his CAD/Parsec GPU use. (The same flaw affects
brf-auto's `gpu_guard_local.ps1` — flagged in the council doc.) **Console-idle (`quser`) is the only
reliable Mats-present signal**, hence idle-only. Reassurance: a GPU at 100% does not lock up the
machine (CPU/RAM/UI stay fine) — but Parsec IS a GPU app, so his remote session would feel it.

## Run / check / stop
- **Standing setup** (running as of 2026-06-12): the `z4-embed-server` guard task + the
  caffeinated Mac client + the self-healing tunnel are all up; the backlog clears in the next
  idle window.
- **Check count** — `cd ~/Projects/mannaminne/py && .venv/bin/python` then `load_conn()` +
  `SELECT count(*), count(embedding) FROM chunks`. (Do NOT shell-source `db.env` for psql —
  the password mangles in the shell; use the tool's own `psycopg` connection.)
- **Restart guard** — `ssh z4 'schtasks /run /tn z4-embed-server'` (idle-only; self-launches
  the server in idle windows).
- **Stop cleanly** — `ssh z4 'schtasks /end /tn z4-embed-server'` **then**
  `ssh z4 'powershell -File E:\llama-embed\kill_embed.ps1'` (the `/end` force-kills the guard
  so its cleanup is skipped → `kill_embed` prevents an orphaned server).
- **Deploy script changes** — `scp z4/*.ps1 z4/*.bat z4:E:/llama-embed/` then restart the task.
- **Carve-out (NOT recommended)** — `-CarveOut 1` runs during Mats-active GPU-idle time, but his
  Parsec use + the `[N/A]` per-process-GPU-memory detection make it unsafe (see Mats-safety).
  Idle-only is production.

## Known caveats (fix when touched)
- **Orphan on force-kill** — `schtasks /end` skips the guard's `finally`, orphaning the server
  (holds VRAM, uncede-protected). Always `kill_embed.ps1` after; or add a graceful stop-marker;
  or a periodic orphan-sweep (cf. brf-auto `docker-logs-orphan-sweep`).
- **AutoCAD not detected** — the Revit-mem cede greps only `Revit`; widen to `acad` before
  trusting CarveOut=1 (moot for CarveOut=0, which cedes on console-activity, not GPU-app name).
- **Throughput ~40/sec** — overnight-clearable. Tunable higher (more `--parallel`, bigger
  client batches) but not worth it for a one-time backlog.
- **Permanently-failing rows** — a chunk that always errors stays NULL → the client backoff-
  loops on it at the tail. Mark/skip if the backlog stalls near-done.

## Visibility
Committed **LOCALLY only — NOT pushed.** The mannaminne remote (`semikolon/ccsearch`) is
PUBLIC; these scripts reveal fleet/Mats topology AND the repo hardcodes a FalkorDB dev
password (`discover_fyr`). Do not push without sanitizing + a visibility audit.

## Cross-refs
- Cross-project Z4 strategy + cede coordination + same-space reasoning:
  `~/dotfiles/docs/z4_local_model_strategy_cross_project_2026_06_11.md`
- mannaminne design: `~/dotfiles/docs/personal_archives_semantic_search_2026_06_10.md`
- Reused guards: `~/Projects/brf-auto/lib/z4_trial/guard/`
