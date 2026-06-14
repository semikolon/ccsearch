# Z4 embedder — mannaminne backlog on Mats's RTX A4000

Semantic-embedding backend for mannaminne (and the same-space fallback story for Graphiti).
Runs Qwen3-Embedding-4B on the Z4's A4000 instead of Darwin's contended GTX-1650 for large
batch backfills. **Status (2026-06-14): backlog complete — 934,764/934,764 chunks embedded, HNSW built, Z4 server/guard/client stopped.** Darwin remains the live query-embedding fallback/standing endpoint.

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
The Z4 is Mats's daily workstation (he works locally, and occasionally remotely via Parsec).
Production runs **always-on (CarveOut=1)** with a **CAD-presence cede** (`run_embed.bat` passes
`-CarveOut 1`):
- **No idle gate** — both auto-idle signals are broken on this box (see below), so the guard runs
  continuously and protects Mats by ceding, not by gating.
- **Cede** (~1.5s, kills the server) when: **Revit/AutoCAD is process-present** in
  `nvidia-smi --query-compute-apps` (Mats doing local CAD — works despite `[N/A]` memory), OR
  `E:\z4-coord\gpu-preempt.flag` appears (a higher-priority Z4 job preempts).
- **Active-Parsec** (rare) is NOT auto-detected (`parsecd.exe` is GPU-present even when nobody's
  connected = false positive). Fredrik flags it manually; the 45-min health loop also watches.
- Reassurance: a GPU at 100% does not lock up the machine (CPU/RAM/UI stay fine); only GPU apps
  (CAD viewport, or an active Parsec stream) feel it — and CAD triggers the cede.

**🚨 Why no idle gate — both auto-signals are broken on this A4000 (2026-06-12):**
- **Per-process GPU memory reads `[N/A]`** (`nvidia-smi --query-compute-apps=...,used_memory` →
  `[N/A]`), so any memory-jump cede is BLIND. (Same flaw hits brf-auto's `gpu_guard_local.ps1` —
  flagged in the council doc.)
- **`quser` console-idle is unreliable** — it reads "active" for hours after Mats physically
  leaves, so an idle-≥20-min gate would essentially never fire (backlog would stall). (This also
  means brf-auto's `LAUNCH_IDLE_MIN` OCR gate may never trigger here — flagged for them.)
- Therefore **process-presence** (is Revit/AutoCAD running?) is the only working "Mats doing GPU
  work" signal, and the embedder runs always-on + cedes on it.

## Run / check / stop
- **Future batch run**: start the `z4-embed-server` guard task, start the SSH tunnel supervisor,
  then run the Mac client under `caffeinate` with
  `MANNAMINNE_EMBED_URL=http://127.0.0.1:8081/v1/embeddings`. The 2026-06-13 backlog is
  complete; this runbook is for future backfills/re-embeds.
- **Check count** — `cd ~/Projects/mannaminne/py && .venv/bin/python` then `load_conn()` +
  `SELECT count(*), count(embedding) FROM chunks`. (Do NOT shell-source `db.env` for psql —
  the password mangles in the shell; use the tool's own `psycopg` connection.)
- **Restart guard** — `ssh z4 'schtasks /run /tn z4-embed-server'` (starts the local guard; it
  launches/stops the server per the current `embed_guard_local.ps1` policy).
- **Stop cleanly** — `ssh z4 'schtasks /end /tn z4-embed-server'` **then**
  `ssh z4 'powershell -File E:\llama-embed\kill_embed.ps1'` (the `/end` force-kills the guard
  so its cleanup is skipped → `kill_embed` prevents an orphaned server).
- **Deploy script changes** — `scp z4/*.ps1 z4/*.bat z4:E:/llama-embed/` then restart the task.
- **Idle-window fallback** — `-CarveOut 0` is available, but is not production on this box:
  `quser` stayed active for hours after Mats left, so idle-only mode can stall indefinitely.

## Known caveats (fix when touched)
- **Orphan on force-kill** — `schtasks /end` skips the guard's `finally`, orphaning the server
  (holds VRAM, uncede-protected). Always `kill_embed.ps1` after; or add a graceful stop-marker;
  or a periodic orphan-sweep (cf. brf-auto `docker-logs-orphan-sweep`).
- **CAD detection is process-presence only** — `Revit|acad` in `nvidia-smi` is the working local
  CAD signal. It deliberately ignores background `parsecd.exe`; active Parsec still needs
  operator/health-loop attention.
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
