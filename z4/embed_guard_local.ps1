# embed_guard_local.ps1 — Z4-side lifecycle manager for the mannaminne embed server.
#
# Runs llama-server (Qwen3-Embedding-4B-Q4_K_M, ~3-4 GB VRAM) serving an
# OpenAI-compat /v1/embeddings endpoint, and CEDES the GPU by KILLING the server
# (which fully frees its VRAM — satisfies the brf-auto "unload != pause" contract).
# The Mac client (mannaminne embed) streams the backlog and retries while the
# server is down; the backlog is idempotent (WHERE embedding IS NULL) so a cede
# loses at most the in-flight batch.
#
# Cede triggers (kill llama-server, free VRAM):
#   (a) E:\z4-coord\gpu-preempt.flag present & fresh  -> a higher-priority job wants the GPU
#   (b) Revit GPU memory jumps > RevitJumpMB while we run -> Mats doing heavy GPU work
#   (c) [idle-window mode only, CarveOut=0] console idle < CedeIdleSec -> Mats active
#
# The model is small (~3-4 GB) so it fits the active-GPU-idle CARVE-OUT
# (Fredrik both-YES, 2026-06-12): in CarveOut=1 it may run while Mats is active,
# ceding only on the flag or a real Revit GPU spike. Revit baseline is re-tracked
# whenever we are NOT running, so a permanent false-cede can't happen.
param(
  [string]$Server     = "E:\llama-embed\llama-server.exe",
  [string]$Model      = "E:\llama-embed\Qwen3-Embedding-4B-Q4_K_M.gguf",
  [int]$Port          = 8081,
  [string]$Flag       = "E:\z4-coord\gpu-preempt.flag",
  [int]$FlagStaleMin  = 15,
  [int]$RevitJumpMB   = 600,
  [int]$CarveOut      = 1,      # 1 = may run while Mats active (cede on flag/Revit only); 0 = idle-window only
  [int]$LaunchIdleMin = 20,     # used when CarveOut=0
  [int]$CedeIdleSec   = 90,     # used when CarveOut=0
  [double]$PollSec    = 2.0,
  [int]$MaxMin        = 1440,
  [switch]$DryRun
)
$qs  = "E:\llama-embed\quser_idle.ps1"
$log = "E:\llama-embed\embed_guard.log"
function Now(){ [DateTime]::Now.ToString('o') }
function Log($m){ "$(Now) $m" | Add-Content $log }
function IdleSec(){ try { [int](& $qs) } catch { 0 } }
function CadPresent(){
  # Is Mats doing local CAD work? Detect by PROCESS PRESENCE (Revit/AutoCAD running), since
  # per-process GPU memory reads [N/A] on this A4000 (memory-jump detection is impossible) and
  # quser console-idle is unreliable here (shows "active" for hours after he leaves).
  # NOTE: Parsec-background is deliberately NOT included — parsecd is GPU-present even when Mats
  # isn't connected; active-Parsec disruption is handled by the operator + the 45-min health loop.
  try {
    $rows = nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>$null
    foreach($r in $rows){ if($r -match 'Revit|acad'){ return $true } }
    $false
  } catch { $false }
}
function FlagFresh(){
  if(-not (Test-Path $Flag)){ return $false }
  try { return ((New-TimeSpan -Start (Get-Item $Flag).LastWriteTime -End (Get-Date)).TotalMinutes -le $FlagStaleMin) } catch { return $true }
}
function ServerProc(){
  Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -EA SilentlyContinue |
    Where-Object { $_.CommandLine -match 'llama-embed' }
}
function StartServer(){
  if(ServerProc){ return }
  Log "START llama-server :$Port"
  if($DryRun){ Log "DRYRUN would-start"; return }
  # A4000 throughput config (vs Darwin's GTX-1650 batch-of-2/single-stream): 8 parallel
  # slots + room for batched requests. Throughput-only — same GGUF/pooling => same-space.
  Start-Process -FilePath $Server -WindowStyle Hidden -ArgumentList @(
    "--model",$Model,"--embedding","--host","0.0.0.0","--port","$Port",
    "-ngl","99","--parallel","4","--ctx-size","8192","--batch-size","4096","--ubatch-size","4096",
    "--threads","4","--threads-batch","8","--flash-attn","on","--mlock","--no-webui"
  ) -RedirectStandardOutput "E:\llama-embed\llama-server.out" -RedirectStandardError "E:\llama-embed\llama-server.err"
}
function StopServer($why){
  $p = ServerProc
  if(-not $p){ return }
  Log "CEDE/STOP ($why)"
  if($DryRun){ Log "DRYRUN would-kill"; return }
  $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
}
Log "guard start (carveout=$CarveOut, dryrun=$DryRun)"
$end = (Get-Date).AddMinutes($MaxMin)
try {
  while((Get-Date) -lt $end){
    $idle = IdleSec
    $cede = $null
    if(FlagFresh){ $cede = "preempt-flag" }              # a higher-priority Z4 job wants the GPU
    elseif((CadPresent)){ $cede = "cad-present" }         # Revit/AutoCAD running => Mats on local CAD
    elseif($CarveOut -eq 0 -and $idle -lt $CedeIdleSec){ $cede = "idle=${idle}s" }
    if($cede){ StopServer $cede; Start-Sleep -Seconds $PollSec; continue }
    $okToRun = ($CarveOut -eq 1) -or ($idle -ge ($LaunchIdleMin*60))
    if($okToRun){ StartServer } else { StopServer "preflight-idle=${idle}s" }
    Start-Sleep -Seconds $PollSec
  }
} finally {
  StopServer "guard-exit"   # never orphan VRAM if the guard ends/dies gracefully
  Log "guard exit"
}
