# smoke_embed.ps1 — one-shot validation: load llama-server, embed test strings,
# report dim + head values + GPU mem, then GUARANTEE a kill. Binds 127.0.0.1
# (local curl, no tunnel). Short-lived; for first-GPU-load validation.
param([int]$WaitSec = 40)
$srv = "E:\llama-embed\llama-server.exe"
$mdl = "E:\llama-embed\Qwen3-Embedding-4B-Q4_K_M.gguf"
$p = Start-Process $srv -PassThru -WindowStyle Hidden -ArgumentList @(
  "--model",$mdl,"--embedding","--host","127.0.0.1","--port","8081",
  "-ngl","99","--ctx-size","512","--batch-size","512","--ubatch-size","512","--flash-attn","on","--no-webui"
) -RedirectStandardOutput "E:\llama-embed\smoke.out" -RedirectStandardError "E:\llama-embed\smoke.err"
$ok = $false; $i = 0
for($i=0; $i -lt $WaitSec; $i++){
  Start-Sleep 1
  try { $h = Invoke-WebRequest -Uri "http://127.0.0.1:8081/health" -UseBasicParsing -TimeoutSec 2; if($h.StatusCode -eq 200){ $ok=$true; break } } catch {}
}
Write-Output ("READY=" + $ok + " after ${i}s")
if($ok){
  try {
    $body = @{ input = @("hello world","skuldsanering and inkasso krav") } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8081/v1/embeddings" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
    $e0 = $r.data[0].embedding
    Write-Output ("COUNT=" + $r.data.Count + " DIM=" + $e0.Count)
    Write-Output ("HELLO_HEAD8=" + (($e0[0..7]) -join ","))
  } catch { Write-Output ("EMBED_ERR: " + $_.Exception.Message) }
  try { Write-Output ("GPU=" + (nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader)) } catch {}
} else {
  Write-Output "SERVER_NOT_READY -- tail of smoke.err:"
  if(Test-Path E:\llama-embed\smoke.err){ Get-Content E:\llama-embed\smoke.err -Tail 8 }
}
Stop-Process -Id $p.Id -Force -EA SilentlyContinue
Start-Sleep 2
Write-Output "KILLED"
