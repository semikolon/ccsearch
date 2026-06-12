# kill_embed.ps1 — cede ONLY the embedder's llama-server, never other processes.
# The Z4 is shared (brf-auto OCR python in E:\z4-ml, possible other jobs). Scope
# the kill to our own server's command line (cf. global CLAUDE.md "pkill -f
# <bare-name> matches production processes too"). Mirrors brf-auto's kill_infer.ps1.
Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'llama-server' -and $_.CommandLine -match 'llama-embed' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
