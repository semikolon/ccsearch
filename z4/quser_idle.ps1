# quser_idle.ps1 — prints the SMALLEST console/RDP session idle time in SECONDS.
# Copied verbatim from the brf-auto Z4 guard (the validated session-aware signal).
#
# WHY quser (not GetLastInputInfo): GetLastInputInfo only reports input for the
# CALLING process's session. An ssh process and a session-0 scheduled task both
# live in non-interactive stations, so their idle counts up forever and never
# reflects Mats's real activity at the physical console. quser queries the session
# manager and reports the true per-session idle for every logged-on session.
# Fail-safe: prints 0 (== active) on any uncertainty -> callers err toward ceding.
try {
  $out = quser 2>$null
  if (-not $out) { '0'; exit }
  $min = $null
  foreach ($l in $out) {
    if ($l -match '(\S+)\s+\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}\s*$') {
      $t = $matches[1]; $s = 0
      if ($t -eq 'none' -or $t -eq '.') { $s = 0 }
      elseif ($t -match '^(\d+)\+(\d+):(\d+)$') { $s = [int]$matches[1]*86400 + [int]$matches[2]*3600 + [int]$matches[3]*60 }
      elseif ($t -match '^(\d+):(\d+)$') { $s = [int]$matches[1]*3600 + [int]$matches[2]*60 }
      elseif ($t -match '^(\d+)$') { $s = [int]$matches[1]*60 }
      else { $s = 0 }
      if ($null -eq $min -or $s -lt $min) { $min = $s }
    }
  }
  if ($null -eq $min) { '0' } else { $min.ToString() }
} catch { '0' }
