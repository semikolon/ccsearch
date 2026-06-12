# idle_research.ps1 — probe reliable idle + per-process GPU signals on the Z4.
# Run via ssh (session 0): the PERF COUNTERS are system-wide (work here); the input/foreground
# Win32 APIs are session-scoped (will read the ssh session, not Mats's session 1 — informative only).
$ErrorActionPreference='SilentlyContinue'
function NameOf($processId){ try { (Get-Process -Id $processId).ProcessName } catch { "?" } }

Write-Output "=== 1. GPU Engine per-process UTILIZATION (PDH counters = what Task Manager uses; bypasses nvidia-smi [N/A]) ==="
$s = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -EA SilentlyContinue).CounterSamples | Where-Object {$_.CookedValue -gt 1}
if(-not $s){ Write-Output "  (no engine >1% util right now)" }
foreach($x in $s){
  if($x.InstanceName -match 'pid_(\d+).*engtype_(\w+)'){
    Write-Output ("  {0,-24} pid {1,-7} eng {2,-12} util {3:N0}%" -f (NameOf $matches[1]),$matches[1],$matches[2],$x.CookedValue)
  }
}

Write-Output "=== 2. GPU Process Memory per-process (nvidia-smi shows [N/A]; this is Task Manager's source) ==="
$m = (Get-Counter '\GPU Process Memory(*)\Local Usage' -EA SilentlyContinue).CounterSamples | Where-Object {$_.CookedValue -gt 50MB}
if(-not $m){ Write-Output "  (none > 50MB / counter unavailable)" }
foreach($x in $m){
  if($x.InstanceName -match 'pid_(\d+)'){ Write-Output ("  {0,-24} pid {1,-7} {2:N0} MB" -f (NameOf $matches[1]),$matches[1],($x.CookedValue/1MB)) }
}

Write-Output "=== 3. GetLastInputInfo idle (session-scoped; from ssh it reads the WRONG station) ==="
Add-Type @'
using System; using System.Runtime.InteropServices;
public class LII { [StructLayout(LayoutKind.Sequential)] public struct L { public uint cbSize; public uint dwTime; }
[DllImport("user32.dll")] static extern bool GetLastInputInfo(ref L p);
[DllImport("kernel32.dll")] static extern uint GetTickCount();
public static double IdleSec(){ var l=new L(); l.cbSize=(uint)Marshal.SizeOf(l); GetLastInputInfo(ref l); return (GetTickCount()-l.dwTime)/1000.0; } }
'@
Write-Output ("  GetLastInputInfo idle = {0:N0}s" -f [LII]::IdleSec())

Write-Output "=== 4. quser (the unreliable one, for comparison) ==="
quser 2>$null

Write-Output "=== 5. lock / LogonUI / screensaver ==="
if(Get-Process LogonUI -EA SilentlyContinue){ Write-Output "  LOCKED (LogonUI running)" } else { Write-Output "  not locked (no LogonUI)" }

Write-Output "=== 6. Parsec: process + active peer connections (established = someone viewing) ==="
$pp=(Get-Process parsecd -EA SilentlyContinue).Id
if($pp){
  Write-Output ("  parsecd pid(s): {0}" -f ($pp -join ','))
  foreach($p in $pp){
    $est = Get-NetTCPConnection -OwningProcess $p -State Established -EA SilentlyContinue
    $udp = Get-NetUDPEndpoint -OwningProcess $p -EA SilentlyContinue | Where-Object { $_.LocalAddress -ne '0.0.0.0' -and $_.LocalAddress -ne '::' }
    Write-Output ("    pid {0}: {1} established TCP, {2} bound UDP" -f $p,($est|Measure-Object).Count,($udp|Measure-Object).Count)
    $est | Select-Object -First 3 | ForEach-Object { Write-Output ("      TCP est -> {0}:{1}" -f $_.RemoteAddress,$_.RemotePort) }
  }
} else { Write-Output "  parsecd not running" }

Write-Output "=== 7. input-injecting suspects running (could keep quser 'active') ==="
Get-Process logioptionsplus_agent,LogiOptionsMgr,parsecd,ScreenToGif,Caffeine,Amphetamine,Powertoys.AlwaysOnTop,PowerToys -EA SilentlyContinue | Select-Object -Expand ProcessName -Unique | ForEach-Object { Write-Output "  $_" }
