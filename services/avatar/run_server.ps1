# Start the live avatar server.
#   .\run_server.ps1
#   .\run_server.ps1 -Backend flashhead
#   .\run_server.ps1 -Backend flashhead-pro -KillExisting
#   .\run_server.ps1 -Backend liveportrait -KillExisting

param(
  [string]$Backend = $env:AVATAR_BACKEND,
  [switch]$KillExisting
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = 8100

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $Backend) { $Backend = "flashhead" }
$env:AVATAR_BACKEND = $Backend.ToLower()

function Get-PortPids([int]$PortNum) {
  $pids = @()
  try {
    $pids = @(Get-NetTCPConnection -LocalPort $PortNum -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique)
  } catch {}
  return @($pids | Where-Object { $_ -and $_ -gt 0 })
}

$existing = Get-PortPids $Port
if ($existing.Count -gt 0) {
  if ($KillExisting) {
    foreach ($procId in $existing) {
      Write-Host "Killing PID $procId on port $Port ..."
      Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
    $existing = Get-PortPids $Port
    if ($existing.Count -gt 0) {
      throw "Port $Port still in use by: $($existing -join ', ')"
    }
  } else {
    Write-Host ""
    Write-Host "ERROR: port $Port already in use by PID(s): $($existing -join ', ')"
    Write-Host "  Another avatar server is already running."
    Write-Host "  Use that one, OR stop it, OR restart with:"
    Write-Host "    .\run_server.ps1 -Backend $Backend -KillExisting"
    Write-Host ""
    exit 1
  }
}

Write-Host "==> Live avatar server: http://127.0.0.1:$Port"
Write-Host "    AVATAR_BACKEND=$($env:AVATAR_BACKEND)  (swap models by restarting with -Backend <id>)"
Write-Host "    Analytics comparison: http://127.0.0.1:$Port/  → Analytics tab"
Set-Location (Join-Path $Root "server")
& (Join-Path $Root ".venv\Scripts\python.exe") -m uvicorn app:app --host 127.0.0.1 --port $Port
