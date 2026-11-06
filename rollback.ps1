<#
.SYNOPSIS
  Removes the LoRA Bypass extension installed by install.ps1.

.DESCRIPTION
  Deletes exactly the files listed in <ForgeDir>\.lora-bypass-install.json, the __pycache__ folders Python created
  inside the extension folder, then the (now empty) extension folders and the manifest. Files that were not
  installed by install.ps1 are kept (and reported). config.json is not touched: the saved options
  lora_bypass_enabled / lora_bypass_verbose stay there and are ignored by Forge without the extension.
  Refuses to run while Forge is running.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\rollback.ps1
#>
param(
  [string]$ForgeDir = "C:\AI\Forge Neo",
  [int]$ForgePort = 7860
)

$ErrorActionPreference = "Stop"
$Name = "sd-forge-lora-bypass"
$Manifest = Join-Path $ForgeDir ".lora-bypass-install.json"
$Ext = Join-Path $ForgeDir "extensions\$Name"

function Fail([string]$msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (Get-NetTCPConnection -LocalPort $ForgePort -State Listen -ErrorAction SilentlyContinue) {
  Fail "Forge is running (port $ForgePort is listening). Close Forge first, then run rollback.ps1 again."
}
if (-not (Test-Path $Manifest)) {
  if (Test-Path $Ext) { Fail "no manifest ($Manifest), but '$Ext' exists: it was not installed by install.ps1. Delete that folder yourself if you want it gone." }
  Write-Host "LoRA Bypass is not installed (no manifest) - nothing to do." -ForegroundColor Green
  exit 0
}
$m = Get-Content $Manifest -Raw | ConvertFrom-Json

$removed = 0
$changed = @()
foreach ($f in $m.files) {
  $p = Join-Path $ForgeDir $f.path
  if (-not (Test-Path $p)) { continue }
  if ((Get-FileHash -Algorithm SHA256 $p).Hash -ne $f.sha256) { $changed += $f.path }
  Remove-Item -LiteralPath $p -Force
  $removed++
}
if (Test-Path $Ext) {
  Get-ChildItem -Path $Ext -Recurse -Directory -Filter "__pycache__" | Sort-Object FullName -Descending |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
  # remove empty folders, deepest first
  Get-ChildItem -Path $Ext -Recurse -Directory | Sort-Object { $_.FullName.Length } -Descending | ForEach-Object {
    if (-not (Get-ChildItem -LiteralPath $_.FullName -Force)) { Remove-Item -LiteralPath $_.FullName -Force }
  }
  if (-not (Get-ChildItem -LiteralPath $Ext -Force)) { Remove-Item -LiteralPath $Ext -Force }
}
Remove-Item -LiteralPath $Manifest -Force

Write-Host "Removed LoRA Bypass $($m.version): $removed file(s)." -ForegroundColor Green
if ($changed.Count -gt 0) { Write-Host "Note: $($changed.Count) of them had been modified after installation: $($changed -join ', ')" }
if (Test-Path $Ext) {
  Write-Host "Kept '$Ext': it contains files that install.ps1 did not put there:" -ForegroundColor Yellow
  Get-ChildItem -Path $Ext -Recurse -File | ForEach-Object { Write-Host "  $($_.FullName)" }
}
