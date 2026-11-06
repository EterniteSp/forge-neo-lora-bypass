<#
.SYNOPSIS
  Installs the LoRA Bypass extension (sd-forge-lora-bypass) into Forge Neo.

.DESCRIPTION
  Copies the folder sd-forge-lora-bypass into <ForgeDir>\extensions\ and writes the manifest
  <ForgeDir>\.lora-bypass-install.json (version + SHA-256 of every installed file), which verify.ps1 and
  rollback.ps1 use. Nothing else is changed: no Forge core files, no git branches, no config.json, no venv.

  Refuses to run while Forge is running (its port is listening), when another version is installed
  (run rollback.ps1 first), or when an extensions\sd-forge-lora-bypass folder exists that this script did not install.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -ForgeDir "D:\Forge Neo"
#>
param(
  [string]$ForgeDir = "C:\AI\Forge Neo",
  [int]$ForgePort = 7860
)

$ErrorActionPreference = "Stop"
$Name = "sd-forge-lora-bypass"
$Src = Join-Path $PSScriptRoot $Name
$Manifest = Join-Path $ForgeDir ".lora-bypass-install.json"
$Dst = Join-Path $ForgeDir "extensions\$Name"

function Fail([string]$msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

function Get-ExtVersion([string]$dir) {
  $f = Join-Path $dir "scripts\lora_bypass.py"
  if (-not (Test-Path $f)) { return $null }
  $m = Select-String -Path $f -Pattern '^VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
  if ($m) { return $m.Matches[0].Groups[1].Value } else { return $null }
}

function Get-SourceFiles {
  Get-ChildItem -Path $Src -Recurse -File | Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne ".pyc" }
}

# ---- checks -----------------------------------------------------------------------------------------------------------
if (-not (Test-Path $Src)) { Fail "extension folder not found next to this script: $Src" }
if (-not (Test-Path (Join-Path $ForgeDir "launch.py")) -or -not (Test-Path (Join-Path $ForgeDir "backend\patcher\base.py"))) {
  Fail "'$ForgeDir' does not look like a Forge Neo folder (launch.py / backend\patcher\base.py missing). Use -ForgeDir."
}
if (Get-NetTCPConnection -LocalPort $ForgePort -State Listen -ErrorAction SilentlyContinue) {
  Fail "Forge is running (port $ForgePort is listening). Close Forge first, then run install.ps1 again."
}
$NewVersion = Get-ExtVersion $Src
if (-not $NewVersion) { Fail "cannot read VERSION from $Src\scripts\lora_bypass.py" }

if (Test-Path $Manifest) {
  $m = Get-Content $Manifest -Raw | ConvertFrom-Json
  if ($m.version -ne $NewVersion) {
    Fail "LoRA Bypass $($m.version) is already installed. Run rollback.ps1 first, then install $NewVersion."
  }
  $bad = @($m.files | Where-Object {
      $p = Join-Path $ForgeDir $_.path
      -not (Test-Path $p) -or (Get-FileHash -Algorithm SHA256 $p).Hash -ne $_.sha256 })
  if ($bad.Count -eq 0) {
    Write-Host "LoRA Bypass $NewVersion is already installed and intact - nothing to do." -ForegroundColor Green
    exit 0
  }
  Fail "LoRA Bypass $NewVersion is installed but $($bad.Count) file(s) differ or are missing. Run rollback.ps1, then install.ps1."
}
if (Test-Path $Dst) {
  Fail "'$Dst' already exists but was not installed by this script (no manifest). Move or delete it yourself, then run install.ps1."
}

# ---- install ----------------------------------------------------------------------------------------------------------
$files = @()
foreach ($f in Get-SourceFiles) {
  $rel = $f.FullName.Substring($Src.Length).TrimStart("\")
  $target = Join-Path $Dst $rel
  New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
  Copy-Item -LiteralPath $f.FullName -Destination $target
  $files += [pscustomobject]@{
    path   = "extensions\$Name\$rel"
    sha256 = (Get-FileHash -Algorithm SHA256 $target).Hash
  }
}
[pscustomobject]@{
  name         = $Name
  version      = $NewVersion
  installed_at = (Get-Date).ToString("s")
  forge_dir    = $ForgeDir
  files        = $files
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $Manifest

Write-Host "Installed LoRA Bypass $NewVersion -> $Dst ($($files.Count) files)." -ForegroundColor Green
Write-Host "Manifest: $Manifest"
Write-Host "Next: start Forge, then Settings -> LoRA Bypass (enabled by default). Check with verify.ps1."
