<#
.SYNOPSIS
  Checks the LoRA Bypass installation in Forge Neo (read-only).

.DESCRIPTION
  - manifest <ForgeDir>\.lora-bypass-install.json exists, every installed file is present and unmodified (SHA-256);
  - the Forge code the extension hooks into is still there (ModelPatcher.add_patches with online_mode,
    OnlineLoRAPatch, object patches, Forge Linear weight functions) - useful after a Forge update;
  - prints the current "Diffusion in Low Bits" mode and the extension's setting from config.json.
  Exit code 0 = OK, 1 = problem found. Changes nothing.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
#>
param(
  [string]$ForgeDir = "C:\AI\Forge Neo"
)

$ErrorActionPreference = "Stop"
$Name = "sd-forge-lora-bypass"
$Manifest = Join-Path $ForgeDir ".lora-bypass-install.json"
$problems = 0

function Ok([string]$m) { Write-Host "  OK    $m" -ForegroundColor Green }
function Bad([string]$m) { Write-Host "  FAIL  $m" -ForegroundColor Red; $script:problems++ }
function Info([string]$m) { Write-Host "  INFO  $m" }

Write-Host "LoRA Bypass - verify ($ForgeDir)"
if (-not (Test-Path $Manifest)) {
  Bad "manifest not found: $Manifest (not installed by install.ps1)"
  exit 1
}
$m = Get-Content $Manifest -Raw | ConvertFrom-Json
Ok "manifest: version $($m.version), installed $($m.installed_at), $(@($m.files).Count) files"

foreach ($f in $m.files) {
  $p = Join-Path $ForgeDir $f.path
  if (-not (Test-Path $p)) { Bad "missing: $($f.path)"; continue }
  if ((Get-FileHash -Algorithm SHA256 $p).Hash -ne $f.sha256) { Bad "modified: $($f.path)" }
}
if ($problems -eq 0) { Ok "all installed files present and unmodified" }

$ext = Join-Path $ForgeDir "extensions\$Name"
$extra = @(Get-ChildItem -Path $ext -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $rel = "extensions\$Name\" + $_.FullName.Substring($ext.Length).TrimStart("\")
    ($_.FullName -notmatch '\\__pycache__\\') -and -not ($m.files.path -contains $rel) })
if ($extra.Count -gt 0) { Info "extra files in the extension folder (not from install.ps1): $($extra.Count)" }

# Forge hook points (the extension falls back to Forge's own path if they change, but then it does nothing)
$checks = @(
  @{ file = "backend\patcher\base.py"; pattern = "def add_patches\(self.*online_mode"; what = "ModelPatcher.add_patches(online_mode)" },
  @{ file = "backend\patcher\base.py"; pattern = "class OnlineLoRAPatch"; what = "OnlineLoRAPatch" },
  @{ file = "backend\patcher\base.py"; pattern = "def add_object_patch"; what = "ModelPatcher.add_object_patch" },
  @{ file = "backend\patcher\base.py"; pattern = "m\.weight_function\.extend\(self\.weight_wrapper_patches"; what = "on-the-fly weight functions" },
  @{ file = "backend\operations.py"; pattern = "def weights_manual_cast"; what = "weights_manual_cast" },
  @{ file = "modules_forge\packages\comfy\weight_adapter\lora.py"; pattern = "class LoRAAdapter"; what = "LoRAAdapter" }
)
foreach ($c in $checks) {
  $p = Join-Path $ForgeDir $c.file
  if ((Test-Path $p) -and (Select-String -Path $p -Pattern $c.pattern -Quiet)) { Ok "Forge: $($c.what)" }
  else { Bad "Forge: $($c.what) not found in $($c.file) - Forge changed, the extension may stay inactive" }
}

$cfg = Join-Path $ForgeDir "config.json"
if (Test-Path $cfg) {
  try {
    $j = Get-Content $cfg -Raw -Encoding UTF8 | ConvertFrom-Json
    $mode = $j.forge_unet_storage_dtype
    if ($mode -like "*(fp16 LoRA)*") { Ok "Diffusion in Low Bits = '$mode' (bypass can work)" }
    else { Info "Diffusion in Low Bits = '$mode' - bypass works only in '... (fp16 LoRA)' modes (and GGUF)" }
    $en = $j.lora_bypass_enabled
    if ($null -eq $en) { Info "lora_bypass_enabled not saved yet (default: on)" }
    elseif ($en) { Ok "LoRA bypass setting: on" } else { Info "LoRA bypass setting: OFF (Settings -> LoRA Bypass)" }
  } catch { Info "config.json could not be parsed: $_" }
}

if ($problems -eq 0) { Write-Host "RESULT: OK" -ForegroundColor Green; exit 0 }
Write-Host "RESULT: $problems problem(s)" -ForegroundColor Red
exit 1
