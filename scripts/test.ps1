<#
Run the smoke test. If -ModelPath is set, the test will load it.
Usage:
  .\scripts\test.ps1
  .\scripts\test.ps1 -ModelPath .\models\gemma-3-4b-it-Q4_K_M.gguf
#>
[CmdletBinding()]
param(
  [string]$ModelPath = "",
  [ValidateSet("Debug","Release")] [string]$Configuration = "Release",
  [string]$BuildDir = ""
)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Resolve-Path (Join-Path $ScriptDir "..")
if (-not $BuildDir) { $BuildDir = Join-Path $Root "build" }

if ($ModelPath) { $env:MODEL_PATH = $ModelPath }

Write-Host "[test] BUILD_DIR=$BuildDir, CONFIG=$Configuration"
if ($env:MODEL_PATH) { Write-Host "[test] MODEL_PATH=$env:MODEL_PATH" } else { Write-Host "[test] MODEL_PATH not set; test may skip model load." }

ctest --test-dir $BuildDir -C $Configuration -V -R smoke
