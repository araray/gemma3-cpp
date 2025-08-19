<#
Build the CPU-only Gemma 3 CLI (llama.cpp backend).

Usage:
  .\scripts\build.ps1                       # Release, BLAS=ON (default)
  .\scripts\build.ps1 -Configuration Debug -BLAS:$false
  .\scripts\build.ps1 -ExtraCMakeFlags "-DLLAMA_NATIVE=OFF"
#>
[CmdletBinding()]
param(
  [ValidateSet("Debug","Release")] [string]$Configuration = "Release",
  [bool]$BLAS = $true,
  [string]$BuildDir = "",
  [string]$ExtraCMakeFlags = ""
)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Resolve-Path (Join-Path $ScriptDir "..")
if (-not $BuildDir) { $BuildDir = Join-Path $Root "build" }

$cmakeArgs = @("-S", $Root, "-B", $BuildDir, "-DCMAKE_BUILD_TYPE=$Configuration")
if ($BLAS) { $cmakeArgs += "-DLLAMA_BLAS=ON" } else { $cmakeArgs += "-DLLAMA_BLAS=OFF" }
if ($ExtraCMakeFlags) { $cmakeArgs += $ExtraCMakeFlags }

Write-Host "[build] cmake $($cmakeArgs -join ' ')"
cmake @cmakeArgs

Write-Host "[build] cmake --build $BuildDir --config $Configuration -j"
cmake --build $BuildDir --config $Configuration -j
Write-Host "[build] done -> $BuildDir"
