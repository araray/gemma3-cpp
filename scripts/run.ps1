<#
Run the CLI with a prompt (reads stdin if -Prompt not given).

Usage:
  .\scripts\run.ps1 -ModelPath .\models\gemma-3-4b-it-Q4_K_M.gguf -Prompt "Hello!"
  "Explain FFT" | .\scripts\run.ps1 -ModelPath .\models\gemma-3-4b-it-Q4_K_M.gguf
#>
[CmdletBinding()]
param(
  [string]$ModelPath = "",
  [string]$Prompt = "",
  [string]$System = "You are a helpful assistant.",
  [int]$NPredict = 512,
  [int]$Ctx = 8192,
  [int]$Threads = 0,
  [ValidateSet("Debug","Release")] [string]$Configuration = "Release",
  [string]$BuildDir = ""
)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Resolve-Path (Join-Path $ScriptDir "..")
if (-not $BuildDir) { $BuildDir = Join-Path $Root "build" }

$Bin = Join-Path $BuildDir "gemma3-cpp"
if (-not (Test-Path $Bin)) { Write-Error "[run] Binary not found: $Bin. Build first."; exit 1 }

if (-not $ModelPath) {
  $ModelPath = $env:MODEL_PATH
  if (-not $ModelPath) { $ModelPath = Join-Path (Join-Path $Root "models") "gemma-3-4b-it-Q4_K_M.gguf" }
}
if (-not (Test-Path $ModelPath)) { Write-Error "[run] Model not found: $ModelPath"; exit 2 }

if ($Threads -le 0) { $Threads = [Environment]::ProcessorCount }

$cmd = @(
  $Bin, "-m", $ModelPath,
  "-c", "$Ctx", "-t", "$Threads", "-n", "$NPredict",
  "--system", $System
)
if ($Prompt) { $cmd += @("-p", $Prompt) }

Write-Host "[run] $($cmd -join ' ')"
& $cmd
