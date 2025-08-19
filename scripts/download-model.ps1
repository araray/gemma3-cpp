<#
Download a Gemma 3 GGUF model into .\models (prefers huggingface-cli; falls back to Invoke-WebRequest).
Default: ggml-org/gemma-3-4b-it-GGUF : gemma-3-4b-it-Q4_K_M.gguf

Usage:
  .\scripts\download-model.ps1
  .\scripts\download-model.ps1 -Repo ggml-org/gemma-3-12b-it-GGUF -File gemma-3-12b-it-Q4_K_M.gguf
#>
[CmdletBinding()]
param(
  [string]$Repo = "ggml-org/gemma-3-4b-it-GGUF",
  [string]$File = "gemma-3-4b-it-Q4_K_M.gguf",
  [string]$ModelDir = ""
)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Resolve-Path (Join-Path $ScriptDir "..")
if (-not $ModelDir) { $ModelDir = Join-Path $Root "models" }
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

Write-Host "[download] repo=$Repo"
Write-Host "[download] file=$File"
Write-Host "[download] dest=$ModelDir\$File"

# Try huggingface-cli first
$hf = (Get-Command "huggingface-cli" -ErrorAction SilentlyContinue)
if ($hf) {
  Write-Host "[download] using huggingface-cli"
  huggingface-cli download $Repo $File --local-dir $ModelDir --local-dir-use-symlinks False
} else {
  $url = "https://huggingface.co/$Repo/resolve/main/$File?download=true"
  Write-Host "[download] huggingface-cli not found; trying Invoke-WebRequest"
  Write-Host "[download] $url"
  $tmp = Join-Path $ModelDir "$File.partial"
  Invoke-WebRequest -Uri $url -OutFile $tmp
  Move-Item -Force $tmp (Join-Path $ModelDir $File)
}
Write-Host "[download] done."
