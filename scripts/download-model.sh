#!/usr/bin/env bash
# Download a Gemma 3 GGUF model into ./models (prefers huggingface-cli; falls back to curl).
# Default: ggml-org/gemma-3-4b-it-GGUF : gemma-3-4b-it-Q4_K_M.gguf
# Usage:
#   ./scripts/download-model.sh
#   REPO=ggml-org/gemma-3-12b-it-GGUF FILE=gemma-3-12b-it-Q4_K_M.gguf ./scripts/download-model.sh
set -euo pipefail

REPO="${REPO:-ggml-org/gemma-3-4b-it-GGUF}"
FILE="${FILE:-gemma-3-4b-it-Q4_K_M.gguf}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"
mkdir -p "${MODEL_DIR}"

echo "[download] repo=${REPO}"
echo "[download] file=${FILE}"
echo "[download] dest=${MODEL_DIR}/${FILE}"

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "[download] using huggingface-cli"
  huggingface-cli download "${REPO}" "${FILE}" --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
else
  echo "[download] huggingface-cli not found; trying curl"
  URL="https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"
  echo "[download] ${URL}"
  curl -L --fail -o "${MODEL_DIR}/${FILE}.partial" "${URL}"
  mv "${MODEL_DIR}/${FILE}.partial" "${MODEL_DIR}/${FILE}"
fi

echo "[download] done."
