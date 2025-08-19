#!/usr/bin/env bash
# Run the smoke test. If MODEL_PATH is set, the test will actually load the model.
# Usage:
#   ./scripts/test.sh
#   MODEL_PATH=./models/gemma-3-4b-it-Q4_K_M.gguf ./scripts/test.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TYPE="${TYPE:-Release}"

export MODEL_PATH="${MODEL_PATH:-${LLM_MODEL:-}}"

echo "[test] Using BUILD_DIR=${BUILD_DIR}, TYPE=${TYPE}"
if [[ -n "${MODEL_PATH}" ]]; then
  echo "[test] MODEL_PATH=${MODEL_PATH}"
else
  echo "[test] MODEL_PATH not set; smoke test will skip model load."
fi

ctest --test-dir "${BUILD_DIR}" -C "${TYPE}" -V -R smoke
