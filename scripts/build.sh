#!/usr/bin/env bash
# Build the CPU-only Gemma 3 CLI on top of llama.cpp
# Usage:
#   ./scripts/build.sh                 # Release, BLAS ON (default)
#   TYPE=Debug BLAS=OFF ./scripts/build.sh
#   EXTRA_CMAKE_FLAGS="-DLLAMA_BLAS_VENDOR=OpenBLAS" ./scripts/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TYPE="${TYPE:-Release}"           # Debug or Release
BLAS="${BLAS:-ON}"                # ON/OFF

CMAKE_FLAGS=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${TYPE}"
  -DLLAMA_BLAS="${BLAS}"
)

# Helpful default vendor selection
uname_s="$(uname -s || true)"
if [[ "${BLAS}" == "ON" ]]; then
  case "${uname_s}" in
    Linux*)  CMAKE_FLAGS+=( -DLLAMA_BLAS_VENDOR=OpenBLAS );;
    Darwin*) : # Apple's Accelerate is auto-picked by llama.cpp; nothing to set
  esac
fi

# EXTRA_CMAKE_FLAGS lets you inject anything (e.g. -DLLAMA_NATIVE=OFF)
if [[ -n "${EXTRA_CMAKE_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMAKE_FLAGS+=( ${EXTRA_CMAKE_FLAGS} )
fi

echo "[build] cmake configure: ${CMAKE_FLAGS[*]}"
cmake "${CMAKE_FLAGS[@]}"

echo "[build] cmake build (${TYPE})"
cmake --build "${BUILD_DIR}" --config "${TYPE}" -j

echo "[build] done -> ${BUILD_DIR}"
