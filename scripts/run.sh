#!/usr/bin/env bash
# Run the CLI with a prompt (reads from stdin if -p/--prompt is omitted).
# Usage:
#   ./scripts/run.sh -m ./models/gemma-3-4b-it-Q4_K_M.gguf -p "Hello!"
#   echo "Explain FFT" | ./scripts/run.sh -m ./models/gemma-3-4b-it-Q4_K_M.gguf
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TYPE="${TYPE:-Release}"

BIN="${BUILD_DIR}/gemma3-cpp"
[[ -f "${BIN}" ]] || { echo "[run] Binary not found: ${BIN}. Build first."; exit 1; }

MODEL_PATH=""
PROMPT=""
SYSTEM="You are a helpful assistant."
N_PREDICT="${N_PREDICT:-512}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-}"

# Determine default threads if not set
if [[ -z "${THREADS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    THREADS="$(sysctl -n hw.ncpu)"
  else
    THREADS="4"
  fi
fi

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)   MODEL_PATH="$2"; shift 2;;
    -p|--prompt)  PROMPT="$2";     shift 2;;
    --system)     SYSTEM="$2";     shift 2;;
    -n|--n-predict) N_PREDICT="$2"; shift 2;;
    -c|--ctx)     CTX="$2";        shift 2;;
    -t|--threads) THREADS="$2";    shift 2;;
    -h|--help)
      echo "Usage: $0 -m MODEL.gguf [-p PROMPT] [--system SYS] [-n N] [-c CTX] [-t THREADS]"
      exit 0;;
    *) echo "[run] Unknown arg: $1"; exit 2;;
  esac
done

# Resolve model path: env fallback -> default models dir
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${MODEL_PATH:-${LLM_MODEL:-${ROOT_DIR}/models/gemma-3-4b-it-Q4_K_M.gguf}}"
fi
[[ -f "${MODEL_PATH}" ]] || { echo "[run] Model not found: ${MODEL_PATH}"; exit 2; }

CMD=( "${BIN}" -m "${MODEL_PATH}" -c "${CTX}" -t "${THREADS}" -n "${N_PREDICT}" --system "${SYSTEM}" )
if [[ -n "${PROMPT}" ]]; then
  CMD+=( -p "${PROMPT}" )
fi

echo "[run] ${CMD[*]}"
exec "${CMD[@]}"
