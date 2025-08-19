#!/usr/bin/env python3
"""
Functional smoke test for gemma3-cpp.

Usage (ctest sets this up):
  MODEL_PATH=./models/gemma-3-270m-it-qat-Q4_0.gguf ctest -V -R smoke
"""
import os, subprocess, sys, shutil

exe = sys.argv[1]
model = os.environ.get("MODEL_PATH") or os.environ.get("LLM_MODEL") or ""
if not model or not os.path.exists(model):
    print("[smoke] MODEL_PATH not set or file missing; skipping (pass MODEL_PATH=...)")
    sys.exit(0)  # skip gracefully

cmd = [exe, "-m", model, "-p", "Say 'ok' and stop.", "-n", "16", "--threads", "2", "--temp", "0.0"]
print("[smoke] running:", " ".join(cmd))
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
print("[stdout]\n", p.stdout[-500:])
print("[stderr]\n", p.stderr[-500:], file=sys.stderr)
# Very weak assertion: produced any text and exit 0
assert p.returncode == 0, f"non-zero exit: {p.returncode}"
assert len(p.stdout.strip()) > 0, "no output"
print("[smoke] PASS")
