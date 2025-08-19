# gemma3-cpp — CPU‑only Gemma 3 CLI (C++ / llama.cpp)

A compact, cross‑platform C++ command‑line tool that runs **Gemma 3** models **on CPU only** (x86_64 or arm64) using the proven **[llama.cpp](https://github.com/ggml-org/llama.cpp)** backend.
It automatically applies the model’s **chat template**, so you can pass a normal prompt and get a streamed answer.

- **OS:** Linux (Ubuntu 22.04+), macOS (Intel/Apple Silicon), Windows 10/11 (x64).
- **CPU‑only:** we force `n_gpu_layers=0` (no CUDA/Metal/OpenCL required).
- **Models:** Gemma 3 **GGUF** (1B, 4B, 12B, 27B) from the **`ggml-org`** HF collections.

---

## Contents

- [Quick Start (Scripts)](#quick-start-scripts)
- [Manual Build & Run](#manual-build--run)
- [Download a Model (GGUF)](#download-a-model-gguf)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Performance Tips (CPU)](#performance-tips-cpu)
- [Notes & Limitations](#notes--limitations)
- [Alternative: google/gemma.cpp (CPU-only)](#alternative-googlegemmacpp-cpu-only)
- [Project Structure](#project-structure)
- [License](#license)
- [References](#references)

---

## Quick Start (Scripts)

> Linux/macOS (bash)
```bash
git clone https://github.com/yourname/gemma3-cpp.git
cd gemma3-cpp
chmod +x scripts/*.sh

# 1) Build (Release, BLAS ON by default)
./scripts/build.sh

# 2) Download a GGUF (default = Gemma 3 4B IT Q4_K_M)
./scripts/download-model.sh

# 3) Run
./scripts/run.sh -m ./models/gemma-3-4b-it-Q4_K_M.gguf -p "Hello, Gemma 3!"
````

> Windows (PowerShell)

```powershell
git clone https://github.com/yourname/gemma3-cpp.git
cd gemma3-cpp
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 1) Build (Release, BLAS ON by default)
.\scripts\build.ps1

# 2) Download a GGUF (default = Gemma 3 4B IT Q4_K_M)
.\scripts\download-model.ps1

# 3) Run
.\scripts\run.ps1 -ModelPath .\models\gemma-3-4b-it-Q4_K_M.gguf -Prompt "Hello, Gemma 3!"
```

---

## Manual Build & Run

### Prerequisites

**Linux (Ubuntu 22.04+)**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
# Optional (faster CPU GEMM):
sudo apt-get install -y libopenblas-dev
```

**macOS (Intel or Apple Silicon)**

```bash
xcode-select --install
brew install cmake git
# BLAS: Apple's Accelerate is used automatically when enabled.
```

**Windows**

* Install **Visual Studio 2022 Build Tools** (Desktop C++ & CMake) or full VS 2022.
* Use **x64 Native Tools Command Prompt for VS 2022** or PowerShell with VS env.

### Configure & build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BLAS=ON
cmake --build build -j
```

> BLAS vendor:
>
> * Linux: add `-DLLAMA_BLAS_VENDOR=OpenBLAS`
> * macOS: Accelerate is picked automatically
> * Windows: typically skip vendor (unless you provide OpenBLAS yourself)

---

## Download a Model (GGUF)

We use **GGUF** Gemma 3 models from the **`ggml-org`** HF collections.

**Option A (scripts – recommended):**

```bash
# Linux/macOS:
./scripts/download-model.sh
# Windows:
.\scripts\download-model.ps1
```

This downloads **`ggml-org/gemma-3-4b-it-GGUF` → `gemma-3-4b-it-Q4_K_M.gguf`** into `./models/`.

**Option B (huggingface-cli):**

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir ./models --local-dir-use-symlinks False
```

> If a repository requires license acceptance, run `huggingface-cli login` first.

---

## Usage Examples

### Basic single‑turn prompt

```bash
./build/gemma3-cpp -m ./models/gemma-3-4b-it-Q4_K_M.gguf -p "Hello!"
```

### Read prompt from stdin

```bash
echo "Explain the Nyquist–Shannon sampling theorem in one paragraph." \
  | ./build/gemma3-cpp -m ./models/gemma-3-4b-it-Q4_K_M.gguf
```

### Control decoding & context

```bash
./build/gemma3-cpp -m ./models/gemma-3-4b-it-Q4_K_M.gguf \
  -n 256 --temp 0.7 --top-k 50 --top-p 0.9 --min-p 0.05 \
  -c 8192 -t 8 \
  -p "Give 10 embedded signal-processing project ideas."
```

### Deterministic (for tests / reproducibility)

```bash
./build/gemma3-cpp -m ./models/gemma-3-4b-it-Q4_K_M.gguf \
  --temp 0.0 --top-k 1 --top-p 1.0 --seed 123 \
  -p "List the first 5 primes."
```

### Customize system prompt

```bash
./build/gemma3-cpp -m ./models/gemma-3-12b-it-Q4_K_M.gguf \
  --system "You are a precise mathematical assistant." \
  -p "Compute the gradient of x^T A x."
```

> The CLI automatically applies the model’s **chat template**, so your prompt is formatted as the model expects.

---

## Testing

We ship a functional **smoke test** (`tests/smoke.py`) that can run without a model (skip) or with one (real inference):

```bash
# Provide a model path for a real decode:
MODEL_PATH=./models/gemma-3-4b-it-Q4_K_M.gguf ctest -V -R smoke
# Or via scripts:
MODEL_PATH=./models/gemma-3-4b-it-Q4_K_M.gguf ./scripts/test.sh
.\scripts\test.ps1 -ModelPath .\models\gemma-3-4b-it-Q4_K_M.gguf
```

---

## Performance Tips (CPU)

* Use **all cores**: `-t $(nproc)` on Linux or the default on macOS/Windows (scripts auto‑detect).
* **BLAS** helps a lot on CPU (`-DLLAMA_BLAS=ON`); use **OpenBLAS** on Linux, **Accelerate** on macOS.
* Prefer **Q4\_K\_M** for a good speed/quality trade‑off on CPU; **Q8\_0** is higher quality but larger.
* If you see `Illegal instruction`, rebuild without native CPU tuning (e.g., set `-DLLAMA_NATIVE=OFF` in `EXTRA_CMAKE_FLAGS`).

---

## Notes & Limitations

* **CPU‑only**: We hard‑set `n_gpu_layers = 0` in `src/main.cpp`.
* **Text‑only**: For Gemma 3 **vision** variants you’ll need the multimodal examples in `llama.cpp` (additional *mmproj* file).
* **Context length**: Default `-c 8192` is CPU‑friendly; Gemma 3 allows much longer contexts but will require more RAM and time.

---

## Alternative: `google/gemma.cpp` (CPU‑only)

If you prefer Google’s small, focused C++ engine for Gemma 2/3, check **`google/gemma.cpp`** plus **Kaggle** “Gemma C++” artifacts (weights in `.sbs`). Build it and run:

```bash
./gemma --tokenizer tokenizer.spm --weights gemma3-4b-it-sfp.sbs
```

Great when you want *only* Gemma and a very small codebase.

---

## Project Structure

```
src/main.cpp        # C++ CLI using libllama (chat template + streamed decode)
CMakeLists.txt      # Fetches ggml-org/llama.cpp via FetchContent (CPU-only config)
tests/smoke.py      # Functional smoke test (skips if MODEL_PATH unset)
scripts/            # Cross-platform build/test/run/download helpers (bash & PowerShell)
```

---

## License

* This project’s code (excluding third‑party dependencies) is MIT.
* `llama.cpp` is MIT (see its repository for details).
* Model weights are under their respective licenses on Hugging Face / Kaggle.

---

## References

1. Google — *Introducing Gemma 3* (sizes, capabilities, downloads)
   [https://blog.google/technology/developers/gemma-3/](https://blog.google/technology/developers/gemma-3/)

2. ggml‑org — *llama.cpp* (C/C++ inference library)
   [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

3. Hugging Face — *Gemma 3 GGUF collection (`ggml-org`)*
   [https://huggingface.co/collections/ggml-org/gemma-3-67d126315ac810df1ad9e913](https://huggingface.co/collections/ggml-org/gemma-3-67d126315ac810df1ad9e913)

4. Hugging Face — *Gemma 3 12B IT `Q4_K_M` GGUF example*
   [https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/blob/main/gemma-3-12b-it-Q4\_K\_M.gguf](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/blob/main/gemma-3-12b-it-Q4_K_M.gguf)

5. llama.cpp wiki — *Chat template support (`llama_chat_apply_template`)*
   [https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama\_chat\_apply\_template](https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)

6. Google — *gemma.cpp* (official CPU‑only engine for Gemma 2/3)
   [https://github.com/google/gemma.cpp](https://github.com/google/gemma.cpp)

7. Hugging Face — *CLI usage docs*
   [https://huggingface.co/docs/huggingface\_hub/guides/cli](https://huggingface.co/docs/huggingface_hub/guides/cli)
