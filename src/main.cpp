/**
 * @file main.cpp
 * @brief CPU-only Gemma 3 inference CLI using libllama (llama.cpp).
 *
 * This tool:
 *  - loads a Gemma 3 GGUF model on CPU (no GPU),
 *  - applies the model's chat template automatically,
 *  - generates a streamed response to a single prompt.
 *
 * Build: CMake project pulls ggml-org/llama.cpp via FetchContent.
 *
 * References:
 *  - llama.cpp supports Gemma 3 GGUF and shows -hf examples.
 *  - Official Gemma 3 GGUF repos: ggml-org/gemma-3-*-GGUF on HF.
 */

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

struct Options {
    std::string model_path;        ///< Path to .gguf file
    std::optional<std::string> prompt; ///< Prompt; if not set, read from stdin
    std::string system = "You are a helpful assistant.";
    int32_t n_ctx     = 8192;      ///< Context tokens (Gemma 3 supports much larger; 8k is CPU-friendly)
    int32_t n_predict = 512;       ///< Max generated tokens
    int32_t threads   = std::max(1u, std::thread::hardware_concurrency());
    int32_t seed      = LLAMA_DEFAULT_SEED;
    float   temp      = 0.8f;
    int32_t top_k     = 40;
    float   top_p     = 0.95f;
    float   min_p     = 0.05f;
    bool    color     = true;
};

static void print_usage(const char *argv0) {
    std::cout <<
R"(gemma3-cpp (CPU-only) — run Gemma 3 GGUF models with libllama

USAGE:
  )" << argv0 << R"( -m /path/to/model.gguf [options] --prompt "Your question..."
  )" << argv0 << R"( -m /path/to/model.gguf [options]    # then type the prompt on stdin and press Enter

REQUIRED:
  -m, --model PATH         GGUF model file (e.g., gemma-3-4b-it-Q4_K_M.gguf)

PROMPT INPUT:
  -p, --prompt TEXT        Single-turn user prompt (if omitted, reads a line from stdin)
  --system TEXT            System message (default: "You are a helpful assistant.")

DECODE / CONTEXT:
  -n, --n-predict N        Max new tokens to generate (default: 512)
  -c, --ctx N              Context length / KV cache (default: 8192)
  -t, --threads N          CPU threads (default: num cores)
  --temp F                 Temperature (default: 0.8)
  --top-k N                top-k (default: 40)
  --top-p F                top-p (default: 0.95)
  --min-p F                min-p (default: 0.05)
  --seed N                 RNG seed (default: random)

MISC:
  --no-color               Disable ANSI coloring
  -h, --help               Show this help
)" << std::endl;
}

static bool parse_int(const char *s, int32_t &out) {
    try { out = std::stoi(s); return true; } catch (...) { return false; }
}
static bool parse_float(const char *s, float &out) {
    try { out = std::stof(s); return true; } catch (...) { return false; }
}

static bool parse_args(int argc, char **argv, Options &opt) {
    if (argc <= 1) { print_usage(argv[0]); return false; }
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto need = [&](const char *name)->const char*{
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; exit(2); }
            return argv[++i];
        };
        if (!strcmp(a, "-m") || !strcmp(a, "--model"))       { opt.model_path = need(a); }
        else if (!strcmp(a, "-p") || !strcmp(a, "--prompt"))  { opt.prompt     = need(a); }
        else if (!strcmp(a, "--system"))                      { opt.system     = need(a); }
        else if (!strcmp(a, "-n") || !strcmp(a, "--n-predict")) { if (!parse_int(need(a), opt.n_predict)) return false; }
        else if (!strcmp(a, "-c") || !strcmp(a, "--ctx"))       { if (!parse_int(need(a), opt.n_ctx)) return false; }
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads"))   { if (!parse_int(need(a), opt.threads)) return false; }
        else if (!strcmp(a, "--temp"))                          { if (!parse_float(need(a), opt.temp)) return false; }
        else if (!strcmp(a, "--top-k"))                         { if (!parse_int(need(a), opt.top_k)) return false; }
        else if (!strcmp(a, "--top-p"))                         { if (!parse_float(need(a), opt.top_p)) return false; }
        else if (!strcmp(a, "--min-p"))                         { if (!parse_float(need(a), opt.min_p)) return false; }
        else if (!strcmp(a, "--seed"))                          { if (!parse_int(need(a), opt.seed)) return false; }
        else if (!strcmp(a, "--no-color"))                      { opt.color = false; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help"))      { print_usage(argv[0]); exit(0); }
        else { std::cerr << "Unknown option: " << a << "\n"; print_usage(argv[0]); return false; }
    }
    if (opt.model_path.empty()) { std::cerr << "ERROR: --model is required\n"; return false; }
    return true;
}

/**
 * @brief Apply the model's chat template to (system, user) messages.
 * Uses llama_model_chat_template() + llama_chat_apply_template().
 */
static std::string format_chat(llama_model *model, const std::string &system, const std::string &user) {
    const char *tmpl = llama_model_chat_template(model, /*name*/nullptr);
    std::vector<llama_chat_message> msgs;
    msgs.push_back({ "system", system.c_str() });
    msgs.push_back({ "user",   user.c_str()   });

    // First call with add_assistant = true so the template asks model to answer
    int needed = llama_chat_apply_template(tmpl, msgs.data(), (int)msgs.size(),
                                           /*add_assistant*/ true, /*buf*/ nullptr, /*size*/ 0);
    if (needed < 0) {
        throw std::runtime_error("failed to apply chat template (sizing)");
    }
    std::string out;
    out.resize(needed);
    int written = llama_chat_apply_template(tmpl, msgs.data(), (int)msgs.size(),
                                            /*add_assistant*/ true, out.data(), (int)out.size());
    if (written < 0) {
        throw std::runtime_error("failed to apply chat template (format)");
    }
    return out;
}

/**
 * @brief Tokenize a string with model vocab.
 */
static std::vector<llama_token> tokenize(const llama_vocab *v, const std::string &s, bool add_bos = true) {
    const int n = -llama_tokenize(v, s.c_str(), (int)s.size(), nullptr, 0, add_bos, /*special*/ true);
    if (n <= 0) return {};
    std::vector<llama_token> ids(n);
    const int ret = llama_tokenize(v, s.c_str(), (int)s.size(), ids.data(), (int)ids.size(), add_bos, /*special*/ true);
    if (ret < 0) throw std::runtime_error("tokenize failed");
    return ids;
}

/**
 * @brief Stream-generate tokens and print them to stdout.
 * Uses llama_sampler_chain with typical samplers: top-k, top-p, min-p, temperature, RNG.
 */
static void generate_stream(llama_context *ctx, const llama_vocab *v, const std::vector<llama_token> &input,
                            const Options &opt) {
    // Submit the entire prompt as one batch:
    llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(input.data()), (int)input.size());

    // Sampler chain
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(opt.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(opt.top_p, /*min_keep*/1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(opt.min_p, /*min_keep*/1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(opt.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(opt.seed));

    int generated = 0;
    // Evaluate prompt (and continue decoding one token at a time)
    while (true) {
        // Check KV size to avoid overflow
        const int n_ctx      = llama_n_ctx(ctx);
        const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), /*seq_id*/0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            std::cerr << "\n[context exceeded; stopping]\n";
            break;
        }

        const int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            std::cerr << "llama_decode failed, ret=" << ret << "\n";
            break;
        }

        // Sample next token
        const llama_token tok = llama_sampler_sample(smpl, ctx, /*idx_last_token*/-1);
        if (llama_vocab_is_eog(v, tok)) break;

        // Convert token to text and stream it
        char buf[256];
        const int n = llama_token_to_piece(v, tok, buf, sizeof(buf), /*special*/ 0, /*lstrip*/ true);
        if (n < 0) break;
        std::string piece(buf, n);
        if (opt.color) std::cout << "\033[33m";
        std::cout << piece << std::flush;
        if (opt.color) std::cout << "\033[0m";

        // Feed the sampled token back
        llama_token t = tok;
        batch = llama_batch_get_one(&t, 1);

        if (++generated >= opt.n_predict) break;
    }

    llama_sampler_free(smpl);
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) { return 2; }

    if (!opt.prompt.has_value()) {
        std::string line;
        if (!std::getline(std::cin, line)) {
            std::cerr << "No prompt provided\n";
            return 2;
        }
        opt.prompt = line;
    }

    // Quiet log: only warnings/errors
    llama_log_set([](ggml_log_level level, const char *text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::fprintf(stderr, "%s", text);
    }, nullptr);

    // Load software backends (CPU)
    ggml_backend_load_all();

    // Model params: enforce CPU-only (n_gpu_layers = 0)
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;          // ensure CPU-only
    mparams.use_mmap     = true;       // mmap helps with large models
    mparams.use_mlock    = false;      // avoid mlock unless you know you want it

    llama_model *model = llama_model_load_from_file(opt.model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model: " << opt.model_path << "\n";
        return 1;
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t)opt.n_ctx;
    cparams.n_batch         = (uint32_t)opt.n_ctx;
    cparams.n_threads       = opt.threads;
    cparams.n_threads_batch = opt.threads;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create llama_context\n";
        llama_model_free(model);
        return 1;
    }

    // Format prompt via the model's chat template
    std::string formatted;
    try {
        formatted = format_chat(model, opt.system, *opt.prompt);
    } catch (const std::exception &e) {
        std::cerr << "Chat templating failed: " << e.what() << "\n";
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Tokenize and run
    auto toks = tokenize(vocab, formatted, /*add_bos*/ true);
    if (opt.color) std::cout << "\033[1;32m"; // green 'assistant' stream color
    generate_stream(ctx, vocab, toks, opt);
    if (opt.color) std::cout << "\033[0m";
    std::cout << std::endl;

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
