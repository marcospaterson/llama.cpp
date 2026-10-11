// llama-eq-trace: record the router decision of every MoE layer for every prompt token.
// See README.md for the CLI and the output format.

#include "arg.h"
#include "common.h"
#include "eq-trace-format.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <nlohmann/json.hpp>
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// output files
// ---------------------------------------------------------------------------

namespace {

struct file_closer {
    void operator()(FILE * f) const {
        if (f) {
            std::fclose(f);
        }
    }
};

// writes the 32 byte header up front, appends records, patches the header on close()
class trace_file {
  public:
    trace_file(const std::string & path, const char * magic) : path_(path) {
        file_.reset(std::fopen(path.c_str(), "wb"));
        if (!file_) {
            throw std::runtime_error("cannot open '" + path + "' for writing");
        }
        std::memcpy(hdr_.magic, magic, 4);
        write_header();
    }

    void write(const void * data, size_t n) {
        if (n && std::fwrite(data, 1, n, file_.get()) != n) {
            throw std::runtime_error("write failed on '" + path_ + "'");
        }
    }

    // header fields other than magic/version/n_records may be set until close()
    eq_trace::header & hdr() { return hdr_; }
    const std::string & path() const { return path_; }

    void close() {
        if (!file_) {
            return;
        }
        if (std::fseek(file_.get(), 0, SEEK_SET) != 0) {
            throw std::runtime_error("seek failed on '" + path_ + "'");
        }
        write_header();
        FILE * f = file_.release();
        if (std::fflush(f) != 0 || std::fclose(f) != 0) {
            throw std::runtime_error("close failed on '" + path_ + "'");
        }
    }

  private:
    void write_header() {
        uint8_t buf[eq_trace::HEADER_SIZE];
        eq_trace::encode_header(hdr_, buf);
        write(buf, sizeof(buf));
    }

    std::string                         path_;
    std::unique_ptr<FILE, file_closer>  file_;
    eq_trace::header                    hdr_;
};

// ---------------------------------------------------------------------------
// tracer: eval callback state
// ---------------------------------------------------------------------------

enum tensor_kind { KIND_NONE, KIND_LOGITS, KIND_TOPK, KIND_HIDDEN };

constexpr uint8_t FLAG_LOGITS = 1;
constexpr uint8_t FLAG_TOPK   = 2;
constexpr uint8_t FLAG_HIDDEN = 4;

// Graph tensors (src/llama-graph.cpp build_moe_ffn, src/models/{llama,olmoe}.cpp):
//   ffn_moe_logits-<il>  router logits  f32 [n_expert, n_tokens]  (gate_inp * ffn_norm, before any bias/mask)
//   ffn_moe_topk-<il>    selected ids   i32 [n_expert_used, n_tokens]
//   ffn_norm-<il>        router input   f32 [n_embd, n_tokens]  (RMS-normed ffn_inp, passed to build_moe_ffn)
const char * const NAME_LOGITS = "ffn_moe_logits-";
const char * const NAME_TOPK   = "ffn_moe_topk-";
const char * const NAME_HIDDEN = "ffn_norm-";

bool parse_layer(const char * s, int & il) {
    if (!*s) {
        return false;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (*end != '\0' || v < 0) {
        return false;
    }
    il = (int) v;
    return true;
}

tensor_kind classify(const char * name, int & il) {
    struct { const char * prefix; tensor_kind kind; } const table[] = {
        { NAME_LOGITS, KIND_LOGITS }, { NAME_TOPK, KIND_TOPK }, { NAME_HIDDEN, KIND_HIDDEN },
    };
    for (const auto & e : table) {
        const size_t n = std::strlen(e.prefix);
        if (std::strncmp(name, e.prefix, n) == 0 && parse_layer(name + n, il)) {
            return e.kind;
        }
    }
    return KIND_NONE;
}

class tracer {
  public:
    tracer(uint32_t n_block, uint32_t n_embd, double hidden_fraction, uint64_t seed) :
        n_block_(n_block), n_embd_(n_embd), fraction_(hidden_fraction), seed_(seed) {}

    static bool callback(struct ggml_tensor * t, bool ask, void * user_data) {
        return static_cast<tracer *>(user_data)->on_tensor(t, ask);
    }

    void begin_prompt(uint32_t prompt_id, uint32_t n_tokens) {
        prompt_id_ = prompt_id;
        n_tokens_  = n_tokens;
        const size_t slots = (size_t) n_tokens * n_block_;
        flags_.assign(slots, 0);
        seen_[0].assign(n_block_, 0);
        seen_[1].assign(n_block_, 0);
        seen_[2].assign(n_block_, 0);
        ids_.resize(top_k_ ? slots * top_k_ : 0);
        logits_.resize(n_expert_ ? slots * n_expert_ : 0);
        if (fraction_ > 0.0) {
            hidden_.resize(slots * n_embd_);
        }
    }

    const std::string & error() const { return error_; }
    uint32_t n_expert() const { return n_expert_; }
    uint32_t top_k() const { return top_k_; }

    // Check the prompt is complete and return the sorted list of MoE layers.
    std::vector<uint32_t> finish_prompt(const std::vector<uint32_t> * expected_layers) {
        std::vector<uint32_t> layers;
        for (uint32_t il = 0; il < n_block_; il++) {
            bool any = false;
            for (int k = 0; k < 2; k++) {
                any = any || seen_[k][il] > 0;
            }
            if (any) {
                layers.push_back(il);
            }
        }
        if (layers.empty()) {
            throw std::runtime_error("no ffn_moe_logits/ffn_moe_topk tensors were observed for prompt " +
                                     std::to_string(prompt_id_) + ": model has no MoE layers or the graph names changed");
        }
        if (expected_layers && *expected_layers != layers) {
            throw std::runtime_error("prompt " + std::to_string(prompt_id_) +
                                     ": the set of MoE layers differs from the first prompt");
        }
        for (uint32_t il : layers) {
            for (int k = 0; k < 2; k++) {
                if (seen_[k][il] != n_tokens_) {
                    throw std::runtime_error(std::string("prompt ") + std::to_string(prompt_id_) + " layer " +
                                             std::to_string(il) + ": expected tensor " + (k == 0 ? NAME_LOGITS : NAME_TOPK) +
                                             std::to_string(il) + " covered " + std::to_string(seen_[k][il]) + " of " +
                                             std::to_string(n_tokens_) + " tokens");
                }
            }
            if (fraction_ > 0.0) {
                for (uint32_t pos = 0; pos < n_tokens_; pos++) {
                    if (eq_trace::token_sampled(seed_, prompt_id_, pos, fraction_) &&
                        !(flags_[slot(pos, il)] & FLAG_HIDDEN)) {
                        throw std::runtime_error(std::string("prompt ") + std::to_string(prompt_id_) + " layer " +
                                                 std::to_string(il) + " token " + std::to_string(pos) +
                                                 ": expected tensor " + NAME_HIDDEN + std::to_string(il) + " never showed up");
                    }
                }
            }
        }
        return layers;
    }

    // Encode all records of the finished prompt, ordered by (token_pos, layer).
    void encode(const std::vector<uint32_t> & layers, std::vector<uint8_t> & routes, std::vector<uint8_t> & hidden,
                uint64_t & n_routes, uint64_t & n_hidden) const {
        const size_t rsz = eq_trace::route_record_size(top_k_, n_expert_);
        const size_t hsz = eq_trace::hidden_record_size(n_embd_);
        routes.clear();
        hidden.clear();
        for (uint32_t pos = 0; pos < n_tokens_; pos++) {
            const bool keep = fraction_ > 0.0 && eq_trace::token_sampled(seed_, prompt_id_, pos, fraction_);
            for (uint32_t il : layers) {
                const size_t s = slot(pos, il);
                size_t off = routes.size();
                routes.resize(off + rsz);
                eq_trace::encode_route_record(routes.data() + off, prompt_id_, pos, (uint16_t) il, &ids_[s * top_k_],
                                              top_k_, &logits_[s * n_expert_], n_expert_);
                n_routes++;
                if (keep) {
                    off = hidden.size();
                    hidden.resize(off + hsz);
                    eq_trace::encode_hidden_record(hidden.data() + off, prompt_id_, pos, (uint16_t) il,
                                                   &hidden_[s * n_embd_], n_embd_);
                    n_hidden++;
                }
            }
        }
    }

  private:
    size_t slot(uint32_t pos, uint32_t il) const { return (size_t) pos * n_block_ + il; }

    bool fail(const std::string & msg) {
        if (error_.empty()) {
            error_ = msg;
        }
        return false; // cancels the graph compute
    }

    // tensor bytes on the host, copied from the device when needed
    const uint8_t * fetch(const struct ggml_tensor * t, std::vector<uint8_t> & tmp) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            return static_cast<const uint8_t *>(t->data);
        }
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        return tmp.data();
    }

    bool on_tensor(struct ggml_tensor * t, bool ask) {
        int il = -1;
        const tensor_kind kind = classify(t->name, il);
        if (kind == KIND_NONE) {
            return ask ? false : true;
        }
        if (ask) {
            return kind != KIND_HIDDEN || fraction_ > 0.0;
        }
        try {
            return collect(t, kind, (uint32_t) il);
        } catch (const std::exception & e) {
            return fail(e.what());
        }
    }

    bool collect(struct ggml_tensor * t, tensor_kind kind, uint32_t il) {
        if (il >= n_block_) {
            return fail(std::string("tensor ") + t->name + ": layer index out of range");
        }
        if (n_tokens_ == 0) {
            return fail(std::string("tensor ") + t->name + " seen outside of a prompt");
        }
        const uint32_t nt = (uint32_t) t->ne[1];
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            return fail(std::string("tensor ") + t->name + ": unexpected shape");
        }
        const int k = kind == KIND_LOGITS ? 0 : kind == KIND_TOPK ? 1 : 2;
        const uint32_t base = seen_[k][il];
        if (base + nt > n_tokens_) {
            return fail(std::string("tensor ") + t->name + ": more tokens than the prompt has");
        }

        const uint32_t sz   = (uint32_t) t->ne[0];
        const size_t   slots = (size_t) n_tokens_ * n_block_;
        if (kind == KIND_LOGITS) {
            if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) {
                return fail(std::string("tensor ") + t->name + ": expected contiguous-row f32");
            }
            if (n_expert_ == 0) {
                n_expert_ = sz;
                logits_.resize(slots * n_expert_);
            } else if (n_expert_ != sz) {
                return fail(std::string("tensor ") + t->name + ": n_expert differs between layers");
            }
            const uint8_t * data = fetch(t, tmp_);
            for (uint32_t r = 0; r < nt; r++) {
                const size_t s = slot(base + r, il);
                std::memcpy(&logits_[s * n_expert_], data + r * t->nb[1], sizeof(float) * n_expert_);
                flags_[s] |= FLAG_LOGITS;
            }
        } else if (kind == KIND_TOPK) {
            if (t->type != GGML_TYPE_I32 || t->nb[0] != sizeof(int32_t)) {
                return fail(std::string("tensor ") + t->name + ": expected i32");
            }
            if (top_k_ == 0) {
                top_k_ = sz;
                ids_.resize(slots * top_k_);
            } else if (top_k_ != sz) {
                return fail(std::string("tensor ") + t->name + ": n_expert_used differs between layers");
            }
            const uint8_t * data = fetch(t, tmp_);
            for (uint32_t r = 0; r < nt; r++) {
                const size_t s = slot(base + r, il);
                for (uint32_t j = 0; j < top_k_; j++) {
                    int32_t v;
                    std::memcpy(&v, data + r * t->nb[1] + j * sizeof(int32_t), sizeof(v));
                    if (v < 0 || v > 0xffff) {
                        return fail(std::string("tensor ") + t->name + ": expert id out of range");
                    }
                    ids_[s * top_k_ + j] = (uint16_t) v;
                }
                flags_[s] |= FLAG_TOPK;
            }
        } else {
            if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) {
                return fail(std::string("tensor ") + t->name + ": expected contiguous-row f32");
            }
            if (sz != n_embd_) {
                return fail(std::string("tensor ") + t->name + ": width differs from n_embd");
            }
            bool any = false;
            for (uint32_t r = 0; r < nt && !any; r++) {
                any = eq_trace::token_sampled(seed_, prompt_id_, base + r, fraction_);
            }
            if (any) {
                const uint8_t * data = fetch(t, tmp_);
                for (uint32_t r = 0; r < nt; r++) {
                    if (!eq_trace::token_sampled(seed_, prompt_id_, base + r, fraction_)) {
                        continue;
                    }
                    const size_t s = slot(base + r, il);
                    for (uint32_t i = 0; i < n_embd_; i++) {
                        float f;
                        std::memcpy(&f, data + r * t->nb[1] + i * sizeof(float), sizeof(f));
                        hidden_[s * n_embd_ + i] = eq_trace::f32_to_f16(f);
                    }
                    flags_[s] |= FLAG_HIDDEN;
                }
            }
        }
        seen_[k][il] = base + nt;
        return true;
    }

    const uint32_t n_block_;
    const uint32_t n_embd_;
    const double   fraction_;
    const uint64_t seed_;

    uint32_t n_expert_ = 0;
    uint32_t top_k_    = 0;

    uint32_t prompt_id_ = 0;
    uint32_t n_tokens_  = 0;

    std::vector<uint8_t>  flags_;
    std::vector<uint32_t> seen_[3]; // tokens seen per layer: logits, topk, hidden
    std::vector<uint16_t> ids_;
    std::vector<float>    logits_;
    std::vector<uint16_t> hidden_;
    std::vector<uint8_t>  tmp_;
    std::string           error_;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct eq_args {
    std::string prompts_path;
    std::string out_prefix;
    int         max_tokens    = 512;
    double      hidden_sample = 0.1;
    uint64_t    seed          = 0;
};

[[noreturn]] void usage_error(const std::string & msg) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: llama-eq-trace -m model.gguf --prompts prompts.jsonl --out PREFIX\n"
                 "                      [--max-tokens N] [--hidden-sample F] [--seed S] [common llama.cpp args]\n",
                 msg.c_str());
    std::exit(2);
}

// Extract our own options from argv, return the remaining args for common_params_parse.
std::vector<char *> split_args(int argc, char ** argv, eq_args & a) {
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        std::string val;
        bool has_val = false;
        const size_t eqp = arg.find('=');
        if (arg.rfind("--", 0) == 0 && eqp != std::string::npos) {
            val     = arg.substr(eqp + 1);
            arg     = arg.substr(0, eqp);
            has_val = true;
        }
        const bool ours = arg == "--prompts" || arg == "--out" || arg == "--max-tokens" || arg == "--hidden-sample" ||
                          arg == "--seed";
        if (!ours) {
            rest.push_back(argv[i]);
            continue;
        }
        if (!has_val) {
            if (i + 1 >= argc) {
                usage_error(arg + " needs a value");
            }
            val = argv[++i];
        }
        try {
            size_t used = 0;
            if (arg == "--prompts") {
                a.prompts_path = val;
            } else if (arg == "--out") {
                a.out_prefix = val;
            } else if (arg == "--max-tokens") {
                a.max_tokens = std::stoi(val, &used);
                if (used != val.size() || a.max_tokens <= 0) {
                    throw std::invalid_argument(val);
                }
            } else if (arg == "--hidden-sample") {
                a.hidden_sample = std::stod(val, &used);
                if (used != val.size() || !(a.hidden_sample >= 0.0 && a.hidden_sample <= 1.0)) {
                    throw std::invalid_argument(val);
                }
            } else {
                a.seed = std::stoull(val, &used);
                if (used != val.size() || val.find('-') != std::string::npos) {
                    throw std::invalid_argument(val);
                }
            }
        } catch (const std::exception &) {
            usage_error("invalid value '" + val + "' for " + arg);
        }
    }
    return rest;
}

struct prompt {
    uint32_t    id;
    std::string text;
};

std::vector<prompt> read_prompts(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open prompts file '" + path + "'");
    }
    std::vector<prompt>           out;
    std::unordered_set<uint32_t>  seen;
    std::string                   line;
    size_t                        lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        const std::string where = path + ":" + std::to_string(lineno) + ": ";
        json j;
        try {
            j = json::parse(line);
        } catch (const std::exception & e) {
            throw std::runtime_error(where + "malformed JSON: " + e.what());
        }
        if (!j.is_object()) {
            throw std::runtime_error(where + "expected a JSON object");
        }
        if (!j.contains("prompt_id") || !j["prompt_id"].is_number_unsigned() ||
            j["prompt_id"].get<uint64_t>() > UINT32_MAX) {
            throw std::runtime_error(where + "'prompt_id' must be an unsigned 32 bit integer");
        }
        if (!j.contains("text") || !j["text"].is_string()) {
            throw std::runtime_error(where + "'text' must be a string");
        }
        prompt p;
        p.id   = j["prompt_id"].get<uint32_t>();
        p.text = j["text"].get<std::string>();
        if (!seen.insert(p.id).second) {
            throw std::runtime_error(where + "duplicate prompt_id " + std::to_string(p.id));
        }
        out.push_back(std::move(p));
    }
    if (out.empty()) {
        throw std::runtime_error("no prompts in '" + path + "'");
    }
    return out;
}

int run(int argc, char ** argv) {
    eq_args a;
    std::vector<char *> rest = split_args(argc, argv, a);
    if (a.prompts_path.empty()) {
        usage_error("--prompts is required");
    }
    if (a.out_prefix.empty()) {
        usage_error("--out is required");
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }

    const std::vector<prompt> prompts = read_prompts(a.prompts_path);

    llama_backend_init();
    llama_numa_init(params.numa);

    params.n_ctx      = a.max_tokens;
    params.n_parallel = 1;
    params.warmup     = false;

    // the tracer needs model dims, which are only known after loading: the callback goes through a holder
    // that is filled in once the model is loaded
    struct tracer_holder {
        tracer * t = nullptr;
    } holder;
    params.cb_eval = [](struct ggml_tensor * t, bool ask, void * ud) -> bool {
        auto * h = static_cast<tracer_holder *>(ud);
        return h->t ? tracer::callback(t, ask, h->t) : (ask ? false : true);
    };
    params.cb_eval_user_data = &holder;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        throw std::runtime_error("failed to load the model or create the context");
    }

    // the model must have experts
    {
        char arch[128] = {0};
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
            throw std::runtime_error("model has no general.architecture key");
        }
        char cnt[32] = {0};
        const std::string key = std::string(arch) + ".expert_count";
        if (llama_model_meta_val_str(model, key.c_str(), cnt, sizeof(cnt)) < 0 || std::atoi(cnt) <= 0) {
            throw std::runtime_error(std::string("model (arch '") + arch + "') has no experts (" + key + " missing or 0)");
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_block    = (uint32_t) llama_model_n_layer(model);
    const uint32_t n_embd     = (uint32_t) llama_model_n_embd(model);
    const int32_t  n_batch    = (int32_t) llama_n_batch(ctx);

    tracer tr(n_block, n_embd, a.hidden_sample, a.seed);
    holder.t = &tr;

    trace_file routes(a.out_prefix + ".routes.bin", "EQTR");
    trace_file hidden(a.out_prefix + ".hidden.bin", "EQTH");

    std::vector<uint32_t> moe_layers;
    bool have_layers = false;
    uint64_t n_prompts = 0, n_tokens_total = 0, n_routes = 0, n_hidden = 0;
    double decode_s = 0.0;
    std::vector<uint8_t> routes_buf, hidden_buf;

    for (const prompt & p : prompts) {
        std::vector<llama_token> tokens = common_tokenize(vocab, p.text, true, false);
        if ((int) tokens.size() > a.max_tokens) {
            tokens.resize(a.max_tokens);
        }
        if (tokens.empty()) {
            throw std::runtime_error("prompt " + std::to_string(p.id) + " tokenizes to zero tokens");
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        tr.begin_prompt(p.id, (uint32_t) tokens.size());

        // all positions need outputs, otherwise the last layer is pruned to the output rows
        common_batch batch(ctx);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.add(tokens[i], (llama_pos) i, 0, true);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int32_t off = 0; off < batch.size(); off += n_batch) {
            const int32_t n = std::min(n_batch, batch.size() - off);
            const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(off, n));
            if (!tr.error().empty()) {
                throw std::runtime_error("prompt " + std::to_string(p.id) + ": " + tr.error());
            }
            if (rc != 0) {
                throw std::runtime_error("prompt " + std::to_string(p.id) + ": llama_process failed with " + std::to_string(rc));
            }
        }
        llama_synchronize(ctx);
        decode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const std::vector<uint32_t> layers = tr.finish_prompt(have_layers ? &moe_layers : nullptr);
        if (!have_layers) {
            moe_layers  = layers;
            have_layers = true;
            for (trace_file * f : { &routes, &hidden }) {
                f->hdr().n_layer  = (uint32_t) moe_layers.size();
                f->hdr().n_expert = tr.n_expert();
                f->hdr().top_k    = tr.top_k();
                f->hdr().n_embd   = n_embd;
            }
        }

        tr.encode(moe_layers, routes_buf, hidden_buf, n_routes, n_hidden);
        routes.write(routes_buf.data(), routes_buf.size());
        hidden.write(hidden_buf.data(), hidden_buf.size());

        n_prompts++;
        n_tokens_total += tokens.size();
    }

    routes.hdr().n_records = n_routes;
    hidden.hdr().n_records = n_hidden;
    routes.close();
    hidden.close();

    std::fprintf(stderr,
                 "eq-trace: prompts=%llu tokens=%llu moe_layers=%zu routes_records=%llu hidden_records=%llu "
                 "decode_time=%.2fs tokens/s=%.1f\n",
                 (unsigned long long) n_prompts, (unsigned long long) n_tokens_total, moe_layers.size(),
                 (unsigned long long) n_routes, (unsigned long long) n_hidden, decode_s,
                 decode_s > 0 ? (double) n_tokens_total / decode_s : 0.0);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-eq-trace: error: %s\n", e.what());
        return 1;
    }
}
