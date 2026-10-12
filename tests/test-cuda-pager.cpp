// Expert pager (CUDA backend, stage 1): checks the paged MUL_MAT_ID path against the CPU backend and the stock CUDA path.
//
// One tiny MoE FFN block (gate, up, swiglu, down; Q4_K / Q6_K experts) is evaluated
//   - on the CPU backend (reference, ggml_compute_forward_mul_mat_id),
//   - on the CUDA backend with weights in the regular CUDA buffer (stock kernels),
//   - on the CUDA backend with weights in the paged buffer: pool 0 (every expert is a CPU miss), a pool large enough for every
//     expert (second pass: all hits), and a pool of about half of the experts (a mix).
// Reports the max abs error of each against the references. Fails when
//   - the pool-0 result differs from the CPU reference by more than 1e-4 of the output's max magnitude,
//   - the all-hit result is not bit-identical to the stock CUDA result (same kernels on the same bytes),
//   - hits + misses != expert evaluations, or a mixed pool is further from the CPU reference than 3e-2.
// Needs a CUDA device; exits 0 with a message when there is none.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-pager.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int64_t K = 512, FF = 256, NE = 8, NU = 4;
static ggml_type TYPE_GU = GGML_TYPE_Q4_K, TYPE_D = GGML_TYPE_Q6_K;

struct result {
    std::vector<float> out;
    ggml_pager_stats   stats = {};
};

enum class variant { swiglu, geglu };

// builds the block in `ctx` from the given weight tensors (already allocated) and returns the output node
static ggml_tensor * build(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * wg, ggml_tensor * wu, ggml_tensor * wd, int T, variant v,
                           ggml_tensor ** x_out, ggml_tensor ** ids_out) {
    ggml_tensor * x        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * ids_full = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NE, T);
    ggml_set_input(x);
    ggml_set_input(ids_full);
    // like build_moe_ffn: a view of the full argsort, non-contiguous for T > 1
    ggml_tensor * ids = ggml_view_2d(ctx, ids_full, NU, T, ids_full->nb[1], 0);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, wu, x, ids);
    ggml_tensor * act  = v == variant::swiglu ? ggml_swiglu_split(ctx, gate, up) : ggml_geglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, wd, act, ids);
    ggml_set_output(down);
    ggml_build_forward_expand(gf, down);
    *x_out   = x;
    *ids_out = ids_full;
    return down;
}

static void make_weight_data(std::vector<std::vector<uint8_t>> & q, std::mt19937 & rng) {
    const ggml_type types[3] = { TYPE_GU, TYPE_GU, TYPE_D };
    const int64_t   rows[3]  = { FF, FF, K };
    const int64_t   cols[3]  = { K, K, FF };
    std::normal_distribution<float> nd(0.0f, 0.05f);
    q.resize(3);
    for (int k = 0; k < 3; k++) {
        std::vector<float> f((size_t) cols[k] * rows[k] * NE);
        for (float & e : f) {
            e = nd(rng);
        }
        q[k].resize(ggml_row_size(types[k], cols[k]) * rows[k] * NE);
        ggml_quantize_chunk(types[k], f.data(), q[k].data(), 0, rows[k] * NE, cols[k], nullptr);
    }
}

// runs the block on `backend` with the weights in `wbuft` (graph tensors in `cbuft`); runs `passes` times, returns the last
static bool run(ggml_backend_t backend, ggml_backend_buffer_type_t wbuft, ggml_backend_buffer_type_t cbuft,
                const std::vector<std::vector<uint8_t>> & q, int T, variant v, const std::vector<float> & x_data,
                const std::vector<int32_t> & ids_data, int passes, result & res) {
    const ggml_type types[3] = { TYPE_GU, TYPE_GU, TYPE_D };
    const int64_t   ne0[3]   = { K, K, FF };
    const int64_t   ne1[3]   = { FF, FF, K };

    ggml_init_params wp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w[3];
    for (int k = 0; k < 3; k++) {
        w[k] = ggml_new_tensor_3d(wctx, types[k], ne0[k], ne1[k], NE);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft);
    if (!wbuf) {
        fprintf(stderr, "weight allocation failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int k = 0; k < 3; k++) {
        ggml_backend_tensor_set(w[k], q[k].data(), 0, q[k].size());
    }

    ggml_init_params gp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * x, * ids;
    ggml_tensor * out = build(gctx, gf, w[0], w[1], w[2], T, v, &x, &ids);

    ggml_gallocr_t galloc = ggml_gallocr_new(cbuft);
    bool ok = ggml_gallocr_alloc_graph(galloc, gf);
    if (ok) {
        for (int p = 0; p < passes; p++) {
            ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));
            ok = ok && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        }
        res.out.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.out.data(), 0, ggml_nbytes(out));
    }
    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return ok;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

static double max_abs(const std::vector<float> & a) {
    double m = 0.0;
    for (float e : a) {
        m = std::max(m, (double) std::fabs(e));
    }
    return m;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (!cuda_dev || !cuda_reg) {
        printf("no CUDA device, skipping\n");
        return 0;
    }
    auto get_stats = (ggml_backend_cuda_pager_get_stats_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_pager_get_stats");
    auto set_kernels = (ggml_backend_cuda_pager_set_cpu_kernels_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_cuda_pager_set_cpu_kernels");
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cuda_reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_stats || !set_kernels || !get_extra) {
        fprintf(stderr, "FAIL: pager proc addresses missing\n");
        return 1;
    }
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    ggml_pager_cpu_kernels kernels = {};
    kernels.get_type_traits_cpu = ggml_get_type_traits_cpu;
    kernels.vec_swiglu_f32 = (ggml_vec_swiglu_f32_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_vec_swiglu_f32");
    if (!kernels.vec_swiglu_f32) {
        fprintf(stderr, "FAIL: CPU backend does not export ggml_backend_cpu_vec_swiglu_f32\n");
        return 1;
    }
    set_kernels(&kernels);

    ggml_backend_buffer_type_t paged = get_extra(cuda_dev)[0];
    if (!paged || strstr(ggml_backend_buft_name(paged), "_PAGED") == nullptr || !ggml_backend_buft_is_host(paged)) {
        fprintf(stderr, "FAIL: paged buffer type missing or not host\n");
        return 1;
    }
    printf("paged buffer type: %s\n", ggml_backend_buft_name(paged));

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t cuda_buft = ggml_backend_dev_buffer_type(cuda_dev);

    std::mt19937 rng(1234);
    struct config_t { int64_t k, ff, ne, nu; ggml_type gu, d; };
    // the first config is tiny; the others have the shapes of OLMoE-1B-7B (2048 x 1024, 64 experts, top 8) and Qwen3-30B-A3B
    const config_t configs[] = {
        { 512, 256, 8, 4, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K },
        { 2048, 1024, 64, 8, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K },
        { 2048, 1024, 64, 8, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K },
        { 2048, 768, 128, 8, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K },
    };
    int failures = 0;
    for (const config_t & cfg : configs) {
    K = cfg.k; FF = cfg.ff; NE = cfg.ne; NU = cfg.nu; TYPE_GU = cfg.gu; TYPE_D = cfg.d;
    printf("config K=%lld FF=%lld experts=%lld used=%lld gate/up %s down %s\n", (long long) K, (long long) FF, (long long) NE, (long long) NU, ggml_type_name(TYPE_GU), ggml_type_name(TYPE_D));
    std::vector<std::vector<uint8_t>> q;
    make_weight_data(q, rng);

    // expert bytes of one (gate, up, down) triplet and of all layers' experts (one layer here)
    const size_t triplet = q[0].size() / NE + q[1].size() / NE + q[2].size() / NE;
    const size_t mib = 1024 * 1024;
    // pools are sized in MiB by the environment variable; the test experts are small, so use the finest granularity available:
    // "all" = generous, "half" = about half of the triplets plus the null slot
    const long long mib_all  = (long long) ((triplet * (NE + 1) + mib - 1) / mib) + 1;
    printf("triplet bytes %zu, all-experts pool %lld MiB\n", triplet, mib_all);

    for (variant v : { variant::swiglu, variant::geglu }) {
        for (int T : { 1, 2, 3, 8 }) {
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> x_data((size_t) K * T);
            for (float & e : x_data) {
                e = nd(rng);
            }
            std::vector<int32_t> ids_data((size_t) NE * T);
            for (int t = 0; t < T; t++) {
                std::vector<int32_t> perm(NE);
                for (int i = 0; i < NE; i++) {
                    perm[i] = i;
                }
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int i = 0; i < NE; i++) {
                    ids_data[(size_t) t * NE + i] = perm[i];
                }
            }

            result ref_cpu, ref_cuda;
            if (!run(cpu, cpu_buft, cpu_buft, q, T, v, x_data, ids_data, 1, ref_cpu)) { return 1; }
            {
                ggml_backend_t b = ggml_backend_dev_init(cuda_dev, nullptr);
                if (!run(b, cuda_buft, cuda_buft, q, T, v, x_data, ids_data, 1, ref_cuda)) { return 1; }
                ggml_backend_free(b);
            }
            const double scale = max_abs(ref_cpu.out);
            printf("variant %s T=%d: |out|max %.3f, stock CUDA vs CPU max abs err %.3e\n", v == variant::swiglu ? "swiglu" : "geglu", T,
                   scale, max_abs_diff(ref_cuda.out, ref_cpu.out));

            struct mode_t { const char * name; long long mib; int passes; };
            for (mode_t m : { mode_t{ "pool 0 (all miss)", 0, 1 }, mode_t{ "pool all, 2nd pass (all hit)", mib_all, 2 }, mode_t{ "pool all, 1st pass (all miss, then fill)", mib_all, 1 },
                            mode_t{ "pool about half, 2nd pass (hits and misses)", mib_all / 2, 2 } }) {
                setenv("GGML_CUDA_PAGER_MIB", std::to_string(m.mib).c_str(), 1);
                ggml_pager_stats s0 = {}, s1 = {};
                get_stats(&s0);
                ggml_backend_t b = ggml_backend_dev_init(cuda_dev, nullptr);
                result r;
                if (!run(b, paged, cuda_buft, q, T, v, x_data, ids_data, m.passes, r)) { return 1; }
                ggml_backend_free(b);
                get_stats(&s1);
                const uint64_t hits = s1.hits - s0.hits, misses = s1.misses - s0.misses;
                const bool fused = v == variant::swiglu;
                // the fused block counts each (token, used expert) once; the single-node path counts it once per node (3)
                const uint64_t per = fused ? 1 : 3;
                const uint64_t evals = per * NU * T * m.passes;
                const double e_cpu  = max_abs_diff(r.out, ref_cpu.out);
                const double e_cuda = max_abs_diff(r.out, ref_cuda.out);
                printf("  %-42s hits %3llu misses %3llu | err vs CPU %.3e (%.1e rel), vs stock CUDA %.3e | %s\n", m.name,
                       (unsigned long long) hits, (unsigned long long) misses, e_cpu, e_cpu / scale, e_cuda, fused ? "fused block" : "single nodes");
                bool bad = false;
                if (hits + misses != evals) {
                    bad = true;
                    fprintf(stderr, "    FAIL: hits + misses = %llu, expected %llu\n", (unsigned long long) (hits + misses), (unsigned long long) evals);
                }
                // the geglu variant runs gelu on the GPU for the single-node path while the CPU reference uses its f16 gelu table
                if (m.mib == 0 && !(e_cpu <= (fused ? 1e-4 : 1e-2) * scale)) { bad = true; fprintf(stderr, "    FAIL: all-miss differs from the CPU reference\n"); }
                if (m.passes == 2 && m.mib == mib_all && fused && e_cuda != 0.0) { bad = true; fprintf(stderr, "    FAIL: all-hit not bit-identical to stock CUDA\n"); }
                if (m.passes == 2 && m.mib == mib_all && hits != per * NU * T) { bad = true; fprintf(stderr, "    FAIL: second pass was not all hits\n"); }
                if (e_cpu > 3e-2 * scale) { bad = true; fprintf(stderr, "    FAIL: error vs CPU too large\n"); }
                failures += bad;
            }
        }
    }

    }

    ggml_backend_free(cpu);
    printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
