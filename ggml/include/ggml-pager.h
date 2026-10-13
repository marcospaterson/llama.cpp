#pragma once

// Expert pager telemetry and CPU kernel injection (CUDA backend, see ggml/src/ggml-cuda/pager.cuh).
// Reached through ggml_backend_reg_get_proc_address() so that users need no compile-time dependency on CUDA.

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    // process-wide totals over all live pagers
    struct ggml_pager_stats {
        uint64_t hits;              // (token, expert) evaluations served from the VRAM pool
        uint64_t misses;            // (token, expert) evaluations served by the CPU
        uint64_t resident_experts;  // experts currently resident (gauge)
        uint64_t pool_bytes;        // VRAM allocated for pools (gauge)
        uint64_t bytes_copied;      // host to VRAM bytes copied to fill slots
        double   cpu_ms;            // wall time of the CPU miss chain (submit to join), summed
        double   sync_ms;           // time the compute thread spent in the pager's stream synchronisation (readback of ids and input)
        double   host_ms;           // compute-thread time inside a block after the synchronisation (lookups, uploads, launches, CPU join)
        uint64_t syncs;             // stream synchronisations performed by the pager
        uint64_t blocks;            // pager-executed blocks (fused or single MUL_MAT_ID)
        uint64_t promotions_issued;     // background promotions handed to the copier
        uint64_t promotions_committed;  // ... whose copy finished and that became resident
        uint64_t promotions_aborted;    // ... committed as aborted (victims reclaimed, no replacement): slot released, never readable
        uint64_t evictions;             // residents evicted by a commit
        uint64_t admission_rejected;    // misses held back by the admission filter
        uint64_t queue_depth;           // promotions in flight (gauge)
        double   copy_gbps;             // achieved copier bandwidth: bytes copied / copier busy time
        double   copy_busy_ms;          // copier busy time
        double   promo_latency_ms_sum;      // sum over promotions (committed or aborted) of the time from issue to commit
        uint64_t promo_latency_tokens_sum;  // ... and of the tokens elapsed (divide by committed + aborted)
        double   bandwidth_gbps;        // host to device bandwidth measured at pool creation (or GGML_CUDA_PAGER_GBPS)
    };

    // "ggml_backend_cuda_pager_get_stats"
    typedef void (*ggml_backend_cuda_pager_get_stats_t)(struct ggml_pager_stats * out);

    // CPU kernels the pager runs for missed experts. The CUDA backend does not link ggml-cpu: the application fetches them
    // from the CPU backend and hands them over once, through "ggml_backend_cuda_pager_set_cpu_kernels".
    typedef const struct ggml_type_traits_cpu * (*ggml_get_type_traits_cpu_t)(enum ggml_type type);
    typedef void (*ggml_vec_swiglu_f32_t)(int n, float * y, const float * x, const float * g); // y = silu(x) * g
    struct ggml_pager_cpu_kernels {
        ggml_get_type_traits_cpu_t get_type_traits_cpu; // "ggml_get_type_traits_cpu" (ggml-cpu.h)
        ggml_vec_swiglu_f32_t      vec_swiglu_f32;      // CPU backend proc address "ggml_backend_cpu_vec_swiglu_f32"
    };
    typedef void (*ggml_backend_cuda_pager_set_cpu_kernels_t)(const struct ggml_pager_cpu_kernels * kernels);

#ifdef __cplusplus
}
#endif
