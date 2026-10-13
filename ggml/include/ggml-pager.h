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

        // ---- current configuration (run-time control, see ggml_pager_config) ----
        uint32_t enabled;               // 1 once a pager exists (the first MoE block of a decode ran)
        uint64_t generation;            // incremented by every applied reconfigure (and the first pool build)
        uint64_t generation_requested;  // incremented by every ggml_backend_cuda_pager_configure call
        double   rebuild_ms;            // duration of the last applied reconfigure (drain, free, allocate, preload)
        int64_t  pool_mib_requested;    // requested pool size (env GGML_CUDA_PAGER_MIB or the last configure), -1 none
        int64_t  pool_mib_obtained;     // pool + staging actually allocated, MiB (0: no pool)
        int32_t  policy;                // 0 lru, 1 lfu, 2 fwlru, 3 static
        int32_t  admit_misses;
        int32_t  admit_window;
        uint32_t n_slots;               // usable slots of the pool
        uint64_t slot_bytes;            // bytes of one slot (padded)
        uint32_t warm_queued;           // warm-set keys accepted by the last reconfigure
        uint32_t warm_skipped;          // ... and keys rejected (unknown layer or expert)
        uint32_t pinned_count;          // keys pinned by the last reconfigure
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

    // Run-time control of the pager. Every field is optional: -1 / NULL keep the current value. The new configuration is applied
    // at the next safe point: the start of the next pager block after its stream synchronisation (when a decode is running), or
    // ggml_backend_cuda_pager_apply() when the caller knows the compute stream is idle (e.g. between two llama_decode calls on
    // the thread that runs them). A pool size change drops the pool and builds a new one (every expert is a miss in between);
    // otherwise the pool is kept and the cache is rebuilt around the experts that are resident.
    struct ggml_pager_config {
        int32_t         pool_mib;     // pool + staging, MiB; 0: no pool; -1: keep
        int32_t         policy;       // 0 lru, 1 lfu, 2 fwlru, 3 static; -1: keep
        int32_t         admit_misses; // -1: keep
        int32_t         admit_window; // -1: keep
        const int32_t * warm;         // n_warm pairs (layer, expert), highest priority first; NULL: none
        int32_t         n_warm;
        const int32_t * pinned;       // n_pinned pairs (layer, expert), never evicted
        int32_t         n_pinned;
    };
    // "ggml_backend_cuda_pager_configure": thread safe; returns the generation this request will produce when applied
    typedef uint64_t (*ggml_backend_cuda_pager_configure_t)(const struct ggml_pager_config * cfg);
    // "ggml_backend_cuda_pager_apply": applies pending requests now (compute stream must be idle); returns the number of pagers
    // reconfigured. Without a pager yet, the request simply becomes the initial configuration.
    typedef int (*ggml_backend_cuda_pager_apply_t)(void);

#ifdef __cplusplus
}
#endif
