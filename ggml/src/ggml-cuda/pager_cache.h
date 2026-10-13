#pragma once

// Facade over the vendored slot cache for the expert pager (no CUDA, no C++20 in this header).

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------------------------------------------------
// Policy cache: expert key (layer * experts_per_layer + expert) -> slot. Host only, no CUDA. This is a C++17 facade over the
// vendored eq::pager::SlotCache (eq-pager/, C++20, compiled separately in pager_cache.cpp), so no C++20 header leaks here.
//
// All experts have one slot size (the sum of the three members' strides), so the cache has a single size class: slot ids are
// 0 .. max_slots()-1 and the device offset of slot s is pool_k + (s + 1) * stride_k (pool slot 0 is the null slot).
// Threading: the compute thread only (the cache has no locks); the copier thread never touches it.
// ---------------------------------------------------------------------------------------------------------------------

struct ggml_cuda_pager_cache_config {
    uint32_t n_layers = 0;
    uint32_t experts_per_layer = 0;
    uint32_t bytes_per_expert = 0;   // slot size, equal for every key
    uint64_t budget_bytes = 0;       // resident bytes
    uint64_t staging_bytes = 0;      // extra bytes for destination slots of in-flight copies
    int      policy = 0;             // 0 lru, 1 lfu, 2 fwlru
    double   half_life_tokens = 64.0;
    uint32_t admit_min_misses = 2;
    uint64_t admit_window_tokens = 16;
    uint64_t max_bytes_per_token = 0;
};

struct ggml_cuda_pager_promotion {
    uint64_t id;
    uint32_t key;
    uint32_t dest_slot;
    uint32_t n_victims;
    uint32_t victim_keys[8];
    uint32_t victim_slots[8];
};

struct ggml_cuda_pager_cache_counters {
    uint64_t issued, promotions, aborted, evictions, admission_rejected, victims_reclaimed;
    uint32_t resident;
};

class ggml_cuda_pager_cache {
public:
    enum commit_status { committed, aborted, stale };

    explicit ggml_cuda_pager_cache(const ggml_cuda_pager_cache_config & cfg);
    ~ggml_cuda_pager_cache();

    uint32_t max_slots() const;
    // slot (>= 0) of a resident key, -1 on a miss; a key is resident only after commit() (or install())
    int  lookup(uint32_t key) const;
    bool touch(uint32_t key, uint64_t token);
    bool on_miss(uint32_t key, uint64_t token);
    bool install(uint32_t key);   // resident now if it fits without eviction (static fill, prefill)
    size_t next_promotions(uint64_t budget_bytes, ggml_cuda_pager_promotion * out, size_t cap);
    commit_status commit(uint64_t id, uint32_t & slot);
    ggml_cuda_pager_cache_counters counters() const;
    bool verify() const;

private:
    struct impl;
    impl * p_;
};

