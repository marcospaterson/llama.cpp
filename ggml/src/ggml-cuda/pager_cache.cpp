// C++20 translation unit: adapts the vendored eq::pager::SlotCache to the C++17 facade in pager.cuh.

#include "pager_cache.h"
#include "eq/pager/slot_cache.hpp"

#include <algorithm>

struct ggml_cuda_pager_cache::impl {
    eq::pager::SlotCache cache;

    static eq::pager::CacheConfig make(const ggml_cuda_pager_cache_config & c) {
        eq::pager::CacheConfig cfg;
        cfg.n_layers          = c.n_layers;
        cfg.experts_per_layer = c.experts_per_layer;
        cfg.expert_bytes.assign((size_t) c.n_layers * c.experts_per_layer, c.bytes_per_expert);
        cfg.budget_bytes      = c.budget_bytes;
        cfg.policy.kind = c.policy == 1 ? eq::pager::PolicyKind::kLfu : c.policy == 2 ? eq::pager::PolicyKind::kFwLru : eq::pager::PolicyKind::kLru;
        cfg.policy.half_life_tokens = c.half_life_tokens;
        cfg.promotion.admit_min_misses   = c.admit_min_misses;
        cfg.promotion.admit_window_tokens = c.admit_window_tokens;
        cfg.promotion.max_bytes_per_token = c.max_bytes_per_token;
        cfg.promotion.staging_bytes       = c.staging_bytes;
        return cfg;
    }
    explicit impl(const ggml_cuda_pager_cache_config & c) : cache(make(c)) {}
};

ggml_cuda_pager_cache::ggml_cuda_pager_cache(const ggml_cuda_pager_cache_config & cfg) : p_(new impl(cfg)) {}
ggml_cuda_pager_cache::~ggml_cuda_pager_cache() { delete p_; }

uint32_t ggml_cuda_pager_cache::max_slots() const { return p_->cache.max_slots(0); }

int ggml_cuda_pager_cache::lookup(uint32_t key) const {
    const uint32_t s = p_->cache.lookup(key);
    return s == eq::pager::kNoSlot ? -1 : (int) s;
}

bool ggml_cuda_pager_cache::touch(uint32_t key, uint64_t token) { return p_->cache.touch(key, token); }
bool ggml_cuda_pager_cache::on_miss(uint32_t key, uint64_t token) { return p_->cache.on_miss(key, token); }
bool ggml_cuda_pager_cache::install(uint32_t key) { return p_->cache.install(key); }

size_t ggml_cuda_pager_cache::next_promotions(uint64_t budget_bytes, ggml_cuda_pager_promotion * out, size_t cap) {
    eq::pager::Promotion tmp[64];
    cap = std::min<size_t>(cap, 64);
    const size_t n = p_->cache.next_promotions(budget_bytes, std::span<eq::pager::Promotion>(tmp, cap));
    for (size_t i = 0; i < n; i++) {
        out[i].id = tmp[i].id;
        out[i].key = tmp[i].key;
        out[i].dest_slot = tmp[i].dest_slot;
        out[i].n_victims = tmp[i].n_victims;
        for (uint32_t v = 0; v < tmp[i].n_victims && v < 8; v++) {
            out[i].victim_keys[v]  = tmp[i].victims[v];
            out[i].victim_slots[v] = tmp[i].victim_slots[v];
        }
    }
    return n;
}

ggml_cuda_pager_cache::commit_status ggml_cuda_pager_cache::commit(uint64_t id, uint32_t & slot) {
    const eq::pager::CommitResult r = p_->cache.commit(id);
    slot = r.slot;
    switch (r.status) {
        case eq::pager::CommitStatus::kCommitted: return committed;
        case eq::pager::CommitStatus::kAborted:   return aborted;
        default:                                  return stale;
    }
}

ggml_cuda_pager_cache_counters ggml_cuda_pager_cache::counters() const {
    const eq::pager::Snapshot s = p_->cache.snapshot();
    return { s.total.issued, s.total.promotions, s.total.aborted, s.total.evictions, s.total.admission_rejected,
             s.total.victims_reclaimed, s.resident_count };
}

bool ggml_cuda_pager_cache::verify() const { return p_->cache.verify_invariants(); }
