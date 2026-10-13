// vendored from elastic-quant cpp/pager at commit 0bd8cbe; do not edit here
#include "eq/pager/slot_cache.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace eq::pager {

namespace {
constexpr std::uint64_t kNever = std::numeric_limits<std::uint64_t>::max();
constexpr double kInf = std::numeric_limits<double>::infinity();
}  // namespace

// ---------------------------------------------------------------- indexed heap

template <class Before>
void SlotCache::Heap<Before>::sift_up(std::uint32_t i) noexcept {
    const ExpertKey k = heap_[i];
    while (i > 0) {
        const std::uint32_t parent = (i - 1) / 2;
        if (!before_(k, heap_[parent])) {
            break;
        }
        heap_[i] = heap_[parent];
        pos_[heap_[i]] = i;
        i = parent;
    }
    heap_[i] = k;
    pos_[k] = i;
}

template <class Before>
void SlotCache::Heap<Before>::sift_down(std::uint32_t i) noexcept {
    const ExpertKey k = heap_[i];
    const auto n = static_cast<std::uint32_t>(heap_.size());
    for (;;) {
        std::uint32_t child = 2 * i + 1;
        if (child >= n) {
            break;
        }
        if (child + 1 < n && before_(heap_[child + 1], heap_[child])) {
            ++child;
        }
        if (!before_(heap_[child], k)) {
            break;
        }
        heap_[i] = heap_[child];
        pos_[heap_[i]] = i;
        i = child;
    }
    heap_[i] = k;
    pos_[k] = i;
}

template <class Before>
void SlotCache::Heap<Before>::push(ExpertKey k) noexcept {
    heap_.push_back(k);  // capacity reserved for all keys in init()
    sift_up(static_cast<std::uint32_t>(heap_.size() - 1));
}

template <class Before>
void SlotCache::Heap<Before>::update(ExpertKey k) noexcept {
    const std::uint32_t i = pos_[k];
    sift_up(i);
    sift_down(pos_[k]);
}

template <class Before>
void SlotCache::Heap<Before>::erase(ExpertKey k) noexcept {
    const std::uint32_t i = pos_[k];
    const ExpertKey last = heap_.back();
    heap_.pop_back();
    pos_[k] = kNone;
    if (last == k) {
        return;
    }
    heap_[i] = last;
    pos_[last] = i;
    sift_up(i);
    sift_down(pos_[last]);
}

template <class Before>
ExpertKey SlotCache::Heap<Before>::pop() noexcept {
    const ExpertKey k = heap_[0];
    erase(k);
    return k;
}

template <class Before>
void SlotCache::Heap<Before>::clear() noexcept {
    for (ExpertKey k : heap_) {
        pos_[k] = kNone;
    }
    heap_.clear();
}

// ---------------------------------------------------------------- construction

SlotCache::SlotCache(CacheConfig config) : cfg_(std::move(config)) {
    if (cfg_.n_layers == 0 || cfg_.experts_per_layer == 0) {
        throw std::invalid_argument("n_layers and experts_per_layer must be positive");
    }
    const std::uint64_t n64 = static_cast<std::uint64_t>(cfg_.n_layers) * cfg_.experts_per_layer;
    if (n64 >= (1ULL << 31)) {
        throw std::invalid_argument("too many experts");
    }
    n_keys_ = static_cast<std::uint32_t>(n64);
    if (cfg_.expert_bytes.size() != n_keys_) {
        throw std::invalid_argument("expert_bytes must have n_layers * experts_per_layer entries");
    }
    for (std::uint32_t b : cfg_.expert_bytes) {
        if (b == 0) {
            throw std::invalid_argument("expert_bytes entries must be positive");
        }
    }
    if (!cfg_.prior.empty() && cfg_.prior.size() != n_keys_) {
        throw std::invalid_argument("prior must be empty or have one entry per expert");
    }
    for (double p : cfg_.prior) {
        if (!std::isfinite(p) || p < 0.0) {
            throw std::invalid_argument("prior entries must be finite and non-negative");
        }
    }
    const PolicyConfig& pc = cfg_.policy;
    if (!(pc.half_life_tokens > 0.0) || !std::isfinite(pc.half_life_tokens)) {
        throw std::invalid_argument("half_life_tokens must be positive and finite");
    }
    if (!(pc.prior_bonus_tokens >= 0.0) || !std::isfinite(pc.prior_bonus_tokens)) {
        throw std::invalid_argument("prior_bonus_tokens must be non-negative and finite");
    }
    const PromotionConfig& rc = cfg_.promotion;
    if (rc.admit_min_misses == 0 || rc.max_inflight == 0 || !std::isfinite(rc.miss_priority)) {
        throw std::invalid_argument("invalid promotion config");
    }
    if (cfg_.max_budget_bytes < cfg_.budget_bytes) {
        cfg_.max_budget_bytes = cfg_.budget_bytes;
    }
    budget_ = cfg_.budget_bytes;
    max_budget_ = cfg_.max_budget_bytes;

    // Size classes.
    class_bytes_ = cfg_.expert_bytes;
    std::sort(class_bytes_.begin(), class_bytes_.end());
    class_bytes_.erase(std::unique(class_bytes_.begin(), class_bytes_.end()), class_bytes_.end());
    if (class_bytes_.size() > 65535) {
        throw std::invalid_argument("too many distinct expert sizes");
    }
    cls_.resize(n_keys_);
    bytes_ = cfg_.expert_bytes;
    std::vector<std::uint32_t> keys_in_class(class_bytes_.size(), 0);
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        const auto it = std::lower_bound(class_bytes_.begin(), class_bytes_.end(), bytes_[k]);
        cls_[k] = static_cast<std::uint16_t>(it - class_bytes_.begin());
        ++keys_in_class[cls_[k]];
    }
    max_slots_.resize(class_bytes_.size());
    next_fresh_.assign(class_bytes_.size(), 0);
    free_slots_.resize(class_bytes_.size());
    for (std::size_t c = 0; c < class_bytes_.size(); ++c) {
        // Live slots of a class <= bytes that can be live / size, and <= keys of the class (a key
        // holds either a resident slot or a destination slot, never both).
        const std::uint64_t by_bytes = (max_budget_ + rc.staging_bytes) / class_bytes_[c];
        max_slots_[c] =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(by_bytes, keys_in_class[c]));
        free_slots_[c].reserve(max_slots_[c]);
    }

    slot_.assign(n_keys_, kNoSlot);
    flags_.assign(n_keys_, 0);
    prio_.assign(n_keys_, 0.0);
    bonus_.assign(n_keys_, 0.0);
    seq_.assign(n_keys_, 0);
    last_token_.assign(n_keys_, kNever);
    miss_count_.assign(n_keys_, 0);
    miss_start_.assign(n_keys_, 0);
    cand_prio_.assign(n_keys_, 0.0);
    cand_seq_.assign(n_keys_, 0);
    cand_token_.assign(n_keys_, 0);
    claimed_by_.assign(n_keys_, kNone);
    mark_.assign(n_keys_, 0);
    scratch_.reserve(n_keys_);
    layer_.assign(cfg_.n_layers, Counters{});

    if (!cfg_.prior.empty()) {
        const double mean = std::accumulate(cfg_.prior.begin(), cfg_.prior.end(), 0.0) / n_keys_;
        if (mean > 0.0) {
            for (std::uint32_t k = 0; k < n_keys_; ++k) {
                bonus_[k] = pc.prior_bonus_tokens * cfg_.prior[k] / mean;
            }
        }
    }
    static_order_.resize(n_keys_);
    std::iota(static_order_.begin(), static_order_.end(), 0U);
    if (!cfg_.prior.empty()) {
        std::stable_sort(
            static_order_.begin(), static_order_.end(),
            [this](ExpertKey a, ExpertKey b) { return cfg_.prior[a] > cfg_.prior[b]; });
    }

    pending_.resize(rc.max_inflight);
    evict_heap_.init(n_keys_, EvictBefore{this});
    queue_.init(n_keys_, QueueBefore{this});
    init_scores();
    if (pc.kind == PolicyKind::kStatic) {
        static_refill();
    }
}

void SlotCache::init_scores() noexcept {
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        switch (cfg_.policy.kind) {
            case PolicyKind::kLfu:
                prio_[k] = -kInf;
                break;
            case PolicyKind::kFwLru:
                prio_[k] = bonus_[k];
                break;
            default:
                prio_[k] = 0.0;
                break;
        }
        seq_[k] = 0;
        last_token_[k] = kNever;
        miss_count_[k] = 0;
    }
}

// ---------------------------------------------------------------- ordering

bool SlotCache::evict_before(ExpertKey a, ExpertKey b) const noexcept {
    if (prio_[a] != prio_[b]) {
        return prio_[a] < prio_[b];
    }
    if (seq_[a] != seq_[b]) {
        return seq_[a] < seq_[b];
    }
    return a < b;
}

bool SlotCache::queue_before(ExpertKey a, ExpertKey b) const noexcept {
    if (cand_prio_[a] != cand_prio_[b]) {
        return cand_prio_[a] > cand_prio_[b];
    }
    if (cand_seq_[a] != cand_seq_[b]) {
        return cand_seq_[a] > cand_seq_[b];
    }
    return a < b;
}

bool SlotCache::protected_now(ExpertKey k) const noexcept {
    return cfg_.promotion.protect_current_token && last_token_[k] == cur_token_;
}

// ---------------------------------------------------------------- slots and residency

std::uint32_t SlotCache::alloc_slot(std::uint32_t cls) noexcept {
    auto& stack = free_slots_[cls];
    if (!stack.empty()) {
        const std::uint32_t s = stack.back();
        stack.pop_back();
        return s;
    }
    if (next_fresh_[cls] < max_slots_[cls]) {
        return next_fresh_[cls]++;
    }
    return kNoSlot;
}

void SlotCache::free_slot(std::uint32_t cls, std::uint32_t slot) noexcept {
    free_slots_[cls].push_back(slot);  // capacity reserved: at most max_slots live ids
}

void SlotCache::make_resident(ExpertKey k, std::uint32_t slot) noexcept {
    slot_[k] = slot;
    resident_bytes_ += bytes_[k];
    ++resident_count_;
    if ((flags_[k] & kPinned) == 0) {
        evict_heap_.push(k);
    }
}

void SlotCache::count_eviction(ExpertKey k) noexcept {
    ++total_.evictions;
    ++layer_[k / cfg_.experts_per_layer].evictions;
}

void SlotCache::evict_resident(ExpertKey k) noexcept {
    if (evict_heap_.contains(k)) {
        evict_heap_.erase(k);
    }
    free_slot(cls_[k], slot_[k]);
    slot_[k] = kNoSlot;
    resident_bytes_ -= bytes_[k];
    --resident_count_;
    count_eviction(k);
}

void SlotCache::restore_victim(ExpertKey k) noexcept {
    if ((flags_[k] & kPinned) == 0 && !evict_heap_.contains(k)) {
        evict_heap_.push(k);
    }
}

ExpertKey SlotCache::pick_victim(bool respect_protection) noexcept {
    ExpertKey found = kNone;
    while (!evict_heap_.empty()) {
        const ExpertKey k = evict_heap_.pop();
        if (respect_protection && protected_now(k)) {
            scratch_.push_back(k);
            continue;
        }
        found = k;
        break;
    }
    for (ExpertKey k : scratch_) {
        evict_heap_.push(k);
    }
    scratch_.clear();
    return found;
}

bool SlotCache::pick_victims(std::uint64_t need, ExpertKey* out, std::uint32_t& n_out,
                             bool respect_protection) noexcept {
    const std::uint32_t start = n_out;
    std::uint64_t got = 0;
    bool ok = true;
    too_many_ = false;
    while (got < need) {
        if (n_out >= kMaxVictims) {
            ok = false;
            too_many_ = true;
            break;
        }
        const ExpertKey k = pick_victim(respect_protection);
        if (k == kNone) {
            ok = false;
            break;
        }
        out[n_out++] = k;
        got += bytes_[k];
    }
    if (!ok) {
        for (std::uint32_t i = start; i < n_out; ++i) {
            restore_victim(out[i]);
        }
        n_out = start;
    }
    return ok;
}

void SlotCache::claim(ExpertKey k, std::uint32_t pending_index) noexcept {
    flags_[k] |= kClaimed;
    claimed_by_[k] = pending_index;
    victim_bytes_ += bytes_[k];
}

void SlotCache::doom(std::uint32_t idx) noexcept {
    Pending& pe = pending_[idx];
    for (std::uint32_t i = 0; i < pe.p.n_victims; ++i) {
        const ExpertKey v = pe.p.victims[i];
        flags_[v] = static_cast<std::uint8_t>(flags_[v] & ~kClaimed);
        claimed_by_[v] = kNone;
        victim_bytes_ -= bytes_[v];
        restore_victim(v);
    }
    pe.p.n_victims = 0;
    planned_bytes_ -= pe.p.bytes;
    pe.doomed = true;
}

void SlotCache::reclaim(ExpertKey v) noexcept {
    const std::uint32_t idx = claimed_by_[v];
    Pending& pe = pending_[idx];
    flags_[v] = static_cast<std::uint8_t>(flags_[v] & ~kClaimed);
    claimed_by_[v] = kNone;
    victim_bytes_ -= bytes_[v];
    ++total_.victims_reclaimed;
    ++layer_[v / cfg_.experts_per_layer].victims_reclaimed;
    std::uint32_t w = 0;
    for (std::uint32_t i = 0; i < pe.p.n_victims; ++i) {
        if (pe.p.victims[i] != v) {
            pe.p.victims[w] = pe.p.victims[i];
            pe.p.victim_slots[w] = pe.p.victim_slots[i];
            ++w;
        }
    }
    pe.p.n_victims = w;
    restore_victim(v);
    const std::uint64_t proj = projected();
    if (proj <= budget_) {
        return;
    }
    const std::uint32_t old_n = pe.p.n_victims;
    std::uint32_t n = old_n;
    if (!pick_victims(proj - budget_, pe.p.victims, n, cfg_.promotion.protect_current_token)) {
        doom(idx);
        // Victims chosen with mixed sizes can cover more than their promotion needed, and later
        // promotions may have relied on that slack: restore I1 by evicting by policy now.
        while (projected() > budget_) {
            const ExpertKey extra = pick_victim(false);
            if (extra == kNone) {
                break;
            }
            evict_resident(extra);
        }
        return;
    }
    for (std::uint32_t i = old_n; i < n; ++i) {
        claim(pe.p.victims[i], idx);
        pe.p.victim_slots[i] = slot_[pe.p.victims[i]];
    }
    pe.p.n_victims = n;
}

void SlotCache::cancel_all_pending() noexcept {
    for (std::uint32_t i = 0; i < pending_.size(); ++i) {
        Pending& pe = pending_[i];
        if (!pe.live) {
            continue;
        }
        if (!pe.doomed) {
            doom(i);
        }
        free_slot(pe.p.size_class, pe.p.dest_slot);
        inflight_bytes_ -= pe.p.bytes;
        flags_[pe.p.key] = static_cast<std::uint8_t>(flags_[pe.p.key] & ~kPending);
        pe.live = false;
        ++total_.cancelled;
        ++layer_[pe.p.key / cfg_.experts_per_layer].cancelled;
    }
    n_pending_ = 0;
}

// ---------------------------------------------------------------- compute thread

void SlotCache::count_touch(ExpertKey k, bool hit) noexcept {
    Counters& l = layer_[k / cfg_.experts_per_layer];
    ++total_.touches;
    ++l.touches;
    if (hit) {
        ++total_.hits;
        ++l.hits;
    } else {
        ++total_.misses;
        ++l.misses;
    }
}

bool SlotCache::touch(ExpertKey key, std::uint64_t token_index) noexcept {
    if (token_index != cur_token_) {
        cur_token_ = token_index;
        token_issued_bytes_ = 0;
    }
    seq_[key] = ++seq_ctr_;
    switch (cfg_.policy.kind) {
        case PolicyKind::kLfu: {
            const double th = static_cast<double>(token_index) / cfg_.policy.half_life_tokens;
            prio_[key] = std::log2(std::exp2(prio_[key] - th) + 1.0) + th;
            break;
        }
        case PolicyKind::kFwLru:
            prio_[key] = static_cast<double>(token_index) + bonus_[key];
            break;
        default:
            break;
    }
    last_token_[key] = token_index;
    const bool hit = slot_[key] != kNoSlot;
    count_touch(key, hit);
    if (hit) {
        if ((flags_[key] & kClaimed) != 0) {
            reclaim(key);
        } else if (evict_heap_.contains(key)) {
            evict_heap_.update(key);
        }
    }
    return hit;
}

void SlotCache::enqueue(ExpertKey k, double priority, bool from_hint,
                        std::uint64_t token) noexcept {
    if (queue_.contains(k)) {
        cand_prio_[k] = std::max(cand_prio_[k], priority);
        cand_seq_[k] = ++seq_ctr_;
        queue_.update(k);
        return;
    }
    cand_prio_[k] = priority;
    cand_seq_[k] = ++seq_ctr_;
    cand_token_[k] = token;
    flags_[k] = from_hint ? static_cast<std::uint8_t>(flags_[k] | kHint)
                          : static_cast<std::uint8_t>(flags_[k] & ~kHint);
    queue_.push(k);
}

bool SlotCache::on_miss(ExpertKey key, std::uint64_t token_index) noexcept {
    if (cfg_.policy.kind == PolicyKind::kStatic || is_resident(key) ||
        (flags_[key] & kPending) != 0) {
        return false;
    }
    const PromotionConfig& rc = cfg_.promotion;
    if (rc.admit_min_misses > 1 && !queue_.contains(key)) {
        const bool expired = rc.admit_window_tokens != 0 && miss_count_[key] != 0 &&
                             token_index - miss_start_[key] > rc.admit_window_tokens;
        if (miss_count_[key] == 0 || expired) {
            miss_count_[key] = 1;
            miss_start_[key] = token_index;
        } else {
            ++miss_count_[key];
        }
        if (miss_count_[key] < rc.admit_min_misses) {
            ++total_.admission_rejected;
            ++layer_[key / cfg_.experts_per_layer].admission_rejected;
            return false;
        }
        miss_count_[key] = 0;
    }
    enqueue(key, rc.miss_priority, false, token_index);
    return true;
}

std::size_t SlotCache::hint(std::span<const ExpertKey> keys, double priority) noexcept {
    if (cfg_.policy.kind == PolicyKind::kStatic) {
        return 0;
    }
    std::size_t n = 0;
    for (ExpertKey k : keys) {
        if (is_resident(k) || (flags_[k] & kPending) != 0) {
            continue;
        }
        enqueue(k, priority, true, cur_token_);
        ++total_.hints;
        ++layer_[k / cfg_.experts_per_layer].hints;
        ++n;
    }
    return n;
}

// ---------------------------------------------------------------- copier control

SlotCache::Issue SlotCache::try_issue(ExpertKey k, Promotion& out) noexcept {
    const std::uint32_t b = bytes_[k];
    const std::uint32_t c = cls_[k];
    if (b > budget_ || pinned_bytes_ + ((flags_[k] & kPinned) != 0 ? 0 : b) > budget_) {
        return Issue::kDrop;  // can never fit
    }
    if (n_pending_ >= pending_.size() ||
        resident_bytes_ + inflight_bytes_ + b > budget_ + cfg_.promotion.staging_bytes) {
        return Issue::kNoVictim;
    }
    const std::uint32_t dest = alloc_slot(c);
    if (dest == kNoSlot) {
        return Issue::kNoVictim;
    }
    const std::uint64_t proj = projected() + b;
    const std::uint64_t need = proj > budget_ ? proj - budget_ : 0;
    ExpertKey victims[kMaxVictims];
    std::uint32_t n = 0;
    if (need > 0 && !pick_victims(need, victims, n, cfg_.promotion.protect_current_token)) {
        free_slot(c, dest);
        return too_many_ ? Issue::kDrop : Issue::kNoVictim;
    }
    if (cfg_.policy.score_gate && (flags_[k] & kHint) == 0) {
        for (std::uint32_t i = 0; i < n; ++i) {
            if (!evict_before(victims[i], k)) {
                for (std::uint32_t j = 0; j < n; ++j) {
                    restore_victim(victims[j]);
                }
                free_slot(c, dest);
                ++total_.gated;
                ++layer_[k / cfg_.experts_per_layer].gated;
                return Issue::kDrop;
            }
        }
    }
    std::uint32_t idx = 0;
    while (pending_[idx].live) {
        ++idx;
    }
    Pending& pe = pending_[idx];
    pe.live = true;
    pe.doomed = false;
    pe.p = Promotion{};
    pe.p.id = next_id_++;
    pe.p.key = k;
    pe.p.bytes = b;
    pe.p.size_class = c;
    pe.p.dest_slot = dest;
    pe.p.n_victims = n;
    for (std::uint32_t i = 0; i < n; ++i) {
        pe.p.victims[i] = victims[i];
        pe.p.victim_slots[i] = slot_[victims[i]];
        claim(victims[i], idx);
    }
    flags_[k] |= kPending;
    planned_bytes_ += b;
    inflight_bytes_ += b;
    ++n_pending_;
    token_issued_bytes_ += b;
    ++total_.issued;
    total_.bytes_issued += b;
    Counters& l = layer_[k / cfg_.experts_per_layer];
    ++l.issued;
    l.bytes_issued += b;
    out = pe.p;
    return Issue::kIssued;
}

std::size_t SlotCache::next_promotions(std::uint64_t budget_bytes,
                                       std::span<Promotion> out) noexcept {
    std::uint64_t remaining = budget_bytes;
    const std::uint64_t cap = cfg_.promotion.max_bytes_per_token;
    if (cap != 0) {
        remaining = std::min(remaining, cap > token_issued_bytes_ ? cap - token_issued_bytes_ : 0);
    }
    const std::uint64_t ttl = cfg_.promotion.candidate_ttl_tokens;
    std::size_t n = 0;
    while (n < out.size() && !queue_.empty()) {
        const ExpertKey k = queue_.top();
        if (is_resident(k) || (flags_[k] & kPending) != 0 ||
            (ttl != 0 && cur_token_ > cand_token_[k] && cur_token_ - cand_token_[k] > ttl)) {
            queue_.pop();
            continue;
        }
        if (bytes_[k] > remaining) {
            break;
        }
        const Issue r = try_issue(k, out[n]);
        if (r == Issue::kNoVictim) {
            break;
        }
        queue_.pop();
        if (r == Issue::kIssued) {
            remaining -= bytes_[k];
            ++n;
        }
    }
    return n;
}

CommitResult SlotCache::commit(std::uint64_t id) noexcept {
    std::uint32_t idx = 0;
    while (idx < pending_.size() && !(pending_[idx].live && pending_[idx].p.id == id)) {
        ++idx;
    }
    if (idx == pending_.size()) {
        return {CommitStatus::kStale, kNoSlot};
    }
    Pending& pe = pending_[idx];
    const Promotion p = pe.p;
    const ExpertKey k = p.key;
    pe.live = false;
    --n_pending_;
    inflight_bytes_ -= p.bytes;
    flags_[k] = static_cast<std::uint8_t>(flags_[k] & ~kPending);
    if (pe.doomed) {
        free_slot(p.size_class, p.dest_slot);
        ++total_.aborted;
        ++layer_[k / cfg_.experts_per_layer].aborted;
        return {CommitStatus::kAborted, kNoSlot};
    }
    planned_bytes_ -= p.bytes;
    for (std::uint32_t i = 0; i < p.n_victims; ++i) {
        const ExpertKey v = p.victims[i];
        flags_[v] = static_cast<std::uint8_t>(flags_[v] & ~kClaimed);
        claimed_by_[v] = kNone;
        victim_bytes_ -= bytes_[v];
        evict_resident(v);
    }
    make_resident(k, p.dest_slot);
    ++total_.promotions;
    total_.bytes_copied += p.bytes;
    Counters& l = layer_[k / cfg_.experts_per_layer];
    ++l.promotions;
    l.bytes_copied += p.bytes;
    return {CommitStatus::kCommitted, p.dest_slot};
}

// ---------------------------------------------------------------- control

bool SlotCache::pin(ExpertKey key) noexcept {
    if ((flags_[key] & kPinned) != 0) {
        return true;
    }
    if (pinned_bytes_ + bytes_[key] > budget_) {
        return false;
    }
    flags_[key] |= kPinned;
    pinned_bytes_ += bytes_[key];
    if (is_resident(key)) {
        if ((flags_[key] & kClaimed) != 0) {
            reclaim(key);
        } else if (evict_heap_.contains(key)) {
            evict_heap_.erase(key);
        }
    } else if (cfg_.policy.kind == PolicyKind::kStatic) {
        static_refill();
    } else if ((flags_[key] & kPending) == 0) {
        enqueue(key, kInf, true, cur_token_);
    }
    return true;
}

void SlotCache::unpin(ExpertKey key) noexcept {
    if ((flags_[key] & kPinned) == 0) {
        return;
    }
    flags_[key] = static_cast<std::uint8_t>(flags_[key] & ~kPinned);
    pinned_bytes_ -= bytes_[key];
    if (is_resident(key) && (flags_[key] & kClaimed) == 0) {
        restore_victim(key);
    }
    if (cfg_.policy.kind == PolicyKind::kStatic) {
        static_refill();
    }
}

bool SlotCache::install(ExpertKey key) noexcept {
    if (is_resident(key)) {
        return true;
    }
    const std::uint32_t b = bytes_[key];
    if ((flags_[key] & kPending) != 0 || projected() + b > budget_ ||
        resident_bytes_ + inflight_bytes_ + b > budget_ + cfg_.promotion.staging_bytes) {
        return false;
    }
    const std::uint32_t slot = alloc_slot(cls_[key]);
    if (slot == kNoSlot) {
        return false;
    }
    if (queue_.contains(key)) {
        queue_.erase(key);
    }
    make_resident(key, slot);
    ++total_.installs;
    ++layer_[key / cfg_.experts_per_layer].installs;
    return true;
}

void SlotCache::static_refill() noexcept {
    std::fill(mark_.begin(), mark_.end(), std::uint8_t{0});
    std::uint64_t used = 0;
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        if ((flags_[k] & kPinned) != 0) {
            mark_[k] = 1;
            used += bytes_[k];
        }
    }
    for (ExpertKey k : static_order_) {
        if (mark_[k] == 0 && used + bytes_[k] <= budget_) {
            mark_[k] = 1;
            used += bytes_[k];
        }
    }
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        if (mark_[k] == 0 && is_resident(k)) {
            evict_resident(k);
        }
    }
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        if (mark_[k] != 0 && !is_resident(k)) {
            install(k);
        }
    }
}

bool SlotCache::set_budget(std::uint64_t budget_bytes) noexcept {
    if (budget_bytes > max_budget_ || pinned_bytes_ > budget_bytes) {
        return false;
    }
    const bool shrink = budget_bytes < budget_;
    if (shrink) {
        cancel_all_pending();
    }
    budget_ = budget_bytes;
    if (cfg_.policy.kind == PolicyKind::kStatic) {
        static_refill();
        return true;
    }
    while (resident_bytes_ > budget_) {
        const ExpertKey v = pick_victim(false);
        if (v == kNone) {
            break;
        }
        evict_resident(v);
    }
    return true;
}

void SlotCache::reset() noexcept {
    cancel_all_pending();
    while (!queue_.empty()) {
        const ExpertKey k = queue_.pop();
        flags_[k] = static_cast<std::uint8_t>(flags_[k] & ~kHint);
    }
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        if (is_resident(k)) {
            if (evict_heap_.contains(k)) {
                evict_heap_.erase(k);
            }
            free_slot(cls_[k], slot_[k]);
            slot_[k] = kNoSlot;
        }
    }
    resident_bytes_ = 0;
    resident_count_ = 0;
    token_issued_bytes_ = 0;
    init_scores();
    if (cfg_.policy.kind == PolicyKind::kStatic) {
        static_refill();
        return;
    }
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        if ((flags_[k] & kPinned) != 0) {
            install(k);
        }
    }
}

// ---------------------------------------------------------------- introspection

Snapshot SlotCache::snapshot() const noexcept {
    Snapshot s;
    s.total = total_;
    s.budget_bytes = budget_;
    s.resident_bytes = resident_bytes_;
    s.inflight_bytes = inflight_bytes_;
    s.pinned_bytes = pinned_bytes_;
    s.current_token = cur_token_;
    s.resident_count = resident_count_;
    s.pending = n_pending_;
    s.queued = static_cast<std::uint32_t>(queue_.size());
    return s;
}

std::size_t SlotCache::resident_keys(std::span<ExpertKey> out) const noexcept {
    std::size_t n = 0;
    for (std::uint32_t k = 0; k < n_keys_ && n < out.size(); ++k) {
        if (slot_[k] != kNoSlot) {
            out[n++] = k;
        }
    }
    return n;
}

bool SlotCache::fail(int line) const noexcept {
    failed_line_ = line;
    return false;
}

bool SlotCache::verify_invariants() const {
    failed_line_ = 0;
    std::uint64_t res = 0;
    std::uint64_t vic = 0;
    std::uint64_t plan = 0;
    std::uint64_t infl = 0;
    std::uint64_t pinned = 0;
    std::uint32_t count = 0;
    std::vector<std::vector<std::uint8_t>> used(class_bytes_.size());
    for (std::size_t c = 0; c < used.size(); ++c) {
        used[c].assign(max_slots_[c], 0);
    }
    for (std::uint32_t k = 0; k < n_keys_; ++k) {
        if ((flags_[k] & kPinned) != 0) {
            pinned += bytes_[k];
        }
        const bool resident = slot_[k] != kNoSlot;
        if (resident) {
            res += bytes_[k];
            ++count;
            if (slot_[k] >= max_slots_[cls_[k]] || used[cls_[k]][slot_[k]] != 0) {
                return fail(__LINE__);
            }
            used[cls_[k]][slot_[k]] = 1;
        }
        const bool claimed = (flags_[k] & kClaimed) != 0;
        if (claimed && !resident) {
            return fail(__LINE__);
        }
        const bool pinned_k = (flags_[k] & kPinned) != 0;
        if (evict_heap_.contains(k) != (resident && !claimed && !pinned_k)) {
            return fail(__LINE__);
        }
        if ((flags_[k] & kPending) != 0 && resident) {
            return fail(__LINE__);
        }
        if (queue_.contains(k) && resident) {
            return fail(__LINE__);
        }
    }
    std::uint32_t live = 0;
    for (const Pending& pe : pending_) {
        if (!pe.live) {
            continue;
        }
        ++live;
        infl += pe.p.bytes;
        if ((flags_[pe.p.key] & kPending) == 0 || slot_[pe.p.key] != kNoSlot) {
            return fail(__LINE__);
        }
        if (pe.p.dest_slot >= max_slots_[pe.p.size_class] ||
            used[pe.p.size_class][pe.p.dest_slot] != 0) {
            return fail(__LINE__);
        }
        used[pe.p.size_class][pe.p.dest_slot] = 1;
        if (pe.doomed) {
            if (pe.p.n_victims != 0) {
                return fail(__LINE__);
            }
            continue;
        }
        plan += pe.p.bytes;
        for (std::uint32_t i = 0; i < pe.p.n_victims; ++i) {
            const ExpertKey v = pe.p.victims[i];
            if ((flags_[v] & kClaimed) == 0 || slot_[v] == kNoSlot) {
                return fail(__LINE__);
            }
            vic += bytes_[v];
        }
    }
    if (res != resident_bytes_ || vic != victim_bytes_ || plan != planned_bytes_ ||
        infl != inflight_bytes_ || pinned != pinned_bytes_ || count != resident_count_ ||
        live != n_pending_) {
        return fail(__LINE__);
    }
    if (res - vic + plan > budget_ || res + infl > budget_ + cfg_.promotion.staging_bytes) {
        return fail(__LINE__);
    }
    if (n_pending_ == 0 && res > budget_) {
        return fail(__LINE__);
    }
    // Free lists hold exactly the ids below the high-water mark that are not live.
    for (std::size_t c = 0; c < used.size(); ++c) {
        std::uint32_t free_n = 0;
        for (std::uint32_t id : free_slots_[c]) {
            if (id >= next_fresh_[c] || used[c][id] != 0) {
                return fail(__LINE__);
            }
            ++free_n;
        }
        std::uint32_t live_n = 0;
        for (std::uint8_t u : used[c]) {
            live_n += u;
        }
        if (live_n + free_n != next_fresh_[c]) {
            return fail(__LINE__);
        }
    }
    if (total_.hits + total_.misses != total_.touches) {
        return fail(__LINE__);
    }
    Counters sum;
    for (const Counters& l : layer_) {
        if (l.hits + l.misses != l.touches) {
            return fail(__LINE__);
        }
        sum.touches += l.touches;
        sum.promotions += l.promotions;
        sum.evictions += l.evictions;
        sum.bytes_copied += l.bytes_copied;
    }
    if (sum.touches != total_.touches || sum.promotions != total_.promotions ||
        sum.evictions != total_.evictions || sum.bytes_copied != total_.bytes_copied) {
        return fail(__LINE__);
    }
    return true;
}

}  // namespace eq::pager
