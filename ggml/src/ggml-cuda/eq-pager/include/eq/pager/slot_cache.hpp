// vendored from elastic-quant cpp/pager at commit 0bd8cbe; do not edit here
#pragma once

// Slot cache policy core of the expert pager (design paper section 5, decision log 2026-10-12).
//
// This is pure bookkeeping: it decides which experts live in VRAM slots, which promotions the
// background copier should perform next and which residents they replace. It performs no copy,
// no CUDA call and no I/O, so the llama.cpp CUDA backend can embed it and the offline replay can
// drive it with a modelled clock.
//
// Keys. An expert is the dense index key = layer * experts_per_layer + expert (layer here is the
// index among the traced MoE layers, 0 .. n_layers-1).
//
// Residency. A resident expert owns one slot: (size class, slot_id). The size class is the rank of
// the expert's byte size among the distinct sizes of the model (Q4_K_M mixes tensor types, so
// sizes can differ by layer); slot_id is a small integer, unique among live slots of that class,
// reused after the slot is freed. The backend maps (class, slot_id) to a device address as
// class_base[class] + slot_id * class_bytes(class), with a region of max_slots(class) slots per
// class (a virtual reservation, or an arena per class). max_slots(class) is a proven bound on the
// number of simultaneously live slots (residents plus in-flight destinations) of that class, so
// slot_id < max_slots(class) always. The policy itself never assumes a slot geometry: the budget
// is a byte count and each resident consumes its own size.
//
// Byte accounting (invariants checked by verify_invariants()):
//   resident_bytes  = bytes of keys that have a slot (includes victims until their commit)
//   victim_bytes    = bytes of residents claimed by an in-flight promotion
//   planned_bytes   = bytes of in-flight promotions that will be inserted (not doomed)
//   inflight_bytes  = bytes of destination slots of all in-flight promotions
//   I1: resident_bytes - victim_bytes + planned_bytes <= budget_bytes        (steady state fits)
//   I2: resident_bytes + inflight_bytes <= budget_bytes + staging_bytes      (physical memory)
//   I3: resident_bytes <= budget_bytes whenever no promotion is in flight.
// A victim stays resident and usable (lookup hits) until commit; the copy writes to a separate
// destination slot, which is why device memory is budget + staging_bytes.
//
// Victim touched before commit (the rule). A claimed victim that is touched or pinned is
// reclaimed at that moment: it stays resident and the promotion keeps its claim list minus that
// victim. If I1 would then fail, a replacement victim is chosen by policy right away (subject to
// protect_current_token). If none exists (everything is protected, pinned or claimed, or more than
// kMaxVictims would be needed) the promotion is doomed: it releases all its victims and its
// planned bytes, its copy still finishes, and commit() returns kAborted and frees the
// destination slot. Because victims of mixed sizes can cover more than a promotion needed and
// later promotions may have relied on that slack, dooming also evicts by policy, immediately,
// until I1 holds again (never happens with a uniform expert size).
//
// Threading contract. The class has no locks and no atomics. Exactly one thread (the "compute
// thread") may call any member function, or the caller must serialise calls. Intended use:
//   compute thread, per expert evaluation:  lookup(key) / touch(key, token) / on_miss(key, token)
//   copier control (also on the compute thread, at a safe point such as a layer boundary):
//       next_promotions(...) hands the copier its work; when a copy has finished the compute
//       thread calls commit(id) at a safe point, which swaps the maps atomically from the
//       compute thread's point of view (nothing is looked up between the two map updates).
// A copier thread only moves bytes; it never calls into this class.
//
// Hot path. lookup, touch, on_miss, hint, next_promotions and commit allocate nothing and throw
// nothing (all storage is sized by the constructor, which does throw std::invalid_argument on a
// bad configuration). Costs: lookup O(1); touch O(1) on a miss, O(log R) on a hit (R residents);
// on_miss O(log C) (C queued candidates); next_promotions and commit O(log R) per victim.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace eq::pager {

using ExpertKey = std::uint32_t;
inline constexpr std::uint32_t kNoSlot = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::size_t kMaxVictims = 8;

enum class PolicyKind : std::uint8_t {
    kLru,     // evict the least recently touched
    kLfu,     // evict the lowest exponentially decayed touch count
    kFwLru,   // evict the lowest (last touch token + prior bonus)
    kStatic,  // fixed resident set by prior, never promotes
};

// Scores (all "lower is evicted first"; ties are broken by older last touch, then lower key):
//   LRU:   no score, only the touch sequence number.
//   LFU:   s_k(t) = sum over touches at tokens u <= t of 2^(-(t - u) / half_life_tokens).
//          Stored as the time invariant log2(s_k) + t / half_life_tokens, so ordering between
//          keys does not change between touches and the eviction heap needs no re-scoring.
//   FWLRU: score_k = last_touch_token_k + prior_bonus_tokens * (prior_k / mean_prior).
//          An expert that is exactly as frequent as the average is treated as if touched
//          prior_bonus_tokens later than it was; 0 gives LRU, a large value orders by prior.
struct PolicyConfig {
    PolicyKind kind = PolicyKind::kLru;
    double half_life_tokens = 64.0;    // LFU
    double prior_bonus_tokens = 64.0;  // FWLRU
    // Miss-origin candidates are admitted for promotion only when they beat every chosen victim
    // under the policy order (TinyLFU-style gate). Hints and pins bypass the gate.
    bool score_gate = false;
};

struct PromotionConfig {
    // Admission filter: promote a missed key once it missed admit_min_misses times within
    // admit_window_tokens tokens (0 = misses never expire). 1 = promote on first miss.
    std::uint32_t admit_min_misses = 1;
    std::uint64_t admit_window_tokens = 0;
    // Cap on bytes handed out by next_promotions per token index (0 = unlimited).
    std::uint64_t max_bytes_per_token = 0;
    // Never choose a victim that was touched in the current token (waived by set_budget).
    bool protect_current_token = true;
    // Candidates older than this many tokens are dropped unissued (0 = never).
    std::uint64_t candidate_ttl_tokens = 0;
    // Queue priority of a miss-origin candidate. Hints carry the caller's priority (convention:
    // predicted probability in (0, 1)); equal priorities issue newest first.
    double miss_priority = 1.0;
    // Extra device bytes for destination slots of in-flight copies.
    std::uint64_t staging_bytes = 0;
    std::uint32_t max_inflight = 64;
};

struct CacheConfig {
    std::uint32_t n_layers = 0;
    std::uint32_t experts_per_layer = 0;
    std::vector<std::uint32_t> expert_bytes;  // size n_layers * experts_per_layer, all > 0
    std::vector<double> prior;                // per-key activation prior, empty = all zero
    std::uint64_t budget_bytes = 0;
    std::uint64_t max_budget_bytes = 0;  // upper bound for set_budget; 0 = budget_bytes
    PolicyConfig policy;
    PromotionConfig promotion;
};

struct Counters {
    std::uint64_t touches = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t promotions = 0;    // committed
    std::uint64_t evictions = 0;     // residents removed (commit, set_budget, static refill)
    std::uint64_t bytes_copied = 0;  // bytes of committed promotions
    std::uint64_t issued = 0;        // handed out by next_promotions
    std::uint64_t bytes_issued = 0;
    std::uint64_t aborted = 0;             // commit found its victims gone and no replacement
    std::uint64_t cancelled = 0;           // dropped by set_budget / reset
    std::uint64_t victims_reclaimed = 0;   // victim touched/pinned before commit
    std::uint64_t admission_rejected = 0;  // miss below the admission threshold
    std::uint64_t gated = 0;               // dropped by the score gate
    std::uint64_t hints = 0;               // hint keys that entered or were refreshed in the queue
    std::uint64_t installs = 0;            // install()/static fill (setup, not counted as copies)
};

struct Snapshot {
    Counters total;
    std::uint64_t budget_bytes = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t inflight_bytes = 0;
    std::uint64_t pinned_bytes = 0;
    std::uint64_t current_token = 0;
    std::uint32_t resident_count = 0;
    std::uint32_t pending = 0;
    std::uint32_t queued = 0;
};

struct Promotion {
    std::uint64_t id = 0;
    ExpertKey key = 0;
    std::uint32_t bytes = 0;
    std::uint32_t size_class = 0;
    std::uint32_t dest_slot = kNoSlot;  // slot to copy into
    std::uint32_t n_victims = 0;
    ExpertKey victims[kMaxVictims] = {};
    std::uint32_t victim_slots[kMaxVictims] = {};  // informational: freed at commit
};

enum class CommitStatus : std::uint8_t {
    kCommitted,  // maps swapped; `slot` is the new resident slot
    kAborted,    // a victim was touched/pinned and no replacement exists: nothing changed
    kStale,      // unknown id (already committed, or cancelled by set_budget/reset)
};

struct CommitResult {
    CommitStatus status = CommitStatus::kStale;
    std::uint32_t slot = kNoSlot;
};

class SlotCache {
  public:
    // Throws std::invalid_argument on an inconsistent configuration.
    explicit SlotCache(CacheConfig config);
    SlotCache(const SlotCache&) = delete;
    SlotCache& operator=(const SlotCache&) = delete;

    // ---- compute thread, per expert evaluation -------------------------------------------
    // Slot of a resident key or kNoSlot. O(1), no side effects.
    std::uint32_t lookup(ExpertKey key) const noexcept { return slot_[key]; }

    // Record an access (hit or miss) at token_index (non-decreasing across calls). Counts
    // hits/misses and updates the policy state. A victim claimed by an in-flight promotion that
    // is touched here is reclaimed (see commit). Returns whether the key was resident.
    bool touch(ExpertKey key, std::uint64_t token_index) noexcept;

    // The access missed: apply the admission filter and enqueue a promotion candidate. Returns
    // whether the key entered the queue. No-op for kStatic and for keys resident or in flight.
    bool on_miss(ExpertKey key, std::uint64_t token_index) noexcept;

    // Prediction (not a miss): enqueue keys with `priority` (higher first), bypassing the
    // admission filter. A key already queued keeps the larger priority. Returns keys enqueued
    // or refreshed. Does not change any policy score.
    std::size_t hint(std::span<const ExpertKey> keys, double priority) noexcept;

    // ---- copier control ------------------------------------------------------------------
    // Fill `out` with the promotions the copier should start now, highest queue priority first,
    // total bytes <= budget_bytes (and the per-token cap), stopping at the first candidate that
    // does not fit or has no available victim. Each promotion reserves its destination slot and
    // claims its victims; victims stay resident and usable until commit(). Returns the count.
    std::size_t next_promotions(std::uint64_t budget_bytes, std::span<Promotion> out) noexcept;

    // The copy of promotion `id` finished: evict its victims and make the new key resident, in
    // one step. If a victim was touched or pinned since it was claimed, it is reclaimed (kept)
    // and a replacement victim is chosen at that moment; if none exists the promotion is
    // aborted at commit (kAborted, nothing evicted, destination slot freed).
    CommitResult commit(std::uint64_t id) noexcept;

    // ---- control -------------------------------------------------------------------------
    // Pinned keys are never evicted and count against the budget. A non-resident key is queued
    // with top priority (and for kStatic the resident set is rebuilt). Returns false when the
    // pinned bytes would exceed the budget.
    bool pin(ExpertKey key) noexcept;
    void unpin(ExpertKey key) noexcept;

    // Make `key` resident now if it fits without eviction (startup / warm start; not a copy).
    bool install(ExpertKey key) noexcept;

    // Shrink or grow the byte budget (<= max_budget_bytes). Shrinking cancels every in-flight
    // promotion (their commit() then returns kStale) and evicts by policy until resident bytes
    // fit. Returns false, changing nothing, if pinned bytes exceed the new budget or it is
    // above max_budget_bytes. For kStatic the fixed set is recomputed for the new budget.
    bool set_budget(std::uint64_t budget_bytes) noexcept;

    // Empty the cache and queue, cancel in-flight promotions, forget scores and miss history.
    // Counters, pins and the budget are kept (pinned keys and the static set are reinstalled).
    void reset() noexcept;

    // ---- introspection -------------------------------------------------------------------
    Snapshot snapshot() const noexcept;
    const Counters& layer_counters(std::uint32_t layer) const noexcept { return layer_[layer]; }
    std::size_t resident_keys(std::span<ExpertKey> out) const noexcept;  // ascending key order
    std::uint32_t size_class(ExpertKey key) const noexcept { return cls_[key]; }
    std::uint32_t n_classes() const noexcept {
        return static_cast<std::uint32_t>(class_bytes_.size());
    }
    std::uint32_t class_bytes(std::uint32_t c) const noexcept { return class_bytes_[c]; }
    std::uint32_t max_slots(std::uint32_t c) const noexcept { return max_slots_[c]; }
    std::uint32_t n_keys() const noexcept { return n_keys_; }
    std::uint64_t budget_bytes() const noexcept { return budget_; }
    bool is_pinned(ExpertKey key) const noexcept { return (flags_[key] & kPinned) != 0; }
    const PolicyConfig& policy() const noexcept { return cfg_.policy; }

    // O(n) consistency check of every invariant above (for tests and debugging, allocates).
    bool verify_invariants() const;
    // Source line of the first failed check of the last verify_invariants() (0 = none).
    int failed_line() const noexcept { return failed_line_; }

  private:
    static constexpr std::uint8_t kPinned = 1;
    static constexpr std::uint8_t kClaimed = 2;  // victim of an in-flight promotion
    static constexpr std::uint8_t kPending = 4;  // destination of an in-flight promotion
    static constexpr std::uint8_t kHint = 8;     // queued candidate came from a hint/pin
    static constexpr std::uint32_t kNone = 0xFFFFFFFFu;

    // Indexed binary heap over keys; `Before(a, b)` true when a must come out first.
    template <class Before>
    class Heap {
      public:
        void init(std::uint32_t n, Before before) {
            before_ = before;
            heap_.clear();
            heap_.reserve(n);
            pos_.assign(n, kNone);
        }
        bool contains(ExpertKey k) const noexcept { return pos_[k] != kNone; }
        std::size_t size() const noexcept { return heap_.size(); }
        bool empty() const noexcept { return heap_.empty(); }
        ExpertKey top() const noexcept { return heap_[0]; }
        void push(ExpertKey k) noexcept;
        void update(ExpertKey k) noexcept;
        void erase(ExpertKey k) noexcept;
        ExpertKey pop() noexcept;
        void clear() noexcept;
        const std::vector<std::uint32_t>& items() const noexcept { return heap_; }

      private:
        void sift_up(std::uint32_t i) noexcept;
        void sift_down(std::uint32_t i) noexcept;
        Before before_{};
        std::vector<std::uint32_t> heap_;
        std::vector<std::uint32_t> pos_;
    };

    struct EvictBefore {
        const SlotCache* c = nullptr;
        bool operator()(ExpertKey a, ExpertKey b) const noexcept { return c->evict_before(a, b); }
    };
    struct QueueBefore {
        const SlotCache* c = nullptr;
        bool operator()(ExpertKey a, ExpertKey b) const noexcept { return c->queue_before(a, b); }
    };

    struct Pending {
        Promotion p;
        bool live = false;
        bool doomed = false;  // will be aborted at commit; holds no victims
    };

    bool evict_before(ExpertKey a, ExpertKey b) const noexcept;
    bool queue_before(ExpertKey a, ExpertKey b) const noexcept;
    bool protected_now(ExpertKey k) const noexcept;
    bool is_resident(ExpertKey k) const noexcept { return slot_[k] != kNoSlot; }

    std::uint32_t alloc_slot(std::uint32_t cls) noexcept;
    void free_slot(std::uint32_t cls, std::uint32_t slot) noexcept;
    void make_resident(ExpertKey k, std::uint32_t slot) noexcept;
    void evict_resident(ExpertKey k) noexcept;  // remove from maps and heap, count the eviction
    ExpertKey pick_victim(bool respect_protection) noexcept;  // removes it from the heap
    // Pick victims worth `need` bytes into out[n_out..], respecting kMaxVictims. All picked
    // victims are removed from the heap; on failure they are restored and false is returned.
    bool pick_victims(std::uint64_t need, ExpertKey* out, std::uint32_t& n_out,
                      bool respect_protection) noexcept;
    void restore_victim(ExpertKey k) noexcept;  // back into the heap (unless pinned)
    void claim(ExpertKey k, std::uint32_t pending_index) noexcept;
    void reclaim(ExpertKey k) noexcept;
    void doom(std::uint32_t pending_index) noexcept;
    void cancel_all_pending() noexcept;
    void enqueue(ExpertKey k, double priority, bool from_hint, std::uint64_t token) noexcept;
    enum class Issue : std::uint8_t { kIssued, kNoVictim, kDrop };
    Issue try_issue(ExpertKey k, Promotion& out) noexcept;
    void static_refill() noexcept;
    void count_eviction(ExpertKey k) noexcept;
    void count_touch(ExpertKey k, bool hit) noexcept;
    void init_scores() noexcept;
    std::uint64_t projected() const noexcept {
        return resident_bytes_ - victim_bytes_ + planned_bytes_;
    }

    CacheConfig cfg_;
    std::uint32_t n_keys_ = 0;
    std::uint64_t budget_ = 0;
    std::uint64_t max_budget_ = 0;

    // Per key.
    std::vector<std::uint32_t> slot_;
    std::vector<std::uint16_t> cls_;
    std::vector<std::uint32_t> bytes_;
    std::vector<std::uint8_t> flags_;
    std::vector<double> prio_;        // policy score (eviction heap primary key)
    std::vector<double> bonus_;       // FWLRU prior bonus in tokens
    std::vector<std::uint64_t> seq_;  // touch sequence number (tie break)
    std::vector<std::uint64_t> last_token_;
    std::vector<std::uint32_t> miss_count_;
    std::vector<std::uint64_t> miss_start_;
    std::vector<double> cand_prio_;
    std::vector<std::uint64_t> cand_seq_;
    std::vector<std::uint64_t> cand_token_;
    std::vector<std::uint32_t> claimed_by_;  // index into pending_ for flagged kClaimed keys
    std::vector<ExpertKey> static_order_;    // keys by prior descending (kStatic)

    // Per size class.
    std::vector<std::uint32_t> class_bytes_;
    std::vector<std::uint32_t> max_slots_;
    std::vector<std::uint32_t> next_fresh_;
    std::vector<std::vector<std::uint32_t>> free_slots_;

    Heap<EvictBefore> evict_heap_;
    Heap<QueueBefore> queue_;
    std::vector<Pending> pending_;
    std::vector<ExpertKey> scratch_;
    std::vector<std::uint8_t> mark_;
    bool too_many_ = false;  // last pick_victims failure was the kMaxVictims limit
    std::uint32_t n_pending_ = 0;
    std::uint64_t next_id_ = 1;
    std::uint64_t seq_ctr_ = 0;
    std::uint64_t cur_token_ = 0;
    std::uint64_t token_issued_bytes_ = 0;

    std::uint64_t resident_bytes_ = 0;
    std::uint64_t victim_bytes_ = 0;
    std::uint64_t planned_bytes_ = 0;
    std::uint64_t inflight_bytes_ = 0;
    std::uint64_t pinned_bytes_ = 0;
    std::uint32_t resident_count_ = 0;

    bool fail(int line) const noexcept;
    mutable int failed_line_ = 0;

    Counters total_;
    std::vector<Counters> layer_;
};

}  // namespace eq::pager
