#include "pager.cuh"
#include "mmvq.cuh"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(__x86_64__)
#define PAGER_CPU_RELAX() __asm__ __volatile__("pause" ::: "memory")
#else
#define PAGER_CPU_RELAX() do {} while (0)
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Slot lifetime (stage 2). Read this before changing anything below.
//
// Actors: the compute thread (runs the backend graph, owns the cache and every decision), the copier thread (moves bytes on its
// own CUDA stream, never touches the cache), the compute stream. A slot is a row of the three pools; slot id s is pool row s+1
// (row 0 is the all-zero null slot read by misses). Per slot state machine (debug builds assert every transition):
//
//   FREE --issue--> COPYING --commit--> READY --claimed by an issued promotion--> VICTIM --commit (evicted)--> FREE
//                      |                  ^                                          |
//                      +--abort-----> FREE  +------ victim touched before commit -----+   (VICTIM -> READY)
//
// Only READY and VICTIM slots are ever handed to a kernel (a kernel gets slot ids only through cache->lookup(), and the cache
// returns a slot only for a key it made resident at commit()). A VICTIM is still resident and usable until its commit.
//
// (a) A destination slot is never read before its copy completed and was committed. The cache maps the key to the slot only
//     inside commit(); the compute thread calls commit(id) only after it popped id from the copier's done list, which the
//     copier pushes after cudaEventSynchronize() on the event recorded behind the last memcpy of that promotion (sync-copy
//     mode: after the memcpy was queued on the compute stream, committed one block later). Device writes that completed
//     before a host-visible event are visible to every kernel launched afterwards. Until then lookup() misses.
//
// (b) A slot freed by a commit cannot be read by a kernel queued before the commit, and cannot be overwritten while such a
//     kernel runs. Every block executes, in this order: [A] cudaStreamSynchronize on the compute stream; [B] commit of the
//     completed promotions (the only place slots are freed); [C] lookups, which build the ids handed to this block's kernels;
//     [D] kernel launches; [E] next_promotions(), which may reuse a slot freed in [B] as a destination.
//     * Every kernel that could read a slot freed in [B] was queued by an earlier block (a kernel reads only slots lookup()
//       returned in [C] of its own block, and a freed slot is no longer returned). By [A] all of those kernels have
//       finished, so the freed slot has no reader left. The debug build records, per slot, the epoch (block counter, advanced
//       at [A]) of the last launch that read it and asserts last_read_epoch < epoch when the slot is freed.
//     * The write into a reused slot starts at [E] or later on the copier stream, i.e. after [B]; the kernels queued at [D]
//       of this block do not read it (it was just freed, so [C] cannot have returned it). The next block's kernels read it
//       only after its commit, which needs the copy to have completed (a).
//     * A second reuse hazard: a promotion issued at [E] of block k and committed at [B] of block k+j. Its destination was
//       FREE at issue (asserted), so no VICTIM/READY owner existed.
//     So a slot freed at commit time in block k is never read after [A] of block k, and never written before [B] of block k.
//     No extra epoch or event is needed because [A] precedes [B] in every block and nothing frees slots elsewhere.
//     This argument relies on one thing: slots are freed ONLY at [B]. set_budget() would break it (see below).
//
// (c) kAborted: the cache frees the destination at commit; its copy is complete (we only commit completed copies), nothing
//     ever mapped it, so it simply becomes FREE. kStale cannot happen (no set_budget/reset); it aborts.
//
// Not supported yet: shrinking the budget at run time. set_budget() cancels every in-flight promotion and frees their
// destinations immediately, while the copier may still be writing them. It would need: drain the copier (wait for every
// in-flight id and drop it) before calling set_budget(), and issue nothing until the drain is done; plus an assertion that no
// kernel queued before it reads an evicted slot, which [A] already gives if the call is made in [B].
//
// Poison (debug builds): a slot is overwritten with a recognisable pattern when it becomes FREE and when it is issued as a
// destination, both on the copier stream (same stream as the later copy, so the order is exact). A kernel reading a slot it
// should not then shows up as a large perplexity change.
// ---------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------
// globals: injected CPU kernels and process-wide telemetry
// ---------------------------------------------------------------------------------------------------------------------

static std::atomic<ggml_get_type_traits_cpu_t> g_cpu_traits{nullptr};
static std::atomic<ggml_vec_swiglu_f32_t>      g_cpu_swiglu{nullptr};

struct pager_global_stats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> resident{0};
    std::atomic<uint64_t> pool_bytes{0};
    std::atomic<uint64_t> bytes_copied{0};
    std::atomic<uint64_t> cpu_us{0};
    std::atomic<uint64_t> sync_us{0};
    std::atomic<uint64_t> host_us{0};
    std::atomic<uint64_t> syncs{0};
    std::atomic<uint64_t> blocks{0};
    std::atomic<uint64_t> promo_issued{0};
    std::atomic<uint64_t> promo_committed{0};
    std::atomic<uint64_t> promo_aborted{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> admission_rejected{0};
    std::atomic<uint64_t> queue_depth{0};
    std::atomic<uint64_t> copy_busy_us{0};
    std::atomic<uint64_t> copy_bytes{0};
    std::atomic<uint64_t> promo_lat_us{0};   // sum over committed or aborted promotions of issue to commit time
    std::atomic<uint64_t> promo_lat_tok{0};  // ... and of the tokens elapsed
    std::atomic<uint64_t> bw_mbps{0};   // measured (or configured) host to device bandwidth, bytes per microsecond
};
static pager_global_stats g_stats;

void ggml_cuda_pager_set_cpu_kernels(const ggml_pager_cpu_kernels * kernels) {
    g_cpu_swiglu.store(kernels->vec_swiglu_f32, std::memory_order_release);
    g_cpu_traits.store(kernels->get_type_traits_cpu, std::memory_order_release); // last: supports_op keys on it
}

void ggml_cuda_pager_get_stats(ggml_pager_stats * out) {
    out->hits             = g_stats.hits.load(std::memory_order_relaxed);
    out->misses           = g_stats.misses.load(std::memory_order_relaxed);
    out->resident_experts = g_stats.resident.load(std::memory_order_relaxed);
    out->pool_bytes       = g_stats.pool_bytes.load(std::memory_order_relaxed);
    out->bytes_copied     = g_stats.bytes_copied.load(std::memory_order_relaxed);
    out->cpu_ms           = g_stats.cpu_us.load(std::memory_order_relaxed) / 1000.0;
    out->sync_ms          = g_stats.sync_us.load(std::memory_order_relaxed) / 1000.0;
    out->host_ms          = g_stats.host_us.load(std::memory_order_relaxed) / 1000.0;
    out->syncs            = g_stats.syncs.load(std::memory_order_relaxed);
    out->blocks           = g_stats.blocks.load(std::memory_order_relaxed);
    out->promotions_issued    = g_stats.promo_issued.load(std::memory_order_relaxed);
    out->promotions_committed = g_stats.promo_committed.load(std::memory_order_relaxed);
    out->promotions_aborted   = g_stats.promo_aborted.load(std::memory_order_relaxed);
    out->evictions            = g_stats.evictions.load(std::memory_order_relaxed);
    out->admission_rejected   = g_stats.admission_rejected.load(std::memory_order_relaxed);
    out->queue_depth          = g_stats.queue_depth.load(std::memory_order_relaxed);
    const uint64_t busy = g_stats.copy_busy_us.load(std::memory_order_relaxed);
    out->copy_gbps            = busy ? (double) g_stats.copy_bytes.load(std::memory_order_relaxed) / (double) busy / 1e3 : 0.0;
    out->copy_busy_ms         = busy / 1000.0;
    out->promo_latency_ms_sum  = g_stats.promo_lat_us.load(std::memory_order_relaxed) / 1000.0;
    out->promo_latency_tokens_sum = g_stats.promo_lat_tok.load(std::memory_order_relaxed);
    out->bandwidth_gbps       = g_stats.bw_mbps.load(std::memory_order_relaxed) / 1e3;
}

static const ggml_type_traits_cpu * pager_cpu_traits(ggml_type type) {
    ggml_get_type_traits_cpu_t fn = g_cpu_traits.load(std::memory_order_acquire);
    return fn ? fn(type) : nullptr;
}

size_t ggml_cuda_pager_budget_bytes() {
    const char * env = getenv("GGML_CUDA_PAGER_MIB");
    if (!env) {
        return 0;
    }
    const long long mib = atoll(env);
    return mib > 0 ? (size_t) mib * 1024 * 1024 : 0;
}

static int pager_n_threads() {
    const char * env = getenv("GGML_CUDA_PAGER_THREADS");
    if (env && atoi(env) > 0) {
        return atoi(env);
    }
    return std::max(1, (int) std::thread::hardware_concurrency() / 2);
}

static uint64_t pager_now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------------------------------
// worker pool: persistent threads, one job at a time, the submitting thread helps while it waits
// ---------------------------------------------------------------------------------------------------------------------

class pager_workers {
public:
    explicit pager_workers(int n_threads) {
        for (int i = 1; i < n_threads; i++) {
            threads_.emplace_back([this] { loop(); });
        }
    }

    ~pager_workers() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto & t : threads_) {
            t.join();
        }
    }

    // returns immediately; workers start on the items
    void submit(int n_items, std::function<void(int)> fn) {
        GGML_ASSERT(!job_active_);
        if (n_items <= 0) {
            return;
        }
        auto j = std::make_shared<job>();
        j->fn = std::move(fn);
        j->n  = n_items;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            job_ = j;
            gen_.fetch_add(1, std::memory_order_release);
        }
        cur_ = j;
        job_active_ = true;
        cv_work_.notify_all();
    }

    // the calling thread processes items too; returns when every item is done
    void join() {
        if (!job_active_) {
            return;
        }
        std::shared_ptr<job> j = std::move(cur_);
        job_active_ = false;
        run_items(*j);
        for (int spin = 0; spin < 4000 && j->done.load(std::memory_order_acquire) != j->n; spin++) {
            PAGER_CPU_RELAX();
        }
        if (j->done.load(std::memory_order_acquire) != j->n) {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_done_.wait(lk, [&] { return j->done.load(std::memory_order_acquire) == j->n; });
        }
    }

private:
    struct job {
        std::function<void(int)> fn;
        int                      n = 0;
        std::atomic<int>         next{0};
        std::atomic<int>         done{0};
    };

    void run_items(job & j) {
        for (;;) {
            const int i = j.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= j.n) {
                break;
            }
            j.fn(i);
            if (j.done.fetch_add(1, std::memory_order_acq_rel) + 1 == j.n) {
                std::lock_guard<std::mutex> lk(mtx_);
                cv_done_.notify_all();
            }
        }
    }

    void loop() {
        uint64_t seen = 0;
        for (;;) {
            // spin briefly so back-to-back layers do not pay a futex wake each time, then block
            const uint64_t t0 = pager_now_us();
            while (gen_.load(std::memory_order_acquire) == seen && pager_now_us() - t0 < 300) {
                PAGER_CPU_RELAX();
            }
            std::shared_ptr<job> j;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_work_.wait(lk, [&] { return stop_ || gen_.load(std::memory_order_acquire) != seen; });
                if (stop_) {
                    return;
                }
                seen = gen_.load(std::memory_order_acquire);
                j    = job_;
            }
            run_items(*j);
        }
    }

    std::vector<std::thread> threads_;
    std::mutex               mtx_;
    std::condition_variable  cv_work_;
    std::condition_variable  cv_done_;
    std::shared_ptr<job>     job_;          // guarded by mtx_
    std::atomic<uint64_t>    gen_{0};
    bool                     stop_ = false; // guarded by mtx_
    // compute thread only
    std::shared_ptr<job>     cur_;
    bool                     job_active_ = false;
};

// ---------------------------------------------------------------------------------------------------------------------
// pinned host staging buffer (compute thread only)
// ---------------------------------------------------------------------------------------------------------------------

struct pager_host_buf {
    void * ptr    = nullptr;
    size_t cap    = 0;
    bool   pinned = false;

    ~pager_host_buf() { release(); }

    void release() {
        if (ptr) {
            if (pinned) {
                CUDA_CHECK(cudaFreeHost(ptr));
            } else {
                free(ptr);
            }
        }
        ptr = nullptr;
        cap = 0;
    }

    // never shrinks; the caller guarantees no queued copy still uses the old storage
    void * ensure(size_t size) {
        if (size <= cap) {
            return ptr;
        }
        release();
        size = std::max<size_t>(size, 4096);
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err == cudaSuccess) {
            pinned = true;
        } else {
            (void) cudaGetLastError();
            ptr    = malloc(size);
            pinned = false;
            GGML_ASSERT(ptr != nullptr);
        }
        cap = size;
        return ptr;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// pager state
// ---------------------------------------------------------------------------------------------------------------------

struct block_match {
    bool          fused = false;
    ggml_tensor * gate = nullptr;    // MUL_MAT_ID nodes
    ggml_tensor * up   = nullptr;
    ggml_tensor * glu  = nullptr;
    ggml_tensor * down = nullptr;
};

struct mv_task {
    const char *    w;
    size_t          row_bytes;
    const void *    xq;
    float *         out;
    int             nrows;
    int             n;
    ggml_vec_dot_t  dot;
};

struct pager_group {
    int                 n = 0;       // 1 (single MUL_MAT_ID, never cached) or 3 (gate, up, down)
    const ggml_tensor * w[3] = {};
    int                 n_expert = 0;
    bool                cached = false;  // fused group whose experts have a slot in the pools
    bool                pinned = true;   // the host copy of the weights is page-locked
};

// ---- copy path ---------------------------------------------------------------------------------------------------------

struct copy_job {
    uint64_t     id = 0;
    int          op = 0;          // 0: copy members, 1: poison only
    int          n = 0;
    const char * src[3] = {};
    char *       dst[3] = {};
    size_t       sz[3] = {};      // bytes to copy per member
    size_t       span[3] = {};    // bytes of the slot row per member (poison)
    bool         pinned = true;
    bool         poison_first = false;
};

// bounce buffers for a pageable source: copied through page-locked memory, two in flight
struct pager_bounce {
    pager_host_buf buf[2];
    cudaEvent_t    ev[2] = {};
    bool           used[2] = {false, false};
    unsigned       next = 0;

    ~pager_bounce() {
        for (cudaEvent_t e : ev) {
            if (e) {
                CUDA_CHECK(cudaEventDestroy(e));
            }
        }
    }
};

static void pager_do_copy(const copy_job & j, cudaStream_t s, pager_bounce & b) {
    if (j.op == 1 || j.poison_first) {
        for (int k = 0; k < j.n; k++) {
            CUDA_CHECK(cudaMemsetAsync(j.dst[k], 0x7B, j.span[k], s));
        }
    }
    if (j.op == 1) {
        return;
    }
    for (int k = 0; k < j.n; k++) {
        if (j.pinned) {
            CUDA_CHECK(cudaMemcpyAsync(j.dst[k], j.src[k], j.sz[k], cudaMemcpyHostToDevice, s));
        } else {
            const unsigned i = b.next++ & 1;
            if (!b.ev[i]) {
                CUDA_CHECK(cudaEventCreateWithFlags(&b.ev[i], cudaEventDisableTiming));
            }
            if (b.used[i]) {
                CUDA_CHECK(cudaEventSynchronize(b.ev[i]));
            }
            void * h = b.buf[i].ensure(j.sz[k]);
            memcpy(h, j.src[k], j.sz[k]);
            CUDA_CHECK(cudaMemcpyAsync(j.dst[k], h, j.sz[k], cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaEventRecord(b.ev[i], s));
            b.used[i] = true;
        }
    }
}

// The copier thread: executes copy jobs on its own stream and reports completed promotion ids. It owns no cache state.
// Copies are queued ahead without waiting (one event per job, polled in submission order), so the stream stays busy while
// earlier promotions are still being reported; a job is reported only after the event behind its last memcpy completed.
class pager_copier {
public:
    pager_copier(int device, size_t depth) : device_(device), depth_(std::max<size_t>(1, depth)) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        for (int i = 0; i < kEvents; i++) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            free_ev_.push_back(e);
        }
        thread_ = std::thread([this] { loop(); });
    }

    ~pager_copier() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
        ggml_cuda_set_device(device_);
        for (cudaEvent_t e : free_ev_) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    void submit(const copy_job & j) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            q_.push_back(j);
        }
        cv_.notify_one();
    }

    // moves the ids of completed copies into `out`
    void drain_done(std::vector<uint64_t> & out) {
        std::lock_guard<std::mutex> lk(mtx_);
        out.insert(out.end(), done_.begin(), done_.end());
        done_.clear();
    }

    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> busy_us{0};

private:
    static constexpr int kEvents = 64;

    struct outstanding {
        cudaEvent_t ev;
        uint64_t    id;
        size_t      bytes;
    };

    void report(const outstanding & o) {
        bytes.fetch_add(o.bytes, std::memory_order_relaxed);
        free_ev_.push_back(o.ev);
        std::lock_guard<std::mutex> lk(mtx_);
        done_.push_back(o.id);
    }

    void reap(bool block) {
        while (!out_.empty()) {
            const outstanding o = out_.front();
            if (block) {
                CUDA_CHECK(cudaEventSynchronize(o.ev));
            } else if (cudaEventQuery(o.ev) != cudaSuccess) {
                (void) cudaGetLastError();
                return;
            }
            out_.pop_front();
            report(o);
            block = false;
        }
    }

    void loop() {
        ggml_cuda_set_device(device_);
        pager_bounce bounce;
        uint64_t mark = 0; // start of the current busy interval, 0 while idle
        for (;;) {
            copy_job j;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                if (q_.empty() && out_.empty()) {
                    if (mark) {
                        busy_us.fetch_add(pager_now_us() - mark, std::memory_order_relaxed);
                        mark = 0;
                    }
                    cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                    if (q_.empty()) {
                        return; // stop requested and nothing left
                    }
                }
                if (!q_.empty() && out_.size() < depth_) {
                    j = q_.front();
                    q_.pop_front();
                    have = true;
                }
            }
            const uint64_t t_now = pager_now_us();
            if (mark) {
                busy_us.fetch_add(t_now - mark, std::memory_order_relaxed);
            }
            mark = t_now;
            if (have) {
                if (j.op == 0 && free_ev_.empty()) {
                    reap(true); // out of events: wait for the oldest job
                }
                pager_do_copy(j, stream_, bounce);
                if (j.op == 0) {
                    size_t total = 0;
                    for (int k = 0; k < j.n; k++) {
                        total += j.sz[k];
                    }
                    cudaEvent_t e = free_ev_.back();
                    free_ev_.pop_back();
                    CUDA_CHECK(cudaEventRecord(e, stream_));
                    out_.push_back({ e, j.id, total });
                }
                reap(false);
            } else {
                reap(true); // nothing new to submit: wait for the oldest copy
            }
        }
    }

    int                     device_;
    size_t                  depth_;   // promotions queued on the copier stream at once
    cudaStream_t            stream_ = nullptr;
    std::thread             thread_;
    std::mutex              mtx_;
    std::condition_variable cv_;
    std::deque<copy_job>    q_;       // guarded by mtx_
    std::vector<uint64_t>   done_;    // guarded by mtx_
    bool                    stop_ = false;
    // copier thread only
    std::vector<cudaEvent_t> free_ev_;
    std::deque<outstanding>  out_;
};

// ---- configuration -----------------------------------------------------------------------------------------------------

static double pager_env_double(const char * name, double def) {
    const char * e = getenv(name);
    return e && *e ? atof(e) : def;
}

static bool pager_env_flag(const char * name) {
    const char * e = getenv(name);
    return e && atoi(e) != 0;
}

enum pager_policy { POLICY_LRU = 0, POLICY_LFU = 1, POLICY_FWLRU = 2, POLICY_STATIC = 3 };

static pager_policy pager_policy_from_env() {
    const char * e = getenv("GGML_CUDA_PAGER_POLICY");
    if (!e || !*e || strcmp(e, "lru") == 0) {
        return POLICY_LRU;
    }
    if (strcmp(e, "lfu") == 0) {
        return POLICY_LFU;
    }
    if (strcmp(e, "fwlru") == 0) {
        return POLICY_FWLRU;
    }
    if (strcmp(e, "static") == 0) {
        return POLICY_STATIC;
    }
    GGML_LOG_WARN("%s: unknown GGML_CUDA_PAGER_POLICY '%s', using lru\n", "pager", e);
    return POLICY_LRU;
}

static uint64_t pager_lcm(uint64_t a, uint64_t b) {
    uint64_t x = a, y = b;
    while (y) {
        const uint64_t t = x % y;
        x = y;
        y = t;
    }
    return a / x * b;
}

// ---- the pager -----------------------------------------------------------------------------------------------------------

enum slot_state : uint8_t { SLOT_FREE = 0, SLOT_COPYING = 1, SLOT_READY = 2, SLOT_VICTIM = 3 };

struct inflight_promo {
    uint64_t issue_us = 0;
    uint64_t issue_tok = 0;
    uint32_t key = 0;
    uint32_t dest = 0;
    uint32_t n_victims = 0;
    uint32_t victim_slots[8] = {};
};

struct ggml_cuda_pager {
    int    device;
    size_t budget;           // GGML_CUDA_PAGER_MIB in bytes (pool + staging + null slot)

    pager_workers            workers;
    std::vector<pager_group> groups;       // index == cache layer
    std::unordered_map<const ggml_tensor *, int> group_of;
    bool   pools_built = false;
    bool   scanned     = false;

    // pool geometry: three role pools sharing the slot index; row 0 of each is the null slot
    std::unique_ptr<ggml_cuda_pager_cache> cache;
    std::unique_ptr<pager_copier>          copier;   // null in sync-copy mode and without a pool
    pager_bounce                           sync_bounce;
    void *   pool[3] = {};
    size_t   stride[3] = {};     // bytes of one slot row per role
    size_t   pool_bytes[3] = {};
    uint32_t n_slots = 0;        // usable slots (slot ids 0 .. n_slots-1)
    uint32_t epl = 0;            // experts per layer of the cache key space
    size_t   slot_bytes = 0;     // sum of the strides, the cache's expert size
    pager_policy policy = POLICY_LRU;
    bool     sync_copy = false;

    // promotions
    std::unordered_map<uint64_t, inflight_promo> inflight;
    std::vector<uint64_t> sync_done;
    size_t   inflight_bytes = 0;
    double   bw_bytes_per_us = 0.0;
    double   credit = 0.0;
    uint64_t last_issue_us = 0;
    double   tok_ema_us = 20000.0;
    uint64_t last_tok_us = 0;
    uint64_t tok = 0;            // token counter, see exec()
    int      last_gid = -1;
    uint64_t epoch = 0;          // blocks started (advanced right after the stream synchronisation)
    cudaStream_t cur_stream = nullptr; // compute stream of the running block (inline jobs)
#ifndef NDEBUG
    bool test_early_commit = pager_env_flag("GGML_CUDA_PAGER_TEST_EARLY_COMMIT"); // debug builds only, see run_job()
    std::unordered_set<uint64_t> early_ids;
    // negative control 2: serve a miss from the slot its key was evicted from (a read of a FREE slot). 1: the state machine
    // assertion fires; 2: assertions skipped, the poison pattern is read and the perplexity must jump.
    int test_stale_read = (int) pager_env_double("GGML_CUDA_PAGER_TEST_STALE_READ", 0);
    std::unordered_map<uint32_t, uint32_t> stale_slot;
#endif

    // debug state machine
    std::vector<uint8_t>  slot_st;
    std::vector<int64_t>  slot_key;
    std::vector<uint8_t>  slot_claims;
    std::vector<uint64_t> slot_last_read;

    pager_host_buf h_ids, h_x, h_remap, h_blob;

    // published deltas
    uint64_t reported_resident = 0, reported_pool_bytes = 0;
    ggml_cuda_pager_cache_counters pub = {};
    uint64_t pub_copied = 0, pub_busy = 0;
    // this pager's own totals (logged when it is destroyed); expected = sum over blocks of n_used * T
    uint64_t n_hits = 0, n_misses = 0, n_blocks = 0, n_expected = 0;

    explicit ggml_cuda_pager(int device) :
        device(device), budget(ggml_cuda_pager_budget_bytes()), workers(pager_n_threads()) {
        policy    = pager_policy_from_env();
        sync_copy = pager_env_flag("GGML_CUDA_PAGER_SYNC_COPY");
        static const char * names[] = { "lru", "lfu", "fwlru", "static" };
        GGML_LOG_INFO("%s: device %d, pool %.0f MiB, policy %s%s, %d CPU threads\n", "pager", device, budget / 1048576.0,
                      names[policy], sync_copy ? " (synchronous copies)" : "", pager_n_threads());
    }

    ~ggml_cuda_pager() {
        // the worker threads are joined by ~pager_workers after this body; they only run inside exec(), which has returned
        ggml_cuda_set_device(device);
        copier.reset(); // joins the copier thread after its queue is empty
        for (int k = 0; k < 3; k++) {
            if (pool[k]) {
                CUDA_CHECK(cudaFree(pool[k]));
            }
        }
        if (n_blocks > 0) {
            const ggml_cuda_pager_cache_counters c = cache ? cache->counters() : ggml_cuda_pager_cache_counters{};
            GGML_LOG_INFO("%s: device %d totals: %llu blocks, hits %llu + misses %llu = %llu (expert evaluations expected: %llu), "
                          "promotions %llu issued / %llu committed / %llu aborted, evictions %llu, resident %u, token counter %llu\n",
                          "pager", device, (unsigned long long) n_blocks, (unsigned long long) n_hits, (unsigned long long) n_misses,
                          (unsigned long long) (n_hits + n_misses), (unsigned long long) n_expected, (unsigned long long) c.issued,
                          (unsigned long long) c.promotions, (unsigned long long) c.aborted, (unsigned long long) c.evictions, c.resident, (unsigned long long) tok);
        }
        g_stats.resident.fetch_sub(reported_resident);
        g_stats.pool_bytes.fetch_sub(reported_pool_bytes);
    }

    // -----------------------------------------------------------------------------------------------------------------
    // graph matching
    // -----------------------------------------------------------------------------------------------------------------

    bool match_block(const ggml_cgraph * cgraph, int i, block_match & m) const {
        if (getenv("GGML_CUDA_PAGER_NO_FUSE")) {
            return false;
        }
        if (i + 3 >= cgraph->n_nodes) {
            return false;
        }
        ggml_tensor * n0  = cgraph->nodes[i];
        ggml_tensor * n1  = cgraph->nodes[i + 1];
        ggml_tensor * glu = cgraph->nodes[i + 2];
        ggml_tensor * dn  = cgraph->nodes[i + 3];
        if (n0->op != GGML_OP_MUL_MAT_ID || n1->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU || dn->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU, GGML_OP_MUL_MAT_ID }, { i + 3 })) {
            return false;
        }
        ggml_tensor * gate = glu->src[0];
        ggml_tensor * up   = glu->src[1];
        if (!((gate == n0 && up == n1) || (gate == n1 && up == n0))) {
            return false;
        }
        const int cc = ggml_cuda_info().devices[device].cc;
        if (cc <= GGML_CUDA_CC_PASCAL) {
            return false; // the fused kernel is not used on Pascal
        }
        // only silu(gate) * up is implemented by the CPU chain
        if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) {
            return false;
        }
        if (!ggml_cuda_should_fuse_mul_mat_for_pager(up, gate, glu)) {
            return false;
        }
        if (dn->src[1] != glu || dn->src[2] != up->src[2]) {
            return false;
        }
        for (const ggml_tensor * t : { gate->src[0], up->src[0], dn->src[0] }) {
            if (ggml_cuda_tensor_paged_device(t) != device) {
                return false;
            }
        }
        if (up->src[1]->ne[1] != 1 || up->src[1]->type != GGML_TYPE_F32 || dn->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
            return false;
        }
        // no memory-range check as in the stock fusions: the block reads x and ids only before its first kernel (host copies, and
        // the kernel quantises x into scratch before writing), so the output of any node may alias them
        m.fused = true;
        m.gate  = gate;
        m.up    = up;
        m.glu   = glu;
        m.down  = dn;
        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // groups and pools
    // -----------------------------------------------------------------------------------------------------------------

    // register the weights of a block (idempotent); the cache layer index equals the group index
    int register_group(const ggml_tensor * const * w, int n) {
        auto it = group_of.find(w[0]);
        if (it != group_of.end()) {
            return it->second;
        }
        pager_group g;
        g.n = n;
        for (int k = 0; k < n; k++) {
            g.w[k] = w[k];
        }
        g.n_expert = (int) w[0]->ne[2];
        g.cached   = false; // decided in build_pools; groups seen later stay uncached (all-miss)
        cudaPointerAttributes attr = {};
        if (cudaPointerGetAttributes(&attr, w[0]->data) == cudaSuccess) {
            g.pinned = attr.type == cudaMemoryTypeHost;
        } else {
            (void) cudaGetLastError();
            g.pinned = false;
        }
        groups.push_back(g);
        group_of[w[0]] = (int) groups.size() - 1;
        return (int) groups.size() - 1;
    }

    static size_t tail_padding(const ggml_tensor * w) {
        return w->ne[0] % MATRIX_ROW_PADDING != 0 ? ggml_row_size(w->type, MATRIX_ROW_PADDING - w->ne[0] % MATRIX_ROW_PADDING) : 0;
    }

    void free_pools() {
        for (int k = 0; k < 3; k++) {
            if (pool[k]) {
                CUDA_CHECK(cudaFree(pool[k]));
                pool[k] = nullptr;
            }
            pool_bytes[k] = 0;
        }
        cache.reset();
        n_slots = 0;
        for (pager_group & g : groups) {
            g.cached = false;
        }
    }

    // Called once, after the first graph has been scanned. Experts of every fused group get one common slot size (the sum
    // of per-role strides, each stride a multiple of the block sizes of all types seen in that role), so the cache has a
    // single size class; the device pools are budget + staging in total. Anything that does not fit stays uncached.
    void build_pools(cudaStream_t stream) {
        pools_built = true;
        ggml_cuda_set_device(device);

        uint64_t unit[3] = { 16, 16, 16 };
        size_t   maxb[3] = {}, tail[3] = {};
        int      n_cached = 0;
        epl = 0;
        for (const pager_group & g : groups) {
            epl = std::max<uint32_t>(epl, g.n_expert);
            if (g.n != 3) {
                continue;
            }
            n_cached++;
            for (int k = 0; k < 3; k++) {
                unit[k] = pager_lcm(unit[k], ggml_type_size(g.w[k]->type));
                maxb[k] = std::max<size_t>(maxb[k], g.w[k]->nb[2]);
                tail[k] = std::max<size_t>(tail[k], tail_padding(g.w[k]));
            }
        }
        if (n_cached == 0 || budget == 0) {
            return;
        }
        slot_bytes = 0;
        for (int k = 0; k < 3; k++) {
            stride[k] = (maxb[k] + unit[k] - 1) / unit[k] * unit[k];
            slot_bytes += stride[k];
        }
        size_t unpadded = 0, worst = 0;
        for (const pager_group & g : groups) {
            if (g.n == 3) {
                size_t b = 0;
                for (int k = 0; k < 3; k++) {
                    b += g.w[k]->nb[2];
                }
                unpadded += b;
                worst = std::max(worst, b);
            }
        }
        const size_t phys_slots = budget / slot_bytes;       // including the null slot
        size_t staging = (size_t) (pager_env_double("GGML_CUDA_PAGER_STAGING_MIB", 64.0) * 1048576.0);
        staging = std::min(staging, budget / 4);
        staging = std::max<size_t>(staging / slot_bytes, 2) * slot_bytes;
        if (phys_slots < 4 || (phys_slots - 1) * slot_bytes <= staging + slot_bytes) {
            GGML_LOG_WARN("%s: pool of %.0f MiB holds fewer than %d slots of %.2f MiB, running without a pool\n", "pager",
                          budget / 1048576.0, 4, slot_bytes / 1048576.0);
            return;
        }
        ggml_cuda_pager_cache_config cc;
        cc.n_layers          = (uint32_t) groups.size();
        cc.experts_per_layer = epl;
        cc.bytes_per_expert  = (uint32_t) slot_bytes;
        cc.staging_bytes     = staging;
        cc.budget_bytes      = (phys_slots - 1) * slot_bytes - staging;
        cc.policy            = policy == POLICY_STATIC ? POLICY_LRU : (int) policy;
        cc.half_life_tokens  = pager_env_double("GGML_CUDA_PAGER_HALFLIFE", 64.0);
        cc.admit_min_misses  = (uint32_t) std::max(1.0, pager_env_double("GGML_CUDA_PAGER_ADMIT_MISSES", 2.0));
        cc.admit_window_tokens = (uint64_t) std::max(0.0, pager_env_double("GGML_CUDA_PAGER_ADMIT_WINDOW", 16.0));
        cc.max_bytes_per_token = (uint64_t) (std::max(0.0, pager_env_double("GGML_CUDA_PAGER_TOKEN_MIB", 0.0)) * 1048576.0);
        try {
            cache = std::make_unique<ggml_cuda_pager_cache>(cc);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: cache configuration rejected: %s; running without a pool\n", "pager", e.what());
            return;
        }
        n_slots = cache->max_slots();
        GGML_ASSERT(n_slots + 1 <= phys_slots);
        for (int k = 0; k < 3; k++) {
            pool_bytes[k] = stride[k] * (size_t) (n_slots + 1) + tail[k];
            cudaError_t err = cudaMalloc(&pool[k], pool_bytes[k]);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("%s: pool allocation of %.1f MiB failed: %s; running without a pool\n", __func__,
                               pool_bytes[k] / 1048576.0, cudaGetErrorString(err));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                free_pools();
                return;
            }
            CUDA_CHECK(cudaMemsetAsync(pool[k], 0, pool_bytes[k], stream)); // null slot and row padding stay zero
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (pager_group & g : groups) {
            g.cached = g.n == 3;
        }
        uint64_t total = 0;
        for (int k = 0; k < 3; k++) {
            total += pool_bytes[k];
        }
        reported_pool_bytes = total;
        g_stats.pool_bytes.fetch_add(total);

        slot_st.assign(n_slots, SLOT_FREE);
        slot_key.assign(n_slots, -1);
        slot_claims.assign(n_slots, 0);
        slot_last_read.assign(n_slots, 0);

        // host to device bandwidth: measured once through the real copy path, or GGML_CUDA_PAGER_GBPS
        const double gbps_env = pager_env_double("GGML_CUDA_PAGER_GBPS", 0.0);
        bool any_pinned = false, any_pageable = false;
        for (const pager_group & g : groups) {
            if (g.n == 3) {
                (g.pinned ? any_pinned : any_pageable) = true;
            }
        }
        if (gbps_env > 0.0) {
            bw_bytes_per_us = gbps_env * 1e3;
        } else {
            bw_bytes_per_us = measure_bandwidth(stream);
        }
        g_stats.bw_mbps.store((uint64_t) (bw_bytes_per_us), std::memory_order_relaxed);
        if (!sync_copy) {
            copier = std::make_unique<pager_copier>(device, (size_t) pager_env_double("GGML_CUDA_PAGER_COPY_DEPTH", 1.0));
        }
        last_issue_us = pager_now_us();
        credit = 0.0;
        GGML_LOG_INFO("%s: %u slots of %.2f MiB (%d layers x %u experts, strides %.2f/%.2f/%.2f MiB; unpadded mean %.2f MiB, largest %.2f MiB), "
                      "budget %.0f MiB + staging %.0f MiB, pool %.0f MiB; host weights %s; copy bandwidth %.1f GB/s%s\n", "pager",
                      n_slots, slot_bytes / 1048576.0, n_cached, epl, stride[0] / 1048576.0, stride[1] / 1048576.0, stride[2] / 1048576.0,
                      unpadded / n_cached / 1048576.0, worst / 1048576.0, cc.budget_bytes / 1048576.0, staging / 1048576.0, total / 1048576.0,
                      any_pageable ? (any_pinned ? "partly pageable" : "pageable (bounce buffers)") : "pinned", bw_bytes_per_us / 1e3,
                      gbps_env > 0.0 ? " (GGML_CUDA_PAGER_GBPS)" : " (measured)");
    }

    // bytes per microsecond of the copy path, median of three 48 MiB copies of the first group's expert weights
    double measure_bandwidth(cudaStream_t stream) {
        const pager_group * g = nullptr;
        for (const pager_group & c : groups) {
            if (c.n == 3) {
                g = &c;
                break;
            }
        }
        const size_t bytes = std::min<size_t>(48u << 20, std::min<size_t>(g->w[0]->nb[2] * g->n_expert, pool_bytes[0] - tail_padding(g->w[0])));
        const size_t nexp  = std::max<size_t>(1, bytes / g->w[0]->nb[2]);
        copy_job j;
        j.n = 1;
        j.pinned = g->pinned;
        double t[3];
        for (int r = 0; r < 3; r++) {
            const uint64_t t0 = pager_now_us();
            size_t total = 0;
            for (size_t e = 0; e < nexp; e++) {
                j.src[0] = (const char *) g->w[0]->data + e * g->w[0]->nb[2];
                j.dst[0] = (char *) pool[0] + stride[0]; // slot 0, overwritten before any use
                j.sz[0]  = g->w[0]->nb[2];
                pager_do_copy(j, stream, sync_bounce);
                total += j.sz[0];
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t[r] = (double) total / std::max<double>(1.0, (double) (pager_now_us() - t0));
        }
        std::sort(t, t + 3);
        CUDA_CHECK(cudaMemsetAsync(pool[0], 0, stride[0] * 2, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return t[1];
    }

    void scan(ggml_cgraph * cgraph, cudaStream_t stream) {
        scanned = true;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_tensor_is_paged(node->src[0])) {
                continue;
            }
            block_match m;
            if (match_block(cgraph, i, m)) {
                const ggml_tensor * w[3] = { m.gate->src[0], m.up->src[0], m.down->src[0] };
                register_group(w, 3);
                i += 3;
            } else {
                const ggml_tensor * w[1] = { node->src[0] };
                register_group(w, 1);
            }
        }
        if (!pools_built) {
            build_pools(stream);
            prefill(stream);
        }
    }

    // test aid: GGML_CUDA_PAGER_PREFILL=1 makes the first experts (graph order) resident before the first block, up to the budget
    void prefill(cudaStream_t stream) {
        if (!cache || !pager_env_flag("GGML_CUDA_PAGER_PREFILL")) {
            return;
        }
        uint64_t n = 0;
        for (size_t g = 0; g < groups.size(); g++) {
            for (int e = 0; groups[g].cached && e < groups[g].n_expert; e++) {
                const uint32_t key = (uint32_t) (g * epl + e);
                if (cache->install(key)) {
                    install_copy(key, stream);
                    n++;
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        GGML_LOG_INFO("%s: prefilled %llu experts\n", "pager", (unsigned long long) n);
    }

    void prepare(ggml_cgraph * cgraph, cudaStream_t stream) {
        if (!scanned && cgraph) {
            scan(cgraph, stream);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // promotions
    // -----------------------------------------------------------------------------------------------------------------

    copy_job make_job(uint32_t key, uint32_t slot, uint64_t id) const {
        const pager_group & g = groups[key / epl];
        const int e = (int) (key % epl);
        copy_job j;
        j.id = id;
        j.n = 3;
        j.pinned = g.pinned;
        for (int k = 0; k < 3; k++) {
            j.src[k]  = (const char *) g.w[k]->data + (size_t) e * g.w[k]->nb[2];
            j.dst[k]  = (char *) pool[k] + (size_t) (slot + 1) * stride[k];
            j.sz[k]   = g.w[k]->nb[2];
            j.span[k] = stride[k];
        }
        return j;
    }

    void run_job(const copy_job & j, cudaStream_t compute_stream) {
        if (copier) {
            copier->submit(j);
#ifndef NDEBUG
            if (j.op == 0 && test_early_commit) {
                // negative control for the poison check: commit before the copy completed, which breaks rule (a)
                sync_done.push_back(j.id);
                early_ids.insert(j.id);
            }
#endif
        } else {
            pager_do_copy(j, compute_stream, sync_bounce);
            if (j.op == 0) {
                sync_done.push_back(j.id);
            }
        }
    }

    // static fill: copy on the compute stream, behind the kernels already queued; ready at once (stream order)
    void install_copy(uint32_t key, cudaStream_t stream) {
        const int slot = cache->lookup(key);
        GGML_ASSERT(slot >= 0);
        const copy_job j = make_job(key, (uint32_t) slot, 0);
        pager_bounce & b = sync_bounce;
        pager_do_copy(j, stream, b);
#ifndef NDEBUG
        GGML_ASSERT(slot_st[slot] == SLOT_FREE);
        slot_st[slot] = SLOT_READY;
        slot_key[slot] = key;
#endif
        size_t total = 0;
        for (int k = 0; k < 3; k++) {
            total += j.sz[k];
        }
        g_stats.bytes_copied.fetch_add(total, std::memory_order_relaxed);
    }

    // [B] commit the promotions whose copies completed
    void collect_and_commit() {
        std::vector<uint64_t> ids;
        if (copier) {
            copier->drain_done(ids);
#ifndef NDEBUG
            if (test_early_commit) {
                std::vector<uint64_t> keep;
                for (uint64_t id : ids) {
                    if (!early_ids.erase(id)) {
                        keep.push_back(id);
                    }
                }
                ids.swap(keep);
                ids.insert(ids.end(), sync_done.begin(), sync_done.end());
                sync_done.clear();
            }
#endif
        } else {
            ids.swap(sync_done);
        }
        for (uint64_t id : ids) {
            auto it = inflight.find(id);
            GGML_ASSERT(it != inflight.end());
            const inflight_promo p = it->second;
            inflight.erase(it);
            g_stats.promo_lat_us.fetch_add(pager_now_us() - p.issue_us, std::memory_order_relaxed);
            g_stats.promo_lat_tok.fetch_add(tok - p.issue_tok, std::memory_order_relaxed);
            inflight_bytes -= slot_bytes;
            uint32_t slot = 0;
            const ggml_cuda_pager_cache::commit_status st = cache->commit(id, slot);
            GGML_ASSERT(st != ggml_cuda_pager_cache::stale);
#ifndef NDEBUG
            GGML_ASSERT(slot_st[p.dest] == SLOT_COPYING);
            if (st == ggml_cuda_pager_cache::committed) {
                GGML_ASSERT(slot == p.dest);
                slot_st[p.dest] = SLOT_READY;
                slot_key[p.dest] = p.key;
            } else {
                slot_st[p.dest] = SLOT_FREE;
                slot_key[p.dest] = -1;
            }
            for (uint32_t i = 0; i < p.n_victims; i++) {
                GGML_ASSERT(slot_claims[p.victim_slots[i]] > 0);
                slot_claims[p.victim_slots[i]]--;
            }
#endif
        }
#ifndef NDEBUG
        if (!ids.empty()) {
            sweep_slots();
        }
#endif
    }

#ifndef NDEBUG
    // Reconcile the debug state machine with the cache after commits: a slot whose key is no longer resident in it was
    // evicted (the cache does not report replacement victims chosen after a reclaim, so they are found here).
    void sweep_slots() {
        for (uint32_t s = 0; s < n_slots; s++) {
            if (slot_st[s] != SLOT_READY && slot_st[s] != SLOT_VICTIM) {
                continue;
            }
            if (cache->lookup((uint32_t) slot_key[s]) != (int) s) {
                // freed: no kernel queued before this block's synchronisation may still read it, none queued after can
                GGML_ASSERT(slot_last_read[s] < epoch);
                if (test_stale_read) {
                    stale_slot[(uint32_t) slot_key[s]] = s;
                }
                slot_st[s] = SLOT_FREE;
                slot_key[s] = -1;
                copy_job pj;
                pj.op = 1;
                pj.n = 3;
                for (int k = 0; k < 3; k++) {
                    pj.dst[k]  = (char *) pool[k] + (size_t) (s + 1) * stride[k];
                    pj.span[k] = stride[k];
                }
                run_job(pj, cur_stream);
            } else {
                slot_st[s] = slot_claims[s] > 0 ? SLOT_VICTIM : SLOT_READY;
            }
        }
    }
#endif

    // [E] hand the copier the promotions the cache wants next, within the bandwidth budget
    void issue_promotions(cudaStream_t stream) {
        const uint64_t now = pager_now_us();
        const double horizon_us = std::min(100000.0, std::max(5000.0, tok_ema_us));
        const double cap = bw_bytes_per_us * horizon_us;
        credit = std::min(cap, credit + bw_bytes_per_us * (double) (now - last_issue_us));
        last_issue_us = now;
        const double avail = cap - (double) inflight_bytes;
        const double allowed = std::min(credit, avail);
        if (allowed < (double) slot_bytes) {
            return;
        }
        ggml_cuda_pager_promotion out[64];
        const size_t n = cache->next_promotions((uint64_t) allowed, out, 64);
        for (size_t i = 0; i < n; i++) {
            const ggml_cuda_pager_promotion & p = out[i];
            inflight_promo ip;
            ip.issue_us = now;
            ip.issue_tok = tok;
            ip.key = p.key;
            ip.dest = p.dest_slot;
            ip.n_victims = p.n_victims;
            for (uint32_t v = 0; v < p.n_victims; v++) {
                ip.victim_slots[v] = p.victim_slots[v];
            }
            inflight[p.id] = ip;
            inflight_bytes += slot_bytes;
            credit -= (double) slot_bytes;
#ifndef NDEBUG
            GGML_ASSERT(slot_st[p.dest_slot] == SLOT_FREE); // a destination is always a free slot (freed at [B] at the latest)
            slot_st[p.dest_slot] = SLOT_COPYING;
            slot_key[p.dest_slot] = p.key;
            for (uint32_t v = 0; v < p.n_victims; v++) {
                GGML_ASSERT(slot_st[p.victim_slots[v]] == SLOT_READY || slot_st[p.victim_slots[v]] == SLOT_VICTIM);
                slot_st[p.victim_slots[v]] = SLOT_VICTIM;
                slot_claims[p.victim_slots[v]]++;
            }
#endif
            copy_job j = make_job(p.key, p.dest_slot, p.id);
#ifndef NDEBUG
            j.poison_first = true;
#endif
            run_job(j, stream);
        }
    }

    void publish_stats() {
        if (!cache) {
            return;
        }
        const ggml_cuda_pager_cache_counters c = cache->counters();
        g_stats.promo_issued.fetch_add(c.issued - pub.issued, std::memory_order_relaxed);
        g_stats.promo_committed.fetch_add(c.promotions - pub.promotions, std::memory_order_relaxed);
        g_stats.promo_aborted.fetch_add(c.aborted - pub.aborted, std::memory_order_relaxed);
        g_stats.evictions.fetch_add(c.evictions - pub.evictions, std::memory_order_relaxed);
        g_stats.admission_rejected.fetch_add(c.admission_rejected - pub.admission_rejected, std::memory_order_relaxed);
        g_stats.resident.fetch_add((uint64_t) c.resident - reported_resident);
        reported_resident = c.resident;
        pub = c;
        g_stats.queue_depth.store(inflight.size(), std::memory_order_relaxed);
        if (copier) {
            const uint64_t b = copier->bytes.load(std::memory_order_relaxed), u = copier->busy_us.load(std::memory_order_relaxed);
            g_stats.bytes_copied.fetch_add(b - pub_copied, std::memory_order_relaxed);
            g_stats.copy_busy_us.fetch_add(u - pub_busy, std::memory_order_relaxed);
            g_stats.copy_bytes.fetch_add(b - pub_copied, std::memory_order_relaxed);
            pub_copied = b;
            pub_busy = u;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // execution
    // -----------------------------------------------------------------------------------------------------------------

    struct run_info {
        int                 gid;
        bool                fused;
        const ggml_tensor * x;        // src[1] of the first stage
        const ggml_tensor * ids;
        ggml_tensor *       out;      // receives the CPU rows (fused: down, single: the node)
        ggml_tensor *       glu;      // fused only
        const ggml_tensor * up_w;     // fused: up weights (src0 of the kernel launch), single: weights
        const ggml_tensor * gate_w;   // fused only
        const ggml_tensor * down_w;   // fused only
    };

    // fake src0 over a role pool: the real tensor's type and shape, the pool as data, the slot stride as the expert stride
    static void make_fake(ggml_tensor & f, const ggml_tensor * real, void * base, size_t stride, int n_rows) {
        f = *real;
        f.data     = base;
        f.ne[2]    = n_rows;
        f.nb[2]    = stride;
        f.nb[3]    = stride * n_rows;
        f.view_src = nullptr;
        f.view_offs = 0;
        f.op       = GGML_OP_NONE;
        memset(f.src, 0, sizeof(f.src));
    }

    void exec(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const run_info & r);
};

// ---------------------------------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------------------------------

// rows: [n_miss][ne0] floats; pos: [n_miss][2] int32 (used-expert channel, token); dst strides in floats
static __global__ void pager_scatter_rows(const float * rows, const int32_t * pos, float * dst, const int64_t ne0, const int64_t s1, const int64_t s2) {
    const int m = blockIdx.x;
    float * d = dst + (int64_t) pos[2*m] * s1 + (int64_t) pos[2*m + 1] * s2;
    for (int64_t j = threadIdx.x; j < ne0; j += blockDim.x) {
        d[j] = rows[(int64_t) m * ne0 + j];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// one pager block
// ---------------------------------------------------------------------------------------------------------------------

void ggml_cuda_pager::exec(ggml_backend_cuda_context & ctx, ggml_cgraph *, const run_info & r) {
    cudaStream_t stream = ctx.stream();
    ggml_cuda_set_device(device);

    pager_group & grp = groups[r.gid];
    cur_stream = stream;

    const ggml_tensor * ids = r.ids;
    const ggml_tensor * x   = r.x;
    const int     n_used = (int) ids->ne[0];
    const int     T      = (int) ids->ne[1];
    const int64_t ne10   = x->ne[0];
    const int64_t ne11   = x->ne[1];
    const int64_t ne0    = r.out->ne[0];
    GGML_ASSERT(ids->ne[2] == 1 && ids->nb[0] == sizeof(int32_t) && x->ne[2] == T && x->nb[0] == sizeof(float));
    GGML_ASSERT(r.out->nb[0] == sizeof(float) && r.out->nb[1] % sizeof(float) == 0 && r.out->nb[2] % sizeof(float) == 0);
    GGML_ASSERT(T <= MMVQ_MAX_BATCH_SIZE);

    // (a) ids and input to the host. Every block starts with this synchronisation, which also guarantees that the previous
    // block's queued copies from the pinned staging buffers below have completed before they are written again.
    int32_t * ids_h = (int32_t *) h_ids.ensure((size_t) n_used * T * sizeof(int32_t));
    float *   x_h   = (float *)   h_x.ensure((size_t) ne10 * ne11 * T * sizeof(float));
    CUDA_CHECK(cudaMemcpy2DAsync(ids_h, n_used * sizeof(int32_t), ids->data, ids->nb[1], n_used * sizeof(int32_t), T, cudaMemcpyDeviceToHost, stream));
    for (int t = 0; t < T; t++) {
        CUDA_CHECK(cudaMemcpy2DAsync(x_h + (size_t) t * ne10 * ne11, ne10 * sizeof(float), (const char *) x->data + t * x->nb[2], x->nb[1],
                                     ne10 * sizeof(float), ne11, cudaMemcpyDeviceToHost, stream));
    }
    const uint64_t t_sync0 = pager_now_us();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const uint64_t t_host0 = pager_now_us();
    g_stats.sync_us.fetch_add(t_host0 - t_sync0, std::memory_order_relaxed);
    g_stats.syncs.fetch_add(1, std::memory_order_relaxed);
    g_stats.blocks.fetch_add(1, std::memory_order_relaxed);

    const bool has_pool = cache && grp.cached;
    // [A] is the synchronisation above. [B]: commit the promotions whose copies completed (the only place slots are freed)
    epoch++;
    if (cache) {
        collect_and_commit();
    }
    // Token index for the cache (touch/on_miss/admission window/LFU decay): the count of tokens evaluated through the pager.
    // A new decode call starts when the block's group index does not exceed the previous block's (layers run in increasing
    // order within a call); the counter then advances by that call's T and every touch of the call uses the new value, so
    // a T = 1 decode is the per-sequence step counter, a batch of T tokens (several sequences, or a short prompt) is one
    // step that counts T tokens, and the index never decreases within a call.
    if (cache && grp.cached) {
        if (r.gid <= last_gid || tok == 0) {
            tok += T;
            const uint64_t now = pager_now_us();
            if (last_tok_us != 0 && now - last_tok_us < 1000000) {
                tok_ema_us = 0.9 * tok_ema_us + 0.1 * (double) (now - last_tok_us);
            }
            last_tok_us = now;
        }
        last_gid = r.gid;
    }

    // (b) hit or miss per (token, used expert); hits are remapped to their slot, misses to the null slot 0
    struct miss_t { int c, t, e; };
    std::vector<miss_t> miss;
    int32_t * remap_h = (int32_t *) h_remap.ensure((size_t) n_used * T * sizeof(int32_t));
    std::vector<uint32_t> static_keys;
    int n_hit = 0;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < n_used; c++) {
            const int e = ids_h[t * n_used + c];
            GGML_ASSERT(e >= 0 && e < grp.n_expert);
            const uint32_t key = (uint32_t) (r.gid * epl + e);
            int slot = has_pool ? cache->lookup(key) : -1;
            if (has_pool) {
                cache->touch(key, tok);
            }
#ifndef NDEBUG
            if (slot < 0 && test_stale_read && has_pool) {
                auto st = stale_slot.find(key);
                if (st != stale_slot.end()) {
                    slot = (int) st->second;
                }
            }
#endif
            if (slot >= 0) {
                remap_h[t * n_used + c] = slot + 1; // pool row 0 is the null slot
                n_hit++;
#ifndef NDEBUG
                if (test_stale_read != 2) {
                    GGML_ASSERT(slot_st[slot] == SLOT_READY || slot_st[slot] == SLOT_VICTIM);
                }
                slot_last_read[slot] = epoch;
#endif
            } else {
                remap_h[t * n_used + c] = 0;
                miss.push_back({ c, t, e });
                if (has_pool) {
                    if (policy == POLICY_STATIC) {
                        static_keys.push_back(key);
                    } else {
                        cache->on_miss(key, tok);
                    }
                }
            }
        }
    }
    const int n_miss = (int) miss.size();
    GGML_ASSERT(n_hit + n_miss == n_used * T);
    n_blocks++;
    n_hits += n_hit;
    n_misses += n_miss;
    n_expected += (uint64_t) n_used * T;
    g_stats.hits.fetch_add(n_hit, std::memory_order_relaxed);
    g_stats.misses.fetch_add(n_miss, std::memory_order_relaxed);

    // (d, first half) CPU chain for the misses, started before the GPU work is enqueued
    const size_t rows_bytes = (size_t) n_miss * ne0 * sizeof(float);
    const size_t blob_bytes = rows_bytes + (size_t) n_miss * 2 * sizeof(int32_t);
    char *    blob_h = n_miss > 0 ? (char *) h_blob.ensure(blob_bytes) : nullptr;
    float *   rows_h = (float *) blob_h;
    int32_t * pos_h  = (int32_t *) (blob_h + rows_bytes);

    constexpr int CHUNK = 32;
    std::vector<mv_task>                 tasks;
    std::vector<std::pair<int, int>>     items;      // (task, first row)
    std::vector<std::vector<uint8_t>>    xq_a, xq_b; // quantised inputs: gate, up (fused) or one set (single)
    std::vector<float>                   gbuf, ubuf;
    std::vector<std::vector<uint8_t>>    hq;
    std::vector<int>                     tok_idx(T * ne11, -1);

    const ggml_type_traits_cpu * tr_a = nullptr, * tr_b = nullptr, * tr_d = nullptr;
    const ggml_tensor * wa = r.fused ? r.gate_w : r.up_w; // first-stage weights (gate or the single weights)
    const ggml_tensor * wb = r.fused ? r.up_w   : nullptr;
    const ggml_tensor * wd = r.fused ? r.down_w : nullptr;
    int64_t ff = 0;
    uint64_t t_cpu0 = 0;

    const auto build_items = [&]() {
        items.clear();
        for (size_t k = 0; k < tasks.size(); k++) {
            for (int r0 = 0; r0 < tasks[k].nrows; r0 += CHUNK) {
                items.emplace_back((int) k, r0);
            }
        }
    };
    const auto run_item = [&](int i) {
        const mv_task & tk = tasks[items[i].first];
        const int r0 = items[i].second;
        const int r1 = std::min(r0 + CHUNK, tk.nrows);
        for (int rr = r0; rr < r1; rr++) {
            tk.dot(tk.n, tk.out + rr, 0, tk.w + (size_t) rr * tk.row_bytes, 0, tk.xq, 0, 1);
        }
    };

    if (n_miss > 0) {
        t_cpu0 = pager_now_us();
        tr_a = pager_cpu_traits(wa->type);
        GGML_ASSERT(tr_a && tr_a->vec_dot);
        const ggml_type_traits_cpu * tq_a = pager_cpu_traits(tr_a->vec_dot_type);
        GGML_ASSERT(tq_a && tq_a->from_float);
        const int64_t K = wa->ne[0];
        for (int m = 0; m < n_miss; m++) {
            pos_h[2 * m]     = miss[m].c;
            pos_h[2 * m + 1] = miss[m].t;
        }
        if (r.fused) {
            ff = wa->ne[1];
            tr_b = pager_cpu_traits(wb->type);
            tr_d = pager_cpu_traits(wd->type);
            GGML_ASSERT(tr_b && tr_b->vec_dot && tr_d && tr_d->vec_dot);
            const ggml_type_traits_cpu * tq_b = pager_cpu_traits(tr_b->vec_dot_type);
            GGML_ASSERT(tq_b && tq_b->from_float);
            // quantise the input of every token that has a miss once per distinct vec_dot type
            for (const miss_t & ms : miss) {
                if (tok_idx[ms.t] < 0) {
                    tok_idx[ms.t] = (int) xq_a.size();
                    xq_a.emplace_back(ggml_row_size(tr_a->vec_dot_type, K));
                    tq_a->from_float(x_h + (size_t) ms.t * K, xq_a.back().data(), K);
                    if (tr_b->vec_dot_type != tr_a->vec_dot_type) {
                        xq_b.emplace_back(ggml_row_size(tr_b->vec_dot_type, K));
                        tq_b->from_float(x_h + (size_t) ms.t * K, xq_b.back().data(), K);
                    }
                }
            }
            gbuf.resize((size_t) n_miss * ff);
            ubuf.resize((size_t) n_miss * ff);
            for (int m = 0; m < n_miss; m++) {
                const int ti = tok_idx[miss[m].t];
                const void * xa = xq_a[ti].data();
                const void * xb = tr_b->vec_dot_type != tr_a->vec_dot_type ? xq_b[ti].data() : xa;
                tasks.push_back({ (const char *) wa->data + (size_t) miss[m].e * wa->nb[2], wa->nb[1], xa, gbuf.data() + (size_t) m * ff, (int) ff, (int) K, tr_a->vec_dot });
                tasks.push_back({ (const char *) wb->data + (size_t) miss[m].e * wb->nb[2], wb->nb[1], xb, ubuf.data() + (size_t) m * ff, (int) ff, (int) K, tr_b->vec_dot });
            }
        } else {
            // one quantised row per distinct (token, input row)
            std::vector<int> xi(n_miss);
            for (int m = 0; m < n_miss; m++) {
                const int64_t i11 = miss[m].c % ne11;
                int & idx = tok_idx[miss[m].t * ne11 + i11];
                if (idx < 0) {
                    idx = (int) xq_a.size();
                    xq_a.emplace_back(ggml_row_size(tr_a->vec_dot_type, K));
                    tq_a->from_float(x_h + ((size_t) miss[m].t * ne11 + i11) * K, xq_a.back().data(), K);
                }
                xi[m] = idx;
            }
            for (int m = 0; m < n_miss; m++) {
                tasks.push_back({ (const char *) wa->data + (size_t) miss[m].e * wa->nb[2], wa->nb[1], xq_a[xi[m]].data(), rows_h + (size_t) m * ne0, (int) ne0, (int) K, tr_a->vec_dot });
            }
        }
        build_items();
        workers.submit((int) items.size(), run_item);
    }

    // (c) the unchanged kernels over the pools, for the experts that hit
    if (n_hit > 0) {
        ggml_cuda_pool_alloc<int32_t> remap_d(ctx.pool(), (size_t) n_used * T);
        CUDA_CHECK(cudaMemcpyAsync(remap_d.get(), remap_h, (size_t) n_used * T * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        ggml_tensor ids_f = {};
        ids_f.type  = GGML_TYPE_I32;
        ids_f.ne[0] = n_used; ids_f.ne[1] = T; ids_f.ne[2] = 1; ids_f.ne[3] = 1;
        ids_f.nb[0] = sizeof(int32_t);
        ids_f.nb[1] = ids_f.nb[0] * n_used;
        ids_f.nb[2] = ids_f.nb[1] * T;
        ids_f.nb[3] = ids_f.nb[2];
        ids_f.data  = remap_d.get();

        ggml_tensor fake_a, fake_b, fake_d;
        GGML_ASSERT(r.fused && has_pool);
        make_fake(fake_a, r.gate_w, pool[0], stride[0], n_slots + 1);
        if (r.fused) {
            make_fake(fake_b, r.up_w, pool[1], stride[1], n_slots + 1);
            make_fake(fake_d, r.down_w, pool[2], stride[2], n_slots + 1);
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate      = &fake_a;
            fusion.glu_op    = ggml_get_glu_op(r.glu);
            fusion.glu_limit = ggml_get_op_params_f32(r.glu, 3);
            ggml_cuda_mul_mat_vec_q(ctx, &fake_b, x, &ids_f, r.glu, &fusion);
            ggml_cuda_mul_mat_vec_q(ctx, &fake_d, r.glu, &ids_f, r.out);
        } else {
            ggml_cuda_mul_mat_vec_q(ctx, &fake_a, x, &ids_f, r.out);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // (d, second half) join the CPU chain
    if (n_miss > 0) {
        workers.join();
        if (r.fused) {
            // silu(gate) * up, exactly as ggml_vec_swiglu_f32 (src0 = gate), then quantise for the down projection
            const ggml_type_traits_cpu * tq_d = pager_cpu_traits(tr_d->vec_dot_type);
            GGML_ASSERT(tq_d && tq_d->from_float);
            std::vector<float> h(ff);
            const ggml_vec_swiglu_f32_t swiglu = g_cpu_swiglu.load(std::memory_order_acquire);
            hq.resize(n_miss);
            tasks.clear();
            for (int m = 0; m < n_miss; m++) {
                const float * g = gbuf.data() + (size_t) m * ff;
                const float * u = ubuf.data() + (size_t) m * ff;
                if (swiglu) {
                    swiglu((int) ff, h.data(), g, u); // the CPU backend's own routine, bit-identical to its GLU op
                } else {
                    for (int64_t j = 0; j < ff; j++) {
                        h[j] = (g[j] / (1.0f + expf(-g[j]))) * u[j];
                    }
                }
                hq[m].resize(ggml_row_size(tr_d->vec_dot_type, ff));
                tq_d->from_float(h.data(), hq[m].data(), ff);
                tasks.push_back({ (const char *) wd->data + (size_t) miss[m].e * wd->nb[2], wd->nb[1], hq[m].data(), rows_h + (size_t) m * ne0, (int) ne0, (int) ff, tr_d->vec_dot });
            }
            build_items();
            workers.submit((int) items.size(), run_item);
            workers.join();
        }
        g_stats.cpu_us.fetch_add(pager_now_us() - t_cpu0, std::memory_order_relaxed);

        // (e) upload the missed rows and scatter them into the output at their (channel, token) positions
        ggml_cuda_pool_alloc<char> blob_d(ctx.pool(), blob_bytes);
        CUDA_CHECK(cudaMemcpyAsync(blob_d.get(), blob_h, blob_bytes, cudaMemcpyHostToDevice, stream));
        pager_scatter_rows<<<n_miss, 256, 0, stream>>>((const float *) blob_d.get(), (const int32_t *) (blob_d.get() + rows_bytes),
            (float *) r.out->data, ne0, r.out->nb[1] / sizeof(float), r.out->nb[2] / sizeof(float));
        CUDA_CHECK(cudaGetLastError());
    }

    // [E] promotions. Static policy: missed experts take a free slot if there is one; the copy is queued behind this block's
    // kernels on the compute stream, so it is complete before any later kernel can look the slot up. Otherwise the cache
    // decides what to promote and the copier does it in the background.
    if (has_pool) {
        if (policy == POLICY_STATIC) {
            std::sort(static_keys.begin(), static_keys.end());
            static_keys.erase(std::unique(static_keys.begin(), static_keys.end()), static_keys.end());
            for (uint32_t key : static_keys) {
                if (cache->install(key)) {
                    install_copy(key, stream);
                }
            }
        } else {
            issue_promotions(stream);
        }
        publish_stats();
    }
    g_stats.host_us.fetch_add(pager_now_us() - t_host0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------------------------------------------------

static ggml_cuda_pager & pager_of(ggml_backend_cuda_context & ctx) {
    if (!ctx.pager) {
        ctx.pager = new ggml_cuda_pager(ctx.device);
    }
    return *ctx.pager;
}

void ggml_cuda_pager_destroy(ggml_cuda_pager * pager) {
    delete pager;
}

int ggml_cuda_pager_try_block(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL_MAT_ID || ggml_cuda_tensor_paged_device(node->src[0]) != ctx.device) {
        return 0;
    }
    ggml_cuda_pager & p = pager_of(ctx);
    p.prepare(cgraph, ctx.stream());

    block_match m;
    if (!p.match_block(cgraph, i, m)) {
        GGML_LOG_DEBUG("%s: no fused block at node %d (%s), running the MUL_MAT_ID nodes one by one\n", "pager", i, node->name);
        return 0;
    }
    const ggml_tensor * w[3] = { m.gate->src[0], m.up->src[0], m.down->src[0] };
    const int gid = p.register_group(w, 3);

    ggml_cuda_pager::run_info r;
    r.gid    = gid;
    r.fused  = true;
    r.x      = m.up->src[1];
    r.ids    = m.up->src[2];
    r.out    = m.down;
    r.glu    = m.glu;
    r.up_w   = m.up->src[0];
    r.gate_w = m.gate->src[0];
    r.down_w = m.down->src[0];
    p.exec(ctx, cgraph, r);
    return 3;
}

void ggml_cuda_pager_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_pager & p = pager_of(ctx);
    if (!p.pools_built) {
        p.scanned = true;
        p.build_pools(ctx.stream());
    }
    const ggml_tensor * w[1] = { dst->src[0] };
    const int gid = p.register_group(w, 1);

    ggml_cuda_pager::run_info r;
    r.gid   = gid;
    r.fused = false;
    r.x     = dst->src[1];
    r.ids   = dst->src[2];
    r.out   = dst;
    r.glu   = nullptr;
    r.up_w  = dst->src[0];
    r.gate_w = r.down_w = nullptr;
    p.exec(ctx, nullptr, r);
}

// ---------------------------------------------------------------------------------------------------------------------
// supports_op
// ---------------------------------------------------------------------------------------------------------------------

static bool pager_mmvq_type_ok(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_pager_supports_op(int device, const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * w   = op->src[0];
    const ggml_tensor * x   = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (!w || !x || !ids || ggml_cuda_tensor_paged_device(w) != device) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && ggml_cuda_tensor_is_paged(op->src[i])) {
            return false;
        }
    }
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (!pager_mmvq_type_ok(w->type) || w->ne[3] != 1 || w->nb[0] != ggml_type_size(w->type) ||
        x->nb[0] != sizeof(float) || op->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1) {
        return false;
    }
    // CPU kernels must have been injected, and exist for this type
    const ggml_type_traits_cpu * tr = pager_cpu_traits(w->type);
    if (!tr || !tr->vec_dot) {
        return false;
    }
    const ggml_type_traits_cpu * tq = pager_cpu_traits(tr->vec_dot_type);
    if (!tq || !tq->from_float || !ggml_is_quantized(w->type)) {
        return false;
    }
    // batches the MMVQ kernels serve; larger batches run on the CPU backend
    const int cc = ggml_cuda_info().devices[device].cc;
    return op->ne[2] <= std::min(MMVQ_MAX_BATCH_SIZE, get_mmvq_mmid_max_batch(w->type, cc)) && ids->ne[1] == op->ne[2] && x->ne[2] == op->ne[2];
}
