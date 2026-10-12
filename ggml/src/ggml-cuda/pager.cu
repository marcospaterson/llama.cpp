#include "pager.cuh"
#include "mmvq.cuh"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#if defined(__x86_64__)
#define PAGER_CPU_RELAX() __asm__ __volatile__("pause" ::: "memory")
#else
#define PAGER_CPU_RELAX() do {} while (0)
#endif

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
// policy cache (static fill)
// ---------------------------------------------------------------------------------------------------------------------

int ggml_cuda_pager_cache::add_class(int n_slots) {
    n_slots_.push_back(n_slots);
    next_free_.push_back(1);
    ready_.emplace_back(n_slots + 1, 0);
    return (int) n_slots_.size() - 1;
}

int ggml_cuda_pager_cache::add_layer(int cls, int n_expert) {
    GGML_ASSERT(cls >= 0 && cls < (int) n_slots_.size());
    layer_cls_.push_back(cls);
    slot_of_.emplace_back(n_expert, 0);
    return (int) layer_cls_.size() - 1;
}

int ggml_cuda_pager_cache::lookup(int layer, int expert) const {
    const int slot = slot_of_[layer][expert];
    return slot > 0 && ready_[layer_cls_[layer]][slot] ? slot : -1;
}

void ggml_cuda_pager_cache::touch(int layer, int expert) {
    GGML_UNUSED(layer);
    GGML_UNUSED(expert);
}

void ggml_cuda_pager_cache::request_promotion(int layer, int expert) {
    if (slot_of_[layer][expert] == 0) {
        pending_.emplace_back(layer, expert);
    }
}

std::vector<ggml_cuda_pager_promotion> ggml_cuda_pager_cache::commit() {
    std::vector<ggml_cuda_pager_promotion> out;
    for (const auto & [layer, expert] : pending_) {
        const int cls = layer_cls_[layer];
        if (slot_of_[layer][expert] != 0 || next_free_[cls] > n_slots_[cls]) {
            continue; // already promoted by an earlier request, or the pool is full: nothing changes
        }
        const int slot = next_free_[cls]++;
        slot_of_[layer][expert] = slot;
        out.push_back({layer, expert, cls, slot});
        n_resident_++;
    }
    pending_.clear();
    return out;
}

void ggml_cuda_pager_cache::mark_ready(const ggml_cuda_pager_promotion & p) {
    ready_[p.cls][p.slot] = 1;
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

struct member_sig {
    ggml_type type;
    int64_t   ne0, ne1;
    size_t    nb1, nb2;
    bool operator==(const member_sig & o) const {
        return type == o.type && ne0 == o.ne0 && ne1 == o.ne1 && nb1 == o.nb1 && nb2 == o.nb2;
    }
};

struct class_sig {
    int        n = 0;
    member_sig m[3] = {};
    bool operator==(const class_sig & o) const {
        if (n != o.n) {
            return false;
        }
        for (int k = 0; k < n; k++) {
            if (!(m[k] == o.m[k])) {
                return false;
            }
        }
        return true;
    }
};

struct pager_class {
    class_sig sig;
    int       n_slots = 0;           // usable slots (excluding the null slot); 0 = no pool, every lookup misses
    void *    pool[3] = {};          // device pools, one per member; slot s of member k at pool[k] + s * sig.m[k].nb2
    size_t    pool_bytes[3] = {};
    int       n_layers = 0;
    int64_t   n_experts = 0;         // sum over layers
};

struct pager_group {
    int                 cls = -1;
    int                 n = 0;       // 1 (single MUL_MAT_ID) or 3 (gate, up, down)
    const ggml_tensor * w[3] = {};
    int                 n_expert = 0;
};

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

struct ggml_cuda_pager {
    int    device;
    size_t budget;

    pager_workers          workers;
    ggml_cuda_pager_cache  cache;
    std::vector<pager_class> classes;
    std::vector<pager_group> groups;       // index == cache layer
    std::unordered_map<const ggml_tensor *, int> group_of;
    bool   pools_built = false;
    bool   scanned     = false;

    pager_host_buf h_ids, h_x, h_remap, h_blob;

    uint64_t reported_resident = 0;
    uint64_t reported_pool_bytes = 0;
    // this pager's own totals (logged when it is destroyed); expected = sum over blocks of n_used * T
    uint64_t n_hits = 0, n_misses = 0, n_blocks = 0, n_expected = 0, n_copied = 0;

    explicit ggml_cuda_pager(int device) :
        device(device), budget(ggml_cuda_pager_budget_bytes()), workers(pager_n_threads()) {
        GGML_LOG_INFO("%s: device %d, pool %.0f MiB, %d CPU threads\n", "pager", device, budget / 1048576.0, pager_n_threads());
    }

    ~ggml_cuda_pager() {
        // the worker threads are joined by ~pager_workers after this body; they only run inside exec(), which has returned
        ggml_cuda_set_device(device);
        for (auto & c : classes) {
            for (int k = 0; k < 3; k++) {
                if (c.pool[k]) {
                    CUDA_CHECK(cudaFree(c.pool[k]));
                }
            }
        }
        if (n_blocks > 0) {
            GGML_LOG_INFO("%s: device %d totals: %llu blocks, hits %llu + misses %llu = %llu (expert evaluations expected: %llu), resident %llu, copied %.1f MiB\n",
                          "pager", device, (unsigned long long) n_blocks, (unsigned long long) n_hits, (unsigned long long) n_misses,
                          (unsigned long long) (n_hits + n_misses), (unsigned long long) n_expected, (unsigned long long) cache.n_resident(), n_copied / 1048576.0);
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
    // groups, classes, pools
    // -----------------------------------------------------------------------------------------------------------------

    static member_sig sig_of(const ggml_tensor * w) {
        return { w->type, w->ne[0], w->ne[1], w->nb[1], w->nb[2] };
    }

    // register the weights of a block (idempotent); the cache layer index equals the group index
    int register_group(const ggml_tensor * const * w, int n) {
        auto it = group_of.find(w[0]);
        if (it != group_of.end()) {
            return it->second;
        }
        pager_group g;
        g.n = n;
        class_sig sig;
        sig.n = n;
        for (int k = 0; k < n; k++) {
            g.w[k]    = w[k];
            sig.m[k]  = sig_of(w[k]);
        }
        g.n_expert = (int) w[0]->ne[2];
        int cls = -1;
        for (size_t c = 0; c < classes.size(); c++) {
            if (classes[c].sig == sig) {
                cls = (int) c;
                break;
            }
        }
        if (cls < 0) {
            pager_class c;
            c.sig = sig;
            classes.push_back(c);
            cls = (int) classes.size() - 1;
        }
        g.cls = cls;
        classes[cls].n_layers++;
        classes[cls].n_experts += g.n_expert;
        groups.push_back(g);
        group_of[w[0]] = (int) groups.size() - 1;
        return (int) groups.size() - 1;
    }

    static size_t expert_bytes_padded_tail(const member_sig & s) {
        return s.ne0 % MATRIX_ROW_PADDING != 0 ? ggml_row_size(s.type, MATRIX_ROW_PADDING - s.ne0 % MATRIX_ROW_PADDING) : 0;
    }

    // Split the budget over the classes in proportion to their total expert bytes (the same resident fraction in every
    // class), allocate the pools (slot 0 and the row padding zeroed) and create the cache entries. Called once, after the
    // first graph has been scanned; groups seen later join an existing class or get a class without pool.
    void build_pools(cudaStream_t stream) {
        pools_built = true;
        const int first_group_of_cache = cache.n_layers();
        GGML_ASSERT(first_group_of_cache == 0);

        // bytes of one slot over all members, and the null slots that are paid for up front
        std::vector<size_t> bps(classes.size(), 0);
        double total_bytes = 0.0;
        size_t null_bytes  = 0;
        for (size_t c = 0; c < classes.size(); c++) {
            for (int k = 0; k < classes[c].sig.n; k++) {
                bps[c] += classes[c].sig.m[k].nb2;
            }
            total_bytes += (double) classes[c].n_experts * bps[c];
            null_bytes  += bps[c];
        }
        double frac = 0.0;
        if (budget > null_bytes && total_bytes > 0.0) {
            frac = std::min(1.0, (double) (budget - null_bytes) / total_bytes);
        }

        ggml_cuda_set_device(device);
        uint64_t pool_total = 0;
        for (size_t c = 0; c < classes.size(); c++) {
            pager_class & pc = classes[c];
            int n_slots = (int) std::min<double>(frac * (double) pc.n_experts, (double) pc.n_experts);
            if (n_slots > 0) {
                bool ok = true;
                for (int k = 0; k < pc.sig.n && ok; k++) {
                    const member_sig & s = pc.sig.m[k];
                    pc.pool_bytes[k] = s.nb2 * (size_t) (n_slots + 1) + expert_bytes_padded_tail(s);
                    cudaError_t err = cudaMalloc(&pc.pool[k], pc.pool_bytes[k]);
                    if (err != cudaSuccess) {
                        (void) cudaGetLastError();
                        GGML_LOG_ERROR("%s: pool allocation of %.1f MiB failed: %s; class %zu runs without a pool\n", __func__,
                                       pc.pool_bytes[k] / 1048576.0, cudaGetErrorString(err), c);
                        ok = false;
                    } else {
                        CUDA_CHECK(cudaMemsetAsync(pc.pool[k], 0, pc.pool_bytes[k], stream));
                    }
                }
                if (!ok) {
                    for (int k = 0; k < 3; k++) {
                        if (pc.pool[k]) {
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                            CUDA_CHECK(cudaFree(pc.pool[k]));
                            pc.pool[k] = nullptr;
                        }
                        pc.pool_bytes[k] = 0;
                    }
                    n_slots = 0;
                }
            }
            pc.n_slots = n_slots;
            for (int k = 0; k < 3; k++) {
                pool_total += pc.pool_bytes[k];
            }
            const int cc = cache.add_class(n_slots);
            GGML_ASSERT(cc == (int) c);
            GGML_LOG_INFO("%s: class %zu: %d layers x %lld experts, %d members, %d slots (%.1f MiB)\n", "pager", c, pc.n_layers,
                          (long long) (pc.n_experts / std::max(1, pc.n_layers)), pc.sig.n, n_slots,
                          (pc.pool_bytes[0] + pc.pool_bytes[1] + pc.pool_bytes[2]) / 1048576.0);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (size_t g = 0; g < groups.size(); g++) {
            const int layer = cache.add_layer(groups[g].cls, groups[g].n_expert);
            GGML_ASSERT(layer == (int) g);
        }
        reported_pool_bytes = pool_total;
        g_stats.pool_bytes.fetch_add(pool_total);

        // test aid: GGML_CUDA_PAGER_PREFILL=1 fills the pools in graph order before the first block (no cold misses)
        if (getenv("GGML_CUDA_PAGER_PREFILL") && atoi(getenv("GGML_CUDA_PAGER_PREFILL")) != 0) {
            for (size_t g = 0; g < groups.size(); g++) {
                if (classes[groups[g].cls].n_slots > 0) {
                    for (int e = 0; e < groups[g].n_expert; e++) {
                        cache.request_promotion((int) g, e);
                    }
                }
            }
            fill(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            GGML_LOG_INFO("%s: prefilled %llu experts\n", "pager", (unsigned long long) cache.n_resident());
        }
    }

    // late groups (not in the graph seen first)
    void sync_cache_layers() {
        while (cache.n_layers() < (int) groups.size()) {
            pager_group & g = groups[cache.n_layers()];
            // classes created after build_pools have no cache entry yet and no pool
            while (cache.n_classes() <= g.cls) {
                cache.add_class(0);
            }
            cache.add_layer(g.cls, g.n_expert);
        }
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
        }
    }

    void prepare(ggml_cgraph * cgraph, cudaStream_t stream) {
        if (!scanned && cgraph) {
            scan(cgraph, stream);
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

    static void make_fake(ggml_tensor & f, const ggml_tensor * real, void * base, int n_slots_with_null) {
        f = *real;
        f.data     = base;
        f.ne[2]    = n_slots_with_null;
        f.nb[3]    = f.nb[2] * n_slots_with_null;
        f.view_src = nullptr;
        f.view_offs = 0;
        f.op       = GGML_OP_NONE;
        memset(f.src, 0, sizeof(f.src));
    }

    void exec(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const run_info & r);

    // let the policy decide the requested promotions and queue the host to VRAM copies on `stream`
    void fill(cudaStream_t stream) {
        for (const ggml_cuda_pager_promotion & p : cache.commit()) {
            const pager_group & pg = groups[p.layer];
            for (int k = 0; k < pg.n; k++) {
                const size_t sz = pg.w[k]->nb[2];
                CUDA_CHECK(cudaMemcpyAsync((char *) classes[p.cls].pool[k] + (size_t) p.slot * sz,
                    (const char *) pg.w[k]->data + (size_t) p.expert * sz, sz, cudaMemcpyHostToDevice, stream));
                g_stats.bytes_copied.fetch_add(sz, std::memory_order_relaxed);
                n_copied += sz;
            }
            cache.mark_ready(p);
        }
        const uint64_t resident = cache.n_resident();
        g_stats.resident.fetch_add(resident - reported_resident);
        reported_resident = resident;
    }
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
    pager_class & pc  = classes[grp.cls];
    const int layer   = r.gid;

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

    // (b) hit or miss per (token, used expert); hits are remapped to their slot, misses to the null slot 0
    struct miss_t { int c, t, e; };
    std::vector<miss_t> miss;
    int32_t * remap_h = (int32_t *) h_remap.ensure((size_t) n_used * T * sizeof(int32_t));
    const bool has_pool = pc.n_slots > 0 && pc.pool[0] != nullptr;
    int n_hit = 0;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < n_used; c++) {
            const int e = ids_h[t * n_used + c];
            GGML_ASSERT(e >= 0 && e < grp.n_expert);
            const int slot = has_pool ? cache.lookup(layer, e) : -1;
            if (slot > 0) {
                remap_h[t * n_used + c] = slot;
                cache.touch(layer, e);
                n_hit++;
            } else {
                remap_h[t * n_used + c] = 0;
                miss.push_back({ c, t, e });
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
#ifndef NDEBUG
        for (int i = 0; i < n_used * T; i++) {
            GGML_ASSERT(remap_h[i] == 0 || cache.slot_ready(grp.cls, remap_h[i]));
        }
#endif
        ggml_tensor ids_f = {};
        ids_f.type  = GGML_TYPE_I32;
        ids_f.ne[0] = n_used; ids_f.ne[1] = T; ids_f.ne[2] = 1; ids_f.ne[3] = 1;
        ids_f.nb[0] = sizeof(int32_t);
        ids_f.nb[1] = ids_f.nb[0] * n_used;
        ids_f.nb[2] = ids_f.nb[1] * T;
        ids_f.nb[3] = ids_f.nb[2];
        ids_f.data  = remap_d.get();

        ggml_tensor fake_a, fake_b, fake_d;
        make_fake(fake_a, r.fused ? r.gate_w : r.up_w, pc.pool[0], pc.n_slots + 1);
        if (r.fused) {
            make_fake(fake_b, r.up_w, pc.pool[1], pc.n_slots + 1);
            make_fake(fake_d, r.down_w, pc.pool[2], pc.n_slots + 1);
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

    // static fill: missed experts take the next free slot; the copy is queued behind this block's kernels on the compute
    // stream, so it is complete before any later kernel can look the slot up
    if (has_pool && n_miss > 0) {
        for (const miss_t & ms : miss) {
            cache.request_promotion(layer, ms.e);
        }
        fill(stream);
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
    p.sync_cache_layers();

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
    p.sync_cache_layers();

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
