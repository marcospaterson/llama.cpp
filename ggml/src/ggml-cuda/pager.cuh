#pragma once

// Expert pager, stage 1 (static fill).
//
// Expert weights live in host memory in the "CUDA<n>_PAGED" buffer type. A VRAM slot pool holds a subset of the experts; the
// CUDA backend executes MUL_MAT_ID nodes whose weights are paged as one unit per layer:
//   1. read the expert ids and the input to the host and synchronise the stream,
//   2. hit/miss lookup per (token, used expert); build remapped ids (hit -> slot, miss -> slot 0, an all-zero "null" slot),
//   3. launch the unchanged MMVQ kernels over the pools (served experts),
//   4. compute the missed (token, expert) pairs on the CPU with the CPU backend's kernels, on a worker pool, concurrently,
//   5. scatter the CPU rows into the kernel output.
// Threads: every call below is made by the thread that runs the backend graph (the "compute thread"); the worker pool only
// executes the matvec chunks it is handed (read-only paged weights and quantised inputs, disjoint output rows).

#include "common.cuh"
#include "ggml-pager.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------------
// Policy cache: (layer, expert) -> slot. Host only, no CUDA. The kernel-facing code in pager.cu uses only these calls, so a
// different policy (LRU, background promotion, eviction) can replace the static fill behind the same interface.
//
// A slot index is valid in all pools of the class of the layer (gate, up and down of an expert share the index). Slot 0 of
// every pool is the permanent null slot; usable slots are 1..n_slots. A slot is returned by lookup() only once ready.
// ---------------------------------------------------------------------------------------------------------------------

struct ggml_cuda_pager_promotion {
    int layer;
    int expert;
    int cls;
    int slot;
};

class ggml_cuda_pager_cache {
public:
    // Classes group layers whose experts share one slot geometry. n_slots excludes the null slot.
    int add_class(int n_slots);
    int add_layer(int cls, int n_expert);

    int n_classes() const { return (int) n_slots_.size(); }
    int n_layers() const { return (int) layer_cls_.size(); }
    int layer_class(int layer) const { return layer_cls_[layer]; }
    int class_slots(int cls) const { return n_slots_[cls]; }

    // slot (>= 1) when the expert is resident and fully written, -1 on a miss
    int lookup(int layer, int expert) const;
    // record a use of a resident expert (no-op for the static fill)
    void touch(int layer, int expert);
    // ask for the expert to become resident; the policy decides when and which slot (static fill: first free slot at commit)
    void request_promotion(int layer, int expert);
    // decide pending requests; the caller must copy each returned promotion into its slot and then call mark_ready()
    std::vector<ggml_cuda_pager_promotion> commit();
    void mark_ready(const ggml_cuda_pager_promotion & p);

    bool slot_ready(int cls, int slot) const { return slot >= 1 && slot <= n_slots_[cls] && ready_[cls][slot] != 0; }
    uint64_t n_resident() const { return n_resident_; }

private:
    std::vector<int>                  layer_cls_;
    std::vector<std::vector<int32_t>> slot_of_;   // [layer][expert] -> slot, 0 = not resident / not promoted
    std::vector<int>                  n_slots_;   // [cls]
    std::vector<int>                  next_free_; // [cls]
    std::vector<std::vector<uint8_t>> ready_;     // [cls][slot]
    std::vector<std::pair<int, int>>  pending_;
    uint64_t                          n_resident_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------------
// Backend hooks (implemented in pager.cu)
// ---------------------------------------------------------------------------------------------------------------------

struct ggml_cuda_pager;

// true for a buffer type / tensor held in a CUDA<n>_PAGED buffer (defined in ggml-cuda.cu)
bool ggml_cuda_buft_is_paged(ggml_backend_buffer_type_t buft);
// device index owning the paged buffer that holds t (or its view_src), -1 when t is not paged
int  ggml_cuda_tensor_paged_device(const ggml_tensor * t);
static inline bool ggml_cuda_tensor_is_paged(const ggml_tensor * t) { return ggml_cuda_tensor_paged_device(t) >= 0; }

// stock fusion checks from ggml-cuda.cu, reused to recognise the block (graph structure only, they never read weights)
bool ggml_cuda_should_fuse_mul_mat_for_pager(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu);

// Pool budget in bytes: GGML_CUDA_PAGER_MIB (default 0).
size_t ggml_cuda_pager_budget_bytes();

// supports_op for an op with a paged source: true only for MUL_MAT_ID with the paged tensor as src[0] that the pager can run
bool ggml_cuda_pager_supports_op(int device, const ggml_tensor * op);

// Execute the fused block starting at node i (MUL_MAT_ID gate, MUL_MAT_ID up, GLU swiglu, MUL_MAT_ID down) when it matches.
// Returns the number of extra nodes consumed (3), or 0 when the nodes at i are not such a block.
int  ggml_cuda_pager_try_block(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
// Execute a single MUL_MAT_ID node whose src[0] is paged.
void ggml_cuda_pager_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_pager_destroy(ggml_cuda_pager * pager);

// proc addresses (see ggml-pager.h)
void ggml_cuda_pager_set_cpu_kernels(const ggml_pager_cpu_kernels * kernels);
void ggml_cuda_pager_get_stats(ggml_pager_stats * out);
