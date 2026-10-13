#pragma once

// Expert pager (stage 2: policy cache, background promotion with eviction; the stage 1 static fill is policy "static").
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
#include "pager_cache.h"

#include <cstdint>
#include <vector>

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
