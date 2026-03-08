#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>
#include <stdexcept>
#include <stdint.h>
#include <string>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int32_t TOKEN_HIT = 0xFFFFFFFF;

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
  }
}

__device__ __forceinline__ int warp_inclusive_scan(int* s_data, int lane_id, int offset, int count, int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;

#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  accumulator = __shfl_sync(0xffffffff, val, 31);
  return accumulator;
}

// Each block processes one request
// IdxType: type for req_pool_indices and seq_lens (int32_t or int64_t), The cuda graph mode requires int32_t
// Layout: [HOT_BUFFER_SIZE slots for LRU] + [page_size slots for newest token]
// newest_slot is at HOT_BUFFER_SIZE (first position of extra page)
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA, typename IdxType>
__global__ void load_cache_to_device_buffer_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int32_t* __restrict__ top_k_device_locs,
    int16_t* __restrict__ diff_map,
    const IdxType* __restrict__ req_pool_indices,
    const IdxType* __restrict__ seq_lens,
    int16_t* __restrict__ lru_slots,
    const int32_t* __restrict__ num_real_reqs,
    int64_t buffer_stride_0,
    int64_t buffer_stride_1,
    int64_t host_stride,
    int64_t diff_map_stride,
    int64_t lru_slot_stride_0,
    int64_t lru_slot_stride_1,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t page_size,
    int64_t layer_id,
    int64_t item_size_bytes) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  // LRU uses all HOT_BUFFER_SIZE slots (newest_slot is now outside at HOT_BUFFER_SIZE)
  constexpr int LRU_SIZE = HOT_BUFFER_SIZE;
  constexpr int NUM_BUFFER_CHUNKS = (LRU_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // Early exit for padded blocks (CUDA graph pads batch to a captured size)
  if (bid >= num_real_reqs[0]) return;

  const int64_t rid = req_pool_indices[bid];
  const int64_t seq_len = seq_lens[bid];
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  // Calculate offsets for this request
  const int top_k_tokens_offset = bid * top_k_tokens_stride;
  const int top_k_device_locs_offset = bid * top_k_device_locs_stride;
  const int diff_map_offset = bid * diff_map_stride;
  const int32_t* my_top_k_tokens = top_k_tokens + top_k_tokens_offset;
  int32_t* my_top_k_device_locs = top_k_device_locs + top_k_device_locs_offset;
  int16_t* my_diff_map = diff_map + diff_map_offset;

  const int buffer_offset = rid * buffer_stride_0 + layer_id * buffer_stride_1;
  const int host_offset = rid * host_stride;
  const int lru_slot_offset = rid * lru_slot_stride_0 + layer_id * lru_slot_stride_1;
  int32_t* my_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  const int32_t* my_device_buffer_locs = device_buffer_locs + buffer_offset;
  const int64_t* my_host_cache_locs = host_cache_locs + host_offset;
  int16_t* my_lru_slots = lru_slots + lru_slot_offset;

  // Fast path: short sequences have all tokens in the device buffer in order.
  // Iterate over top-k entries (not seq_len) to stay within output bounds.
  if (seq_len <= HOT_BUFFER_SIZE) {
    const int count = (seq_len < NUM_TOP_K) ? static_cast<int>(seq_len) : NUM_TOP_K;
    for (int i = tid; i < count; i += BLOCK_SIZE) {
      int32_t token_pos = my_top_k_tokens[i];
      if (token_pos >= 0 && token_pos < seq_len) {
        int32_t page_start = my_device_buffer_locs[token_pos * page_size];
        my_top_k_device_locs[i] = page_start / page_size;
      }
    }
    return;
  }

  __shared__ int32_t s_top_k_tokens[NUM_TOP_K];
  __shared__ int32_t s_chunk_offset[NUM_BUFFER_CHUNKS + 1];
  __shared__ int32_t s_missed_tokens[NUM_TOP_K];
  __shared__ int32_t s_evictable_slots[NUM_TOP_K];
  __shared__ int32_t s_total_misses;
  __shared__ int32_t s_total_hits;
  __shared__ int32_t s_total_evictable;
  __shared__ int32_t s_newest_hit;
  __shared__ bool s_lru_bitmap[HOT_BUFFER_SIZE];
  __shared__ int16_t s_lru_slots_out[LRU_SIZE];

  // Initialize shared memory counters used across phases.
  if (tid == 0) {
    s_total_misses = 0;
    s_total_hits = 0;
    s_total_evictable = 0;
    s_newest_hit = 0;
  }

  const int newest_slot = HOT_BUFFER_SIZE;
  // For page-wise topk, use page id.
  const int32_t newest_token = seq_len - 1;
  // (seq_len >= 0) ? static_cast<int32_t>((page_size > 1) ? (seq_len / page_size) : seq_len) : -1;

  // Build diff_map for top-k tokens and reset shared buffers.
  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    int32_t top_k_val = my_top_k_tokens[i];
    my_diff_map[top_k_val] = i;
    s_top_k_tokens[i] = top_k_val;
    s_evictable_slots[i] = -1;
  }
  for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
    s_lru_bitmap[i] = false;
  }

  __syncthreads();

  // If topk includes the latest token, bind its canonical occurrence to
  // newest_slot (at HOT_BUFFER_SIZE) and mark it as a hit.
  // newest_slot is at the first position of the extra page, excluded from LRU tracking.
  if (tid == 0 && newest_token >= 0 && newest_token < diff_map_stride) {
    const int newest_topk_idx = my_diff_map[newest_token];
    if (newest_topk_idx >= 0) {
      s_top_k_tokens[newest_topk_idx] = TOKEN_HIT;
      my_top_k_device_locs[newest_topk_idx] = my_device_buffer_locs[newest_slot];
      my_device_buffer_tokens[newest_slot] = newest_token;
      s_newest_hit = 1;
    }
  }
  __syncthreads();

  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < HOT_BUFFER_SIZE);
    const int32_t buf_slot = has_valid_slot ? static_cast<int32_t>(my_lru_slots[slot_idx]) : -1;
    const bool has_valid_buf_slot = has_valid_slot && (buf_slot >= 0) && (buf_slot < HOT_BUFFER_SIZE);
    int32_t my_buffer_token = has_valid_buf_slot ? my_device_buffer_tokens[buf_slot] : -1;
    int my_found_top_k_idx = my_buffer_token >= 0 ? my_diff_map[my_buffer_token] : -1;

    // Record hits
    if (my_found_top_k_idx >= 0 && has_valid_buf_slot) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      my_top_k_device_locs[my_found_top_k_idx] = my_device_buffer_locs[buf_slot];
    }
    __syncthreads();

    bool is_hit = my_found_top_k_idx != -1;
    int local_hit_offset = 0;

    // seems unnecessary
    if (warp_id == 0) {
      const int base_chunk = iter * NUM_WARPS;
      const int idx = base_chunk + lane_id + 1;
      if (idx < NUM_BUFFER_CHUNKS + 1) {
        s_chunk_offset[idx] = 0;
      }
    }
    __syncthreads();
    if (has_valid_chunk) {
      const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, is_hit);
      local_hit_offset = __popc(hit_mask & lanes_before);
      int warp_hit_count = __popc(hit_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_hit_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_hit_count =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_hit_count);
      if (tid == 0) {
        s_total_hits = total_hit_count;
      }
    }
    __syncthreads();

    if (is_hit && has_valid_buf_slot) {
      int hit_offset = s_chunk_offset[chunk_idx] + local_hit_offset;
      s_lru_slots_out[hit_offset] = buf_slot;
      s_lru_bitmap[buf_slot] = true;
    }
  }
  __syncthreads();

  // Move staged hits to the tail so hits are most recent in LRU order.
  for (int i = s_total_hits - 1 - tid; i >= 0; i -= BLOCK_SIZE) {
    const int dst = LRU_SIZE - s_total_hits + i;
    s_lru_slots_out[dst] = s_lru_slots_out[i];
  }
  __syncthreads();

  // Second pass to collect evictable slots
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  int total_evictable = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < LRU_SIZE);
    const int32_t buf_slot = has_valid_slot ? static_cast<int32_t>(my_lru_slots[slot_idx]) : -1;
    const bool has_valid_buf_slot = has_valid_slot && (buf_slot >= 0) && (buf_slot < HOT_BUFFER_SIZE);
    bool is_evictable = has_valid_buf_slot && !s_lru_bitmap[buf_slot];
    int local_evictable_offset = 0;
    if (warp_id == 0) {
      const int base_chunk = iter * NUM_WARPS;
      const int idx = base_chunk + lane_id + 1;
      if (idx < NUM_BUFFER_CHUNKS + 1) {
        s_chunk_offset[idx] = 0;
      }
    }
    __syncthreads();

    if (has_valid_chunk) {
      const unsigned int evictable_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_evictable_offset = __popc(evictable_mask & lanes_before);
      const int warp_evictable_count = __popc(evictable_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_evictable_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_evictable =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evictable);
    }
    __syncthreads();

    if (is_evictable && has_valid_buf_slot) {
      const int evictable_offset = s_chunk_offset[chunk_idx] + local_evictable_offset;
      int num_misses = NUM_TOP_K - s_total_hits - s_newest_hit;
      if (num_misses < 0) {
        num_misses = 0;
      }
      if (evictable_offset < num_misses) {
        s_evictable_slots[evictable_offset] = buf_slot;
        s_lru_slots_out[LRU_SIZE - s_total_hits - 1 - evictable_offset] = buf_slot;
      } else {
        s_lru_slots_out[evictable_offset - num_misses] = buf_slot;
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    s_total_evictable = total_evictable;
  }

  for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
    if (i < NUM_TOP_K) {
      int32_t top_k_val = my_top_k_tokens[i];
      my_diff_map[top_k_val] = -1;
    }
    if (i < LRU_SIZE) {
      my_lru_slots[i] = s_lru_slots_out[i];
    }
  }
  // Reset offsets for the miss counting phase.
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;

    const int chunk_token_start = chunk_idx * WARP_SIZE;
    const int my_token_idx = chunk_token_start + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

    int32_t my_token = 0;
    bool is_miss = false;
    int local_miss_offset = 0;

    if (has_valid_token) {
      is_miss = s_top_k_tokens[my_token_idx] != TOKEN_HIT;
      if (is_miss) {
        my_token = s_top_k_tokens[my_token_idx];
      }
    }

    // Intra-warp communication for miss counting.
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
    if (warp_id == 0) {
      const int base_chunk = iter * NUM_WARPS;
      const int idx = base_chunk + lane_id + 1;
      if (idx < NUM_TOKEN_CHUNKS + 1) {
        s_chunk_offset[idx] = 0;
      }
    }
    __syncthreads();
    if (has_valid_chunk) {
      local_miss_offset = __popc(miss_mask & lanes_before);
      const int warp_miss_count = __popc(miss_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      s_total_misses =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, s_total_misses);
    }
    __syncthreads();

    // Clamp misses to the number of available evictable slots.
    if (tid == 0 && s_total_misses > s_total_evictable) {
      s_total_misses = s_total_evictable;
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      if (miss_offset < s_total_evictable) {
        int evict_slot = s_evictable_slots[miss_offset];
        s_missed_tokens[miss_offset] = my_token;
        if (evict_slot >= 0 && evict_slot < HOT_BUFFER_SIZE) {
          my_top_k_device_locs[my_token_idx] = my_device_buffer_locs[evict_slot];
          my_device_buffer_tokens[evict_slot] = my_token;
        } else {
          my_top_k_device_locs[my_token_idx] = -1;
        }
      }
    }
    __syncthreads();
  }

  // Ensure all shared memory writes from the miss-counting phase are visible.
  __syncthreads();

  // Fused H2D transfer: each warp copies one miss directly.
  for (int miss_base = 0; miss_base < s_total_misses; miss_base += NUM_WARPS) {
    const int miss_idx = miss_base + warp_id;
    if (miss_idx < s_total_misses) {
      const int32_t miss_token = s_missed_tokens[miss_idx];
      const int evict_slot = s_evictable_slots[miss_idx];

      if (evict_slot >= 0 && evict_slot < HOT_BUFFER_SIZE && miss_token >= 0) {
        for (int p = 0; p < page_size; p++) {
          const int64_t src_loc = my_host_cache_locs[miss_token * page_size + p];
          const int64_t dst_loc = static_cast<int64_t>(my_device_buffer_locs[evict_slot]) * page_size + p;

          const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
          auto dst_k = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
          transfer_item_warp(lane_id, src_k, dst_k, item_size_bytes);

          if constexpr (!IsMLA) {
            const auto src_v = static_cast<const char*>(host_cache_v) + src_loc * item_size_bytes;
            auto dst_v = static_cast<char*>(device_buffer_v) + dst_loc * item_size_bytes;
            transfer_item_warp(lane_id, src_v, dst_v, item_size_bytes);
          }
        }
      }
    }
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
struct SparseCacheKernel {
  template <typename IdxType>
  static void
  run(tvm::ffi::TensorView top_k_tokens,
      tvm::ffi::TensorView device_buffer_tokens,
      tvm::ffi::TensorView host_cache_locs,
      tvm::ffi::TensorView device_buffer_locs,
      tvm::ffi::TensorView host_cache_k,
      tvm::ffi::TensorView host_cache_v,
      tvm::ffi::TensorView device_buffer_k,
      tvm::ffi::TensorView device_buffer_v,
      tvm::ffi::TensorView top_k_device_locs,
      tvm::ffi::TensorView diff_map,
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView lru_slots,
      tvm::ffi::TensorView num_real_reqs,
      int64_t page_size,
      int64_t layer_id,
      int64_t item_size_bytes) {
    using namespace host;

    const int64_t bs = top_k_tokens.shape()[0];
    const int64_t host_stride = host_cache_locs.shape()[1];
    const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
    const int64_t buffer_stride_1 = device_buffer_tokens.strides()[1];
    const int64_t diff_map_stride = diff_map.shape()[1];
    const int64_t lru_slot_stride_0 = lru_slots.strides()[0];
    const int64_t lru_slot_stride_1 = lru_slots.strides()[1];
    const int64_t top_k_tokens_stride = top_k_tokens.strides()[0];
    const int64_t top_k_device_locs_stride = top_k_device_locs.strides()[0];

    const int32_t* top_k_tokens_ptr = static_cast<const int32_t*>(top_k_tokens.data_ptr());
    int32_t* device_buffer_tokens_ptr = static_cast<int32_t*>(device_buffer_tokens.data_ptr());
    const int64_t* host_cache_locs_ptr = static_cast<const int64_t*>(host_cache_locs.data_ptr());
    const int32_t* device_buffer_locs_ptr = static_cast<const int32_t*>(device_buffer_locs.data_ptr());
    const void* host_cache_k_ptr = host_cache_k.data_ptr();
    const void* host_cache_v_ptr = (IsMLA || host_cache_v.ndim() == 0) ? nullptr : host_cache_v.data_ptr();
    void* device_buffer_k_ptr = device_buffer_k.data_ptr();
    void* device_buffer_v_ptr = (IsMLA || device_buffer_v.ndim() == 0) ? nullptr : device_buffer_v.data_ptr();
    int32_t* top_k_device_locs_ptr = static_cast<int32_t*>(top_k_device_locs.data_ptr());
    int16_t* diff_map_ptr = static_cast<int16_t*>(diff_map.data_ptr());
    const IdxType* req_pool_indices_ptr = static_cast<const IdxType*>(req_pool_indices.data_ptr());
    const IdxType* seq_lens_ptr = static_cast<const IdxType*>(seq_lens.data_ptr());
    int16_t* lru_slots_ptr = static_cast<int16_t*>(lru_slots.data_ptr());
    const int32_t* num_real_reqs_ptr = static_cast<const int32_t*>(num_real_reqs.data_ptr());

    const auto device = LaunchKernel::resolve_device(top_k_tokens.device());

    LaunchKernel(bs, BLOCK_SIZE, device)(
        load_cache_to_device_buffer_kernel<BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA, IdxType>,
        top_k_tokens_ptr,
        device_buffer_tokens_ptr,
        host_cache_locs_ptr,
        device_buffer_locs_ptr,
        host_cache_k_ptr,
        host_cache_v_ptr,
        device_buffer_k_ptr,
        device_buffer_v_ptr,
        top_k_device_locs_ptr,
        diff_map_ptr,
        req_pool_indices_ptr,
        seq_lens_ptr,
        lru_slots_ptr,
        num_real_reqs_ptr,
        buffer_stride_0,
        buffer_stride_1,
        host_stride,
        diff_map_stride,
        lru_slot_stride_0,
        lru_slot_stride_1,
        top_k_tokens_stride,
        top_k_device_locs_stride,
        page_size,
        layer_id,
        item_size_bytes);
  }
};

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
void load_cache_to_device_buffer(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView host_cache_v,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView device_buffer_v,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView diff_map,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView lru_slots,
    tvm::ffi::TensorView num_real_reqs,
    int64_t page_size,
    int64_t layer_id,
    int64_t item_size_bytes) {
  const auto& dtype = req_pool_indices.dtype();
  const bool is_int64 = (dtype.bits == 64);

  if (is_int64) {
    SparseCacheKernel<BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA>::template run<int64_t>(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        diff_map,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        layer_id,
        item_size_bytes);
  } else {
    SparseCacheKernel<BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA>::template run<int32_t>(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        diff_map,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        layer_id,
        item_size_bytes);
  }
}

}  // namespace
