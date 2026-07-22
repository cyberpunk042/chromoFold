#include "chromofold/kv_cuda.h"
#include "chromofold/detail/block_huffman_device.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>

namespace {

__device__ __forceinline__ int row_bit_position(const uint32_t *words, const int32_t *offsets,
                                                const int32_t *lut, int max_length, int block_size,
                                                long flat_index) {
    const int block = static_cast<int>(flat_index / block_size);
    int bit = offsets[block];
    const int skip = static_cast<int>(flat_index - static_cast<long>(block) * block_size);
    for (int i = 0; i < skip; ++i) {
        const int symbol_length = cf_bh_decode_at(words, lut, max_length, bit);
        bit += symbol_length >> 8;
    }
    return bit;
}

__device__ __forceinline__ void online_add(float score, const float *value, int dim,
                                           float *maximum, float *sum, float *weighted,
                                           int *initialized) {
    if (!*initialized) {
        *maximum = score;
        *sum = 1.0f;
        for (int d = 0; d < dim; ++d) weighted[d] = value[d];
        *initialized = 1;
        return;
    }
    const float next_max = fmaxf(*maximum, score);
    const float old_scale = expf(*maximum - next_max);
    const float new_scale = expf(score - next_max);
    *sum = *sum * old_scale + new_scale;
    for (int d = 0; d < dim; ++d) weighted[d] = weighted[d] * old_scale + value[d] * new_scale;
    *maximum = next_max;
}

__global__ void paged_attention_kernel(cf_kv_paged_attention_desc desc) {
    const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t work = desc.query_count * desc.query_head_count;
    if (linear >= work) return;

    const uint32_t query_row = linear / desc.query_head_count;
    const uint32_t query_head = linear % desc.query_head_count;
    const uint32_t kv_head = query_head / desc.gqa_group_size;
    const uint32_t query_token = desc.query_token_begin + query_row;
    const uint32_t window_begin = desc.causal_window == 0 || query_token + 1 <= desc.causal_window
                                      ? 0u
                                      : query_token + 1 - desc.causal_window;

    float query[CF_KV_MAX_HEAD_DIM];
    float weighted[CF_KV_MAX_HEAD_DIM];
    const long query_base = (static_cast<long>(query_row) * desc.query_head_count + query_head) * desc.head_dim;
    for (uint32_t d = 0; d < desc.head_dim; ++d) {
        query[d] = desc.queries[query_base + d];
        weighted[d] = 0.0f;
    }

    float maximum = -CUDART_INF_F;
    float sum = 0.0f;
    int initialized = 0;

    for (uint32_t p = 0; p < desc.page_count; ++p) {
        const cf_kv_device_page page = desc.pages[p];
        if (page.kv_head != kv_head) continue;
        for (uint32_t local = 0; local < page.token_count; ++local) {
            const uint32_t token = page.token_begin + local;
            if (token < window_begin || token > query_token) continue;

            float score = 0.0f;
            if (page.fixed_code_length) {
                // Fixed-width fast path: read each 32-bit word once (a symbol never crosses a word since bits|32),
                // extract head_dim symbols directly; the row's start bit is exact (no forward decode).
                const uint32_t bits = page.fixed_code_length;
                const uint32_t mask = (1u << bits) - 1u;
                const long bit0 = static_cast<long>(local) * page.head_dim * bits;
                uint32_t widx = static_cast<uint32_t>(bit0 >> 5);
                uint32_t word = page.k_words[widx];
                for (uint32_t d = 0; d < desc.head_dim; ++d) {
                    const long bit = bit0 + static_cast<long>(d) * bits;
                    const uint32_t wi = static_cast<uint32_t>(bit >> 5);
                    if (wi != widx) { widx = wi; word = page.k_words[wi]; }
                    const uint32_t shift = (32u - bits) - static_cast<uint32_t>(bit & 31);
                    const int quantized = static_cast<int>((word >> shift) & mask);
                    score += query[d] * (static_cast<float>(quantized - static_cast<int>(page.zero_point)) * page.k_scales[d]);
                }
            } else {
                int k_bit = row_bit_position(page.k_words, page.k_block_offsets, page.k_lut,
                                             static_cast<int>(page.k_max_code_length),
                                             static_cast<int>(page.block_size),
                                             static_cast<long>(local) * page.head_dim);
                for (uint32_t d = 0; d < desc.head_dim; ++d) {
                    const int symbol_length = cf_bh_decode_at(page.k_words, page.k_lut,
                                                              static_cast<int>(page.k_max_code_length), k_bit);
                    const int quantized = symbol_length & 0xff;
                    k_bit += symbol_length >> 8;
                    score += query[d] * (static_cast<float>(quantized - static_cast<int>(page.zero_point)) * page.k_scales[d]);
                }
            }
            score *= desc.softmax_scale;

            const float v_scale = page.v_scales[local];
            if (page.fixed_code_length) {
                // Fused: decode V and apply the online-softmax rescale in one pass — no value[] temp (drops a
                // CF_KV_MAX_HEAD_DIM local array). Same semantics as online_add(), inlined.
                const uint32_t bits = page.fixed_code_length;
                const uint32_t mask = (1u << bits) - 1u;
                const long bit0 = static_cast<long>(local) * page.head_dim * bits;
                uint32_t widx = static_cast<uint32_t>(bit0 >> 5);
                uint32_t word = page.v_words[widx];
                const int zp = static_cast<int>(page.zero_point);
                if (!initialized) {
                    maximum = score; sum = 1.0f; initialized = 1;
                    for (uint32_t d = 0; d < desc.head_dim; ++d) {
                        const long bit = bit0 + static_cast<long>(d) * bits;
                        const uint32_t wi = static_cast<uint32_t>(bit >> 5);
                        if (wi != widx) { widx = wi; word = page.v_words[wi]; }
                        const uint32_t shift = (32u - bits) - static_cast<uint32_t>(bit & 31);
                        weighted[d] = static_cast<float>(static_cast<int>((word >> shift) & mask) - zp) * v_scale;
                    }
                } else {
                    const float nm = fmaxf(maximum, score);
                    const float a = __expf(maximum - nm), b = __expf(score - nm);
                    sum = sum * a + b;
                    for (uint32_t d = 0; d < desc.head_dim; ++d) {
                        const long bit = bit0 + static_cast<long>(d) * bits;
                        const uint32_t wi = static_cast<uint32_t>(bit >> 5);
                        if (wi != widx) { widx = wi; word = page.v_words[wi]; }
                        const uint32_t shift = (32u - bits) - static_cast<uint32_t>(bit & 31);
                        const float vdeq = static_cast<float>(static_cast<int>((word >> shift) & mask) - zp) * v_scale;
                        weighted[d] = weighted[d] * a + vdeq * b;
                    }
                    maximum = nm;
                }
            } else {
                float value[CF_KV_MAX_HEAD_DIM];
                int v_bit = row_bit_position(page.v_words, page.v_block_offsets, page.v_lut,
                                             static_cast<int>(page.v_max_code_length),
                                             static_cast<int>(page.block_size),
                                             static_cast<long>(local) * page.head_dim);
                for (uint32_t d = 0; d < desc.head_dim; ++d) {
                    const int symbol_length = cf_bh_decode_at(page.v_words, page.v_lut,
                                                              static_cast<int>(page.v_max_code_length), v_bit);
                    const int quantized = symbol_length & 0xff;
                    v_bit += symbol_length >> 8;
                    value[d] = static_cast<float>(quantized - static_cast<int>(page.zero_point)) * v_scale;
                }
                online_add(score, value, static_cast<int>(desc.head_dim), &maximum, &sum, weighted, &initialized);
            }
        }
    }

    if (desc.active_token_count != 0) {
        for (uint32_t local = 0; local < desc.active_token_count; ++local) {
            const uint32_t token = desc.active_token_begin + local;
            if (token < window_begin || token > query_token) continue;
            const long base = (static_cast<long>(local) * desc.kv_head_count + kv_head) * desc.head_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < desc.head_dim; ++d) score += query[d] * desc.active_k[base + d];
            score *= desc.softmax_scale;
            online_add(score, desc.active_v + base, static_cast<int>(desc.head_dim),
                       &maximum, &sum, weighted, &initialized);
        }
    }

    const long out_base = (static_cast<long>(query_row) * desc.query_head_count + query_head) * desc.head_dim;
    if (!initialized || !(sum > 0.0f)) {
        for (uint32_t d = 0; d < desc.head_dim; ++d) desc.output[out_base + d] = 0.0f;
        return;
    }
    const float inverse = 1.0f / sum;
    for (uint32_t d = 0; d < desc.head_dim; ++d) desc.output[out_base + d] = weighted[d] * inverse;
}

// Warp-cooperative variant: one warp per (query_row, query_head), head_dim split across the 32 lanes (DPL dims
// each, strided dim = lane + 32*i for coalesced reads). Per-lane q[]/w[] are register-resident (no local arrays);
// the QK dot product is a butterfly-shuffle reduction. All lanes share query_token/kv_head, so the causal check
// and page filter are warp-uniform (no divergence). Assumes fixed-width codes (all current codecs); a variable
// Huffman codec must route to the scalar kernel above.
template <int DPL>
__global__ void paged_attention_warp_kernel(cf_kv_paged_attention_desc desc) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x & 31u);
    if (warp >= desc.query_count * desc.query_head_count) return;

    const uint32_t query_row = warp / desc.query_head_count;
    const uint32_t query_head = warp % desc.query_head_count;
    const uint32_t kv_head = query_head / desc.gqa_group_size;
    const uint32_t query_token = desc.query_token_begin + query_row;
    const uint32_t window_begin = desc.causal_window == 0 || query_token + 1 <= desc.causal_window
                                      ? 0u : query_token + 1 - desc.causal_window;
    const uint32_t head_dim = static_cast<uint32_t>(DPL) * 32u;

    float q[DPL], w[DPL];
    const long qbase = (static_cast<long>(query_row) * desc.query_head_count + query_head) * head_dim;
#pragma unroll
    for (int i = 0; i < DPL; ++i) { q[i] = desc.queries[qbase + lane + 32 * i]; w[i] = 0.0f; }

    float maximum = -CUDART_INF_F, sum = 0.0f;
    int initialized = 0;

    for (uint32_t p = 0; p < desc.page_count; ++p) {
        const cf_kv_device_page page = desc.pages[p];
        if (page.kv_head != kv_head) continue;
        const uint32_t bits = page.fixed_code_length;
        const uint32_t mask = (1u << bits) - 1u;
        const int zp = static_cast<int>(page.zero_point);
        for (uint32_t local = 0; local < page.token_count; ++local) {
            const uint32_t token = page.token_begin + local;
            if (token < window_begin || token > query_token) continue;
            const long rowbit = static_cast<long>(local) * head_dim * bits;
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; ++i) {
                const int d = lane + 32 * i;
                const long bit = rowbit + static_cast<long>(d) * bits;
                const uint32_t word = page.k_words[bit >> 5];
                const uint32_t shift = (32u - bits) - static_cast<uint32_t>(bit & 31);
                partial += q[i] * (static_cast<float>(static_cast<int>((word >> shift) & mask) - zp) * page.k_scales[d]);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, o);
            const float score = partial * desc.softmax_scale;

            const float v_scale = page.v_scales[local];
            const float nm = initialized ? fmaxf(maximum, score) : score;
            const float a = initialized ? __expf(maximum - nm) : 0.0f;
            const float b = initialized ? __expf(score - nm) : 1.0f;
            sum = initialized ? sum * a + b : 1.0f;
#pragma unroll
            for (int i = 0; i < DPL; ++i) {
                const int d = lane + 32 * i;
                const long bit = rowbit + static_cast<long>(d) * bits;
                const uint32_t word = page.v_words[bit >> 5];
                const uint32_t shift = (32u - bits) - static_cast<uint32_t>(bit & 31);
                const float vdeq = static_cast<float>(static_cast<int>((word >> shift) & mask) - zp) * v_scale;
                w[i] = initialized ? w[i] * a + vdeq * b : vdeq;
            }
            maximum = nm;
            initialized = 1;
        }
    }

    if (desc.active_token_count != 0) {
        for (uint32_t local = 0; local < desc.active_token_count; ++local) {
            const uint32_t token = desc.active_token_begin + local;
            if (token < window_begin || token > query_token) continue;
            const long base = (static_cast<long>(local) * desc.kv_head_count + kv_head) * head_dim;
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; ++i) partial += q[i] * desc.active_k[base + lane + 32 * i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, o);
            const float score = partial * desc.softmax_scale;
            const float nm = initialized ? fmaxf(maximum, score) : score;
            const float a = initialized ? __expf(maximum - nm) : 0.0f;
            const float b = initialized ? __expf(score - nm) : 1.0f;
            sum = initialized ? sum * a + b : 1.0f;
#pragma unroll
            for (int i = 0; i < DPL; ++i) {
                const float vv = desc.active_v[base + lane + 32 * i];
                w[i] = initialized ? w[i] * a + vv * b : vv;
            }
            maximum = nm;
            initialized = 1;
        }
    }

    const long obase = (static_cast<long>(query_row) * desc.query_head_count + query_head) * head_dim;
    const float inv = (initialized && sum > 0.0f) ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) desc.output[obase + lane + 32 * i] = w[i] * inv;
}

}  // namespace

extern "C" cf_status cf_kv_validate_paged_attention_desc(const cf_kv_paged_attention_desc *desc) {
    if (desc == nullptr || desc->struct_size != sizeof(cf_kv_paged_attention_desc) ||
        desc->abi_version != CF_KV_CUDA_ABI_VERSION || desc->queries == nullptr || desc->output == nullptr ||
        desc->query_count == 0 || desc->query_head_count == 0 || desc->kv_head_count == 0 ||
        desc->gqa_group_size == 0 || desc->head_dim == 0 || desc->head_dim > CF_KV_MAX_HEAD_DIM ||
        !::isfinite(desc->softmax_scale)) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if (desc->query_head_count != desc->kv_head_count * desc->gqa_group_size ||
        desc->page_count > CF_KV_CUDA_MAX_PAGES_PER_LAUNCH ||
        (desc->page_count != 0 && desc->pages == nullptr) ||
        (desc->active_token_count != 0 && (desc->active_k == nullptr || desc->active_v == nullptr))) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if (desc->query_token_begin > UINT32_MAX - desc->query_count ||
        desc->active_token_begin > UINT32_MAX - desc->active_token_count) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    return CF_OK;
}

extern "C" cf_status cf_kv_paged_attention_async(const cf_kv_paged_attention_desc *desc, void *stream) {
    const cf_status valid = cf_kv_validate_paged_attention_desc(desc);
    if (valid != CF_OK) return valid;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const uint32_t work = desc->query_count * desc->query_head_count;
    if (desc->head_dim % 32u == 0u && desc->head_dim <= CF_KV_MAX_HEAD_DIM) {
        // Warp-cooperative path (fixed-width codecs): one warp per (query, head), 4 warps/block.
        const uint32_t warps_per_block = 4u;
        const uint32_t threads = warps_per_block * 32u;
        const uint32_t blocks = (work + warps_per_block - 1u) / warps_per_block;
        switch (desc->head_dim / 32u) {
            case 1: paged_attention_warp_kernel<1><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 2: paged_attention_warp_kernel<2><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 3: paged_attention_warp_kernel<3><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 4: paged_attention_warp_kernel<4><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 5: paged_attention_warp_kernel<5><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 6: paged_attention_warp_kernel<6><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 7: paged_attention_warp_kernel<7><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            case 8: paged_attention_warp_kernel<8><<<blocks, threads, 0, cuda_stream>>>(*desc); break;
            default: break;
        }
    } else {
        constexpr uint32_t threads = 64;
        paged_attention_kernel<<<(work + threads - 1) / threads, threads, 0, cuda_stream>>>(*desc);
    }
    return cudaPeekAtLastError() == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}
