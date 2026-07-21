#include "chromofold/kv_cuda.h"
#include "chromofold/detail/block_huffman_device.cuh"

#include <cuda_runtime.h>
#include <cmath>
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

            int k_bit = row_bit_position(page.k_words, page.k_block_offsets, page.k_lut,
                                         static_cast<int>(page.k_max_code_length),
                                         static_cast<int>(page.block_size),
                                         static_cast<long>(local) * page.head_dim);
            float score = 0.0f;
            for (uint32_t d = 0; d < desc.head_dim; ++d) {
                const int symbol_length = cf_bh_decode_at(page.k_words, page.k_lut,
                                                          static_cast<int>(page.k_max_code_length), k_bit);
                const int quantized = symbol_length & 0xff;
                k_bit += symbol_length >> 8;
                score += query[d] * (static_cast<float>(quantized - static_cast<int>(page.zero_point)) *
                                     page.k_scales[d]);
            }
            score *= desc.softmax_scale;

            float value[CF_KV_MAX_HEAD_DIM];
            int v_bit = row_bit_position(page.v_words, page.v_block_offsets, page.v_lut,
                                         static_cast<int>(page.v_max_code_length),
                                         static_cast<int>(page.block_size),
                                         static_cast<long>(local) * page.head_dim);
            const float v_scale = page.v_scales[local];
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

}  // namespace

extern "C" cf_status cf_kv_validate_paged_attention_desc(const cf_kv_paged_attention_desc *desc) {
    if (desc == nullptr || desc->struct_size != sizeof(cf_kv_paged_attention_desc) ||
        desc->abi_version != CF_KV_CUDA_ABI_VERSION || desc->queries == nullptr || desc->output == nullptr ||
        desc->query_count == 0 || desc->query_head_count == 0 || desc->kv_head_count == 0 ||
        desc->gqa_group_size == 0 || desc->head_dim == 0 || desc->head_dim > CF_KV_MAX_HEAD_DIM ||
        !std::isfinite(desc->softmax_scale)) {
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
    constexpr uint32_t threads = 64;
    paged_attention_kernel<<<(work + threads - 1) / threads, threads, 0, cuda_stream>>>(*desc);
    return cudaPeekAtLastError() == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}
