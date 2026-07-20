#include "chromofold/kv_runtime.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace {

bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == nullptr || a > std::numeric_limits<uint64_t>::max() - b) return false;
    *out = a + b;
    return true;
}

bool mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool counted_pointer(const void *pointer, uint64_t count) {
    return count == 0 || pointer != nullptr;
}

bool finite(double value) {
    return std::isfinite(value);
}

cf_status add_array_bytes(uint64_t count, uint64_t element_size, uint64_t *total) {
    uint64_t bytes = 0;
    uint64_t next = 0;
    if (!mul_u64(count, element_size, &bytes) || !add_u64(*total, bytes, &next)) {
        return CF_ERR_UNSUPPORTED;
    }
    *total = next;
    return CF_OK;
}

}  // namespace

extern "C" cf_status cf_kv_validate_page(const cf_kv_page_view *page) {
    if (page == nullptr || page->struct_size != sizeof(cf_kv_page_view) ||
        page->abi_version != CF_KV_RUNTIME_ABI_VERSION) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if ((page->flags & CF_KV_PAGE_FLAG_SEALED) == 0 || page->token_count == 0 ||
        page->head_dim == 0 || page->head_dim > CF_KV_MAX_HEAD_DIM || page->block_size == 0) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if (!counted_pointer(page->k_words, page->k_word_count) || page->k_word_count == 0 ||
        !counted_pointer(page->v_words, page->v_word_count) || page->v_word_count == 0 ||
        !counted_pointer(page->k_block_offsets, page->k_block_count) || page->k_block_count == 0 ||
        !counted_pointer(page->v_block_offsets, page->v_block_count) || page->v_block_count == 0 ||
        !counted_pointer(page->k_lut, page->k_lut_count) || page->k_lut_count == 0 ||
        !counted_pointer(page->v_lut, page->v_lut_count) || page->v_lut_count == 0 ||
        !counted_pointer(page->k_scales, page->k_scale_count) ||
        !counted_pointer(page->v_scales, page->v_scale_count)) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if (page->k_scale_count != page->head_dim || page->v_scale_count != page->token_count) {
        return CF_ERR_INVALID_ARGUMENT;
    }

    uint64_t values = 0;
    if (!mul_u64(page->token_count, page->head_dim, &values)) return CF_ERR_UNSUPPORTED;
    const uint64_t minimum_blocks = (values + page->block_size - 1u) / page->block_size;
    if (page->k_block_count < minimum_blocks || page->v_block_count < minimum_blocks) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if (page->token_begin > std::numeric_limits<uint32_t>::max() - page->token_count) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    return CF_OK;
}

extern "C" cf_status cf_kv_validate_layout(const cf_kv_cache_layout *layout) {
    if (layout == nullptr || layout->struct_size != sizeof(cf_kv_cache_layout) ||
        layout->abi_version != CF_KV_RUNTIME_ABI_VERSION || layout->layer_count == 0 ||
        layout->kv_head_count == 0 || layout->head_dim == 0 ||
        layout->head_dim > CF_KV_MAX_HEAD_DIM || layout->page_capacity_tokens == 0 ||
        layout->active_element_bytes == 0 || layout->active_element_bytes > 8 ||
        layout->active_token_count > layout->page_capacity_tokens ||
        (layout->page_count != 0 && layout->pages == nullptr)) {
        return CF_ERR_INVALID_ARGUMENT;
    }

    uint64_t expected_active_bytes = 0;
    uint64_t active_values = 0;
    if (!mul_u64(layout->active_token_count, layout->layer_count, &active_values) ||
        !mul_u64(active_values, layout->kv_head_count, &active_values) ||
        !mul_u64(active_values, layout->head_dim, &active_values) ||
        !mul_u64(active_values, layout->active_element_bytes, &expected_active_bytes)) {
        return CF_ERR_UNSUPPORTED;
    }
    if (layout->active_k_bytes < expected_active_bytes || layout->active_v_bytes < expected_active_bytes) {
        return CF_ERR_INVALID_ARGUMENT;
    }

    for (uint64_t i = 0; i < layout->page_count; ++i) {
        const cf_kv_page_view &page = layout->pages[i];
        const cf_status status = cf_kv_validate_page(&page);
        if (status != CF_OK) return status;
        if (page.layer >= layout->layer_count || page.kv_head >= layout->kv_head_count ||
            page.head_dim != layout->head_dim || page.token_count > layout->page_capacity_tokens ||
            page.token_begin + page.token_count > layout->active_token_begin) {
            return CF_ERR_INVALID_ARGUMENT;
        }

        for (uint64_t j = i + 1; j < layout->page_count; ++j) {
            const cf_kv_page_view &other = layout->pages[j];
            if (other.layer != page.layer || other.kv_head != page.kv_head) continue;
            const uint64_t page_end = static_cast<uint64_t>(page.token_begin) + page.token_count;
            const uint64_t other_end = static_cast<uint64_t>(other.token_begin) + other.token_count;
            if (page.token_begin < other_end && other.token_begin < page_end) {
                return CF_ERR_INVALID_ARGUMENT;
            }
        }
    }
    return CF_OK;
}

extern "C" cf_status cf_kv_page_resident_bytes(const cf_kv_page_view *page, uint64_t *out_bytes) {
    if (out_bytes == nullptr) return CF_ERR_INVALID_ARGUMENT;
    const cf_status status = cf_kv_validate_page(page);
    if (status != CF_OK) return status;

    uint64_t total = 0;
    if (add_array_bytes(page->k_word_count, sizeof(uint32_t), &total) != CF_OK ||
        add_array_bytes(page->v_word_count, sizeof(uint32_t), &total) != CF_OK ||
        add_array_bytes(page->k_block_count, sizeof(int32_t), &total) != CF_OK ||
        add_array_bytes(page->v_block_count, sizeof(int32_t), &total) != CF_OK ||
        add_array_bytes(page->k_lut_count, sizeof(int32_t), &total) != CF_OK ||
        add_array_bytes(page->v_lut_count, sizeof(int32_t), &total) != CF_OK ||
        add_array_bytes(page->k_scale_count, sizeof(float), &total) != CF_OK ||
        add_array_bytes(page->v_scale_count, sizeof(float), &total) != CF_OK) {
        return CF_ERR_UNSUPPORTED;
    }
    *out_bytes = total;
    return CF_OK;
}

extern "C" cf_status cf_kv_cache_memory_stats(const cf_kv_cache_layout *layout, cf_kv_memory_stats *out_stats) {
    if (out_stats == nullptr) return CF_ERR_INVALID_ARGUMENT;
    const cf_status status = cf_kv_validate_layout(layout);
    if (status != CF_OK) return status;

    cf_kv_memory_stats stats{};
    stats.page_count = layout->page_count;
    stats.active_tokens = layout->active_token_count;
    if (!add_u64(layout->active_k_bytes, layout->active_v_bytes, &stats.active_payload_bytes) ||
        !mul_u64(layout->page_count, sizeof(cf_kv_page_view), &stats.sealed_descriptor_bytes)) {
        return CF_ERR_UNSUPPORTED;
    }

    for (uint64_t i = 0; i < layout->page_count; ++i) {
        uint64_t page_bytes = 0;
        uint64_t next = 0;
        if (cf_kv_page_resident_bytes(&layout->pages[i], &page_bytes) != CF_OK ||
            !add_u64(stats.sealed_payload_bytes, page_bytes, &next)) {
            return CF_ERR_UNSUPPORTED;
        }
        stats.sealed_payload_bytes = next;
        if (!add_u64(stats.sealed_tokens, layout->pages[i].token_count, &next)) {
            return CF_ERR_UNSUPPORTED;
        }
        stats.sealed_tokens = next;
    }

    uint64_t total = 0;
    if (!add_u64(stats.sealed_payload_bytes, stats.sealed_descriptor_bytes, &total) ||
        !add_u64(total, stats.active_payload_bytes, &total)) {
        return CF_ERR_UNSUPPORTED;
    }
    stats.total_resident_bytes = total;
    *out_stats = stats;
    return CF_OK;
}

extern "C" cf_status cf_online_softmax_init(cf_online_softmax_state *state, double *weighted_values,
                                               uint32_t dim) {
    if (state == nullptr || weighted_values == nullptr || dim == 0 || dim > CF_KV_MAX_HEAD_DIM) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    state->max_score = -std::numeric_limits<double>::infinity();
    state->exp_sum = 0.0;
    state->weighted_values = weighted_values;
    state->dim = dim;
    state->initialized = 0;
    std::memset(weighted_values, 0, sizeof(double) * dim);
    return CF_OK;
}

extern "C" cf_status cf_online_softmax_accumulate(cf_online_softmax_state *state, double score,
                                                     const float *value) {
    if (state == nullptr || state->weighted_values == nullptr || value == nullptr || state->dim == 0 ||
        !finite(score)) {
        return CF_ERR_INVALID_ARGUMENT;
    }

    if (!state->initialized) {
        state->max_score = score;
        state->exp_sum = 1.0;
        for (uint32_t d = 0; d < state->dim; ++d) state->weighted_values[d] = value[d];
        state->initialized = 1;
        return CF_OK;
    }

    const double next_max = score > state->max_score ? score : state->max_score;
    const double old_scale = std::exp(state->max_score - next_max);
    const double new_scale = std::exp(score - next_max);
    state->exp_sum = state->exp_sum * old_scale + new_scale;
    for (uint32_t d = 0; d < state->dim; ++d) {
        state->weighted_values[d] = state->weighted_values[d] * old_scale + static_cast<double>(value[d]) * new_scale;
    }
    state->max_score = next_max;
    return CF_OK;
}

extern "C" cf_status cf_online_softmax_merge(cf_online_softmax_state *dst,
                                                const cf_online_softmax_state *src) {
    if (dst == nullptr || src == nullptr || dst->weighted_values == nullptr || src->weighted_values == nullptr ||
        dst->dim == 0 || dst->dim != src->dim) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    if (!src->initialized) return CF_OK;
    if (!dst->initialized) {
        dst->max_score = src->max_score;
        dst->exp_sum = src->exp_sum;
        dst->initialized = 1;
        std::memcpy(dst->weighted_values, src->weighted_values, sizeof(double) * dst->dim);
        return CF_OK;
    }

    const double next_max = src->max_score > dst->max_score ? src->max_score : dst->max_score;
    const double dst_scale = std::exp(dst->max_score - next_max);
    const double src_scale = std::exp(src->max_score - next_max);
    dst->exp_sum = dst->exp_sum * dst_scale + src->exp_sum * src_scale;
    for (uint32_t d = 0; d < dst->dim; ++d) {
        dst->weighted_values[d] = dst->weighted_values[d] * dst_scale + src->weighted_values[d] * src_scale;
    }
    dst->max_score = next_max;
    return CF_OK;
}

extern "C" cf_status cf_online_softmax_finalize(const cf_online_softmax_state *state, float *out) {
    if (state == nullptr || out == nullptr || state->weighted_values == nullptr || !state->initialized ||
        state->dim == 0 || !finite(state->exp_sum) || state->exp_sum <= 0.0) {
        return CF_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t d = 0; d < state->dim; ++d) {
        out[d] = static_cast<float>(state->weighted_values[d] / state->exp_sum);
    }
    return CF_OK;
}
