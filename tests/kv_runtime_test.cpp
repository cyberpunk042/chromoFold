#include "chromofold/kv_runtime.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

const void *dummy() { return reinterpret_cast<const void *>(uintptr_t{0x1000}); }

template <typename T>
const T *ptr() {
    return static_cast<const T *>(dummy());
}

cf_kv_page_view page(uint32_t begin, uint32_t count, uint32_t layer = 0, uint32_t head = 0) {
    cf_kv_page_view result{};
    result.struct_size = sizeof(result);
    result.abi_version = CF_KV_RUNTIME_ABI_VERSION;
    result.flags = CF_KV_PAGE_FLAG_SEALED | CF_KV_PAGE_FLAG_DEVICE_RESIDENT;
    result.layer = layer;
    result.kv_head = head;
    result.token_begin = begin;
    result.token_count = count;
    result.head_dim = 4;
    result.block_size = 4;
    result.k_words = ptr<uint32_t>();
    result.k_word_count = 3;
    result.k_block_offsets = ptr<int32_t>();
    result.k_block_count = count;
    result.k_lut = ptr<int32_t>();
    result.k_lut_count = 8;
    result.k_scales = ptr<float>();
    result.k_scale_count = 4;
    result.v_words = ptr<uint32_t>();
    result.v_word_count = 4;
    result.v_block_offsets = ptr<int32_t>();
    result.v_block_count = count;
    result.v_lut = ptr<int32_t>();
    result.v_lut_count = 8;
    result.v_scales = ptr<float>();
    result.v_scale_count = count;
    return result;
}

void test_page_validation_and_accounting() {
    cf_kv_page_view valid = page(0, 4);
    assert(cf_kv_validate_page(&valid) == CF_OK);

    uint64_t bytes = 0;
    assert(cf_kv_page_resident_bytes(&valid, &bytes) == CF_OK);
    const uint64_t expected =
        (3 + 4) * sizeof(uint32_t) +
        (4 + 4 + 8 + 8) * sizeof(int32_t) +
        (4 + 4) * sizeof(float);
    assert(bytes == expected);

    cf_kv_page_view invalid = valid;
    invalid.v_scale_count = 3;
    assert(cf_kv_validate_page(&invalid) == CF_ERR_INVALID_ARGUMENT);

    invalid = valid;
    invalid.flags = 0;
    assert(cf_kv_validate_page(&invalid) == CF_ERR_INVALID_ARGUMENT);
}

void test_layout_and_overlap_detection() {
    std::vector<cf_kv_page_view> pages{page(0, 4), page(4, 4)};
    cf_kv_cache_layout layout{};
    layout.struct_size = sizeof(layout);
    layout.abi_version = CF_KV_RUNTIME_ABI_VERSION;
    layout.layer_count = 1;
    layout.kv_head_count = 1;
    layout.head_dim = 4;
    layout.page_capacity_tokens = 4;
    layout.active_token_begin = 8;
    layout.active_token_count = 2;
    layout.active_element_bytes = 2;
    layout.pages = pages.data();
    layout.page_count = pages.size();
    layout.active_k_bytes = 2 * 1 * 1 * 4 * 2;
    layout.active_v_bytes = layout.active_k_bytes;

    assert(cf_kv_validate_layout(&layout) == CF_OK);

    cf_kv_memory_stats stats{};
    assert(cf_kv_cache_memory_stats(&layout, &stats) == CF_OK);
    assert(stats.page_count == 2);
    assert(stats.sealed_tokens == 8);
    assert(stats.active_tokens == 2);
    assert(stats.active_payload_bytes == layout.active_k_bytes + layout.active_v_bytes);
    assert(stats.total_resident_bytes == stats.sealed_payload_bytes + stats.sealed_descriptor_bytes +
                                             stats.active_payload_bytes);

    pages[1].token_begin = 3;
    assert(cf_kv_validate_layout(&layout) == CF_ERR_INVALID_ARGUMENT);

    pages[1].token_begin = 4;
    layout.active_k_bytes = 1;
    assert(cf_kv_validate_layout(&layout) == CF_ERR_INVALID_ARGUMENT);
}

std::vector<float> dense_softmax(const std::vector<double> &scores,
                                 const std::vector<std::vector<float>> &values) {
    double maximum = scores[0];
    for (double score : scores) maximum = std::max(maximum, score);
    double sum = 0.0;
    std::vector<double> weighted(values[0].size(), 0.0);
    for (size_t i = 0; i < scores.size(); ++i) {
        const double weight = std::exp(scores[i] - maximum);
        sum += weight;
        for (size_t d = 0; d < weighted.size(); ++d) weighted[d] += weight * values[i][d];
    }
    std::vector<float> result(weighted.size());
    for (size_t d = 0; d < weighted.size(); ++d) result[d] = static_cast<float>(weighted[d] / sum);
    return result;
}

void test_online_softmax_page_merge() {
    const std::vector<double> scores{-1000.0, -2.0, 3.0, 2.5, 900.0, 899.0};
    const std::vector<std::vector<float>> values{
        {1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f},
        {0.f, 0.f, 0.f, 1.f}, {1.f, 2.f, 3.f, 4.f}, {4.f, 3.f, 2.f, 1.f}};

    double first_values[4];
    double second_values[4];
    double merged_values[4];
    cf_online_softmax_state first{};
    cf_online_softmax_state second{};
    cf_online_softmax_state merged{};
    assert(cf_online_softmax_init(&first, first_values, 4) == CF_OK);
    assert(cf_online_softmax_init(&second, second_values, 4) == CF_OK);
    assert(cf_online_softmax_init(&merged, merged_values, 4) == CF_OK);

    for (size_t i = 0; i < 3; ++i) {
        assert(cf_online_softmax_accumulate(&first, scores[i], values[i].data()) == CF_OK);
    }
    for (size_t i = 3; i < scores.size(); ++i) {
        assert(cf_online_softmax_accumulate(&second, scores[i], values[i].data()) == CF_OK);
    }
    assert(cf_online_softmax_merge(&merged, &first) == CF_OK);
    assert(cf_online_softmax_merge(&merged, &second) == CF_OK);

    float actual[4];
    assert(cf_online_softmax_finalize(&merged, actual) == CF_OK);
    const std::vector<float> expected = dense_softmax(scores, values);
    for (size_t d = 0; d < expected.size(); ++d) {
        assert(std::abs(actual[d] - expected[d]) < 1e-6f);
    }

    cf_online_softmax_state empty{};
    double empty_values[4];
    assert(cf_online_softmax_init(&empty, empty_values, 4) == CF_OK);
    assert(cf_online_softmax_finalize(&empty, actual) == CF_ERR_INVALID_ARGUMENT);
}

}  // namespace

int main() {
    test_page_validation_and_accounting();
    test_layout_and_overlap_detection();
    test_online_softmax_page_merge();
    std::cout << "M9 KV runtime tests passed\n";
    return 0;
}
