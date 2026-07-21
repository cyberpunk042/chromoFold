#include "chromofold/kv_reference.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using chromofold::reference::ActiveKvTail;
using chromofold::reference::AttentionDiagnostics;
using chromofold::reference::AttentionShape;
using chromofold::reference::SealedKvPage;
using chromofold::reference::dequantize_page;
using chromofold::reference::paged_attention;
using chromofold::reference::seal_int4_page;

namespace {

bool near(float a, float b, float tolerance = 2e-5f) {
    return std::abs(a - b) <= tolerance * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
}

void expect_throw(const auto &callable) {
    bool threw = false;
    try {
        callable();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

std::vector<float> make_values(uint32_t tokens, uint32_t dim, float phase) {
    std::vector<float> result(static_cast<std::size_t>(tokens) * dim);
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t d = 0; d < dim; ++d) {
            const float x = static_cast<float>((t + 1) * (d + 3));
            result[static_cast<std::size_t>(t) * dim + d] =
                std::sin(x * 0.17f + phase) * (0.25f + 0.03f * static_cast<float>(d));
        }
    }
    return result;
}

void test_sealing_orientation_and_error() {
    constexpr uint32_t tokens = 9;
    constexpr uint32_t dim = 7;
    const auto k = make_values(tokens, dim, 0.1f);
    const auto v = make_values(tokens, dim, 0.7f);
    const SealedKvPage page = seal_int4_page(k.data(), v.data(), 12, tokens, 0, 1, dim);
    assert(page.valid());
    assert(page.value_count() == static_cast<std::size_t>(tokens) * dim);
    assert(page.packed_k.size() == (page.value_count() + 1) / 2);
    assert(page.k_scales.size() == dim);
    assert(page.v_scales.size() == tokens);

    std::vector<float> dk;
    std::vector<float> dv;
    dequantize_page(page, &dk, &dv);
    float max_k_error = 0.0f;
    float max_v_error = 0.0f;
    for (std::size_t i = 0; i < k.size(); ++i) {
        max_k_error = std::max(max_k_error, std::abs(k[i] - dk[i]));
        max_v_error = std::max(max_v_error, std::abs(v[i] - dv[i]));
    }
    float max_k_scale = *std::max_element(page.k_scales.begin(), page.k_scales.end());
    float max_v_scale = *std::max_element(page.v_scales.begin(), page.v_scales.end());
    assert(max_k_error <= max_k_scale * 0.51f);
    assert(max_v_error <= max_v_scale * 0.51f);
    assert(page.resident_bytes() < 2 * page.value_count() * sizeof(float));
}

std::vector<float> dense_attention(const std::vector<std::vector<float>> &keys,
                                   const std::vector<std::vector<float>> &values,
                                   const std::vector<uint32_t> &tokens,
                                   const std::vector<float> &queries,
                                   const AttentionShape &shape) {
    std::vector<float> output(static_cast<std::size_t>(shape.query_count) * shape.query_head_count * shape.head_dim);
    for (uint32_t q = 0; q < shape.query_count; ++q) {
        const uint32_t query_token = shape.query_token_begin + q;
        const uint32_t earliest = shape.causal_window == 0 || query_token + 1 <= shape.causal_window
                                      ? 0
                                      : query_token + 1 - shape.causal_window;
        for (uint32_t h = 0; h < shape.query_head_count; ++h) {
            const uint32_t kvh = static_cast<uint32_t>((static_cast<uint64_t>(h) * shape.kv_head_count) /
                                                       shape.query_head_count);
            const float *query = queries.data() + (static_cast<std::size_t>(q) * shape.query_head_count + h) * shape.head_dim;
            double maximum = -1e300;
            std::vector<double> scores;
            std::vector<std::size_t> selected;
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                if (tokens[i] < earliest || tokens[i] > query_token) continue;
                double dot = 0.0;
                const float *key = keys[kvh].data() + i * shape.head_dim;
                for (uint32_t d = 0; d < shape.head_dim; ++d) dot += query[d] * key[d];
                dot *= shape.score_scale;
                scores.push_back(dot);
                selected.push_back(i);
                maximum = std::max(maximum, dot);
            }
            assert(!selected.empty());
            double denominator = 0.0;
            std::vector<double> weighted(shape.head_dim, 0.0);
            for (std::size_t j = 0; j < selected.size(); ++j) {
                const double weight = std::exp(scores[j] - maximum);
                denominator += weight;
                const float *value = values[kvh].data() + selected[j] * shape.head_dim;
                for (uint32_t d = 0; d < shape.head_dim; ++d) weighted[d] += weight * value[d];
            }
            float *dst = output.data() + (static_cast<std::size_t>(q) * shape.query_head_count + h) * shape.head_dim;
            for (uint32_t d = 0; d < shape.head_dim; ++d) dst[d] = static_cast<float>(weighted[d] / denominator);
        }
    }
    return output;
}

void test_paged_matches_dense_with_gqa_and_tail() {
    constexpr uint32_t dim = 8;
    constexpr uint32_t kv_heads = 2;
    constexpr uint32_t query_heads = 4;
    constexpr uint32_t sealed_tokens = 10;
    constexpr uint32_t active_tokens = 3;

    std::vector<SealedKvPage> pages;
    std::vector<ActiveKvTail> tails;
    std::vector<std::vector<float>> dense_k(kv_heads);
    std::vector<std::vector<float>> dense_v(kv_heads);
    std::vector<std::vector<float>> active_k(kv_heads);
    std::vector<std::vector<float>> active_v(kv_heads);

    for (uint32_t h = 0; h < kv_heads; ++h) {
        const auto source_k = make_values(sealed_tokens, dim, 0.2f + h);
        const auto source_v = make_values(sealed_tokens, dim, 0.9f + h);
        pages.push_back(seal_int4_page(source_k.data(), source_v.data(), 0, 4, 0, h, dim));
        pages.push_back(seal_int4_page(source_k.data() + 4 * dim, source_v.data() + 4 * dim, 4, 6, 0, h, dim));
        std::vector<float> p0k, p0v, p1k, p1v;
        dequantize_page(pages[2 * h], &p0k, &p0v);
        dequantize_page(pages[2 * h + 1], &p1k, &p1v);
        dense_k[h] = p0k;
        dense_k[h].insert(dense_k[h].end(), p1k.begin(), p1k.end());
        dense_v[h] = p0v;
        dense_v[h].insert(dense_v[h].end(), p1v.begin(), p1v.end());
        active_k[h] = make_values(active_tokens, dim, 1.3f + h);
        active_v[h] = make_values(active_tokens, dim, 1.7f + h);
        dense_k[h].insert(dense_k[h].end(), active_k[h].begin(), active_k[h].end());
        dense_v[h].insert(dense_v[h].end(), active_v[h].begin(), active_v[h].end());
        tails.push_back(ActiveKvTail{sealed_tokens, active_tokens, 0, h, dim,
                                     active_k[h].data(), active_v[h].data()});
    }

    const auto queries = make_values(3, query_heads * dim, 2.1f);
    AttentionShape shape{3, query_heads, kv_heads, dim, 10, 7, 1.0f / std::sqrt(static_cast<float>(dim))};
    std::vector<float> actual;
    AttentionDiagnostics diagnostics{};
    paged_attention(pages, tails, queries.data(), shape, &actual, &diagnostics);

    std::vector<uint32_t> tokens;
    for (uint32_t t = 0; t < sealed_tokens + active_tokens; ++t) tokens.push_back(t);
    const auto expected = dense_attention(dense_k, dense_v, tokens, queries, shape);
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) assert(near(actual[i], expected[i]));
    assert(diagnostics.sealed_values > 0);
    assert(diagnostics.active_values > 0);
    assert(diagnostics.attended_values == diagnostics.sealed_values + diagnostics.active_values);
    assert(diagnostics.sealed_payload_bytes > 0);

    std::reverse(pages.begin(), pages.end());
    std::vector<float> reordered;
    paged_attention(pages, tails, queries.data(), shape, &reordered);
    for (std::size_t i = 0; i < actual.size(); ++i) assert(near(actual[i], reordered[i]));
}

void test_rejections() {
    constexpr uint32_t dim = 4;
    const auto k = make_values(4, dim, 0.0f);
    const auto v = make_values(4, dim, 0.5f);
    auto a = seal_int4_page(k.data(), v.data(), 0, 4, 0, 0, dim);
    auto b = seal_int4_page(k.data(), v.data(), 3, 4, 0, 0, dim);
    const auto q = make_values(1, dim, 0.2f);
    AttentionShape shape{1, 1, 1, dim, 3, 0, 0.5f};
    std::vector<float> output;
    expect_throw([&] { paged_attention({a, b}, {}, q.data(), shape, &output); });

    shape.query_head_count = 3;
    shape.kv_head_count = 2;
    expect_throw([&] { paged_attention({a}, {}, q.data(), shape, &output); });

    auto bad = k;
    bad[0] = std::numeric_limits<float>::infinity();
    expect_throw([&] { (void)seal_int4_page(bad.data(), v.data(), 0, 4, 0, 0, dim); });
}

} // namespace

int main() {
    test_sealing_orientation_and_error();
    test_paged_matches_dense_with_gqa_and_tail();
    test_rejections();
    std::cout << "M9 page sealing and paged attention reference tests passed\n";
    return 0;
}
