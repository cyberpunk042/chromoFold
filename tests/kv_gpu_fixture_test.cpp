#include "chromofold/kv_gpu_fixture.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t tokens = 9;
    constexpr std::uint32_t dim = 32;
    std::vector<float> keys(tokens * dim);
    std::vector<float> values(tokens * dim);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t d = 0; d < dim; ++d) {
            const std::size_t i = static_cast<std::size_t>(t) * dim + d;
            keys[i] = std::sin(static_cast<float>((t + 1) * (d + 3)) * 0.031f) * 3.0f;
            values[i] = std::cos(static_cast<float>((t + 2) * (d + 1)) * 0.023f) * 2.0f;
        }
    }

    const auto page = chromofold::gpu_fixture::encode_int4_page(keys, values, 1024, tokens, dim, 3, 64);
    chromofold::gpu_fixture::validate_page(page);
    assert(page.k.block_offsets.size() == (tokens * dim + 63) / 64);
    assert(page.v.block_offsets.size() == page.k.block_offsets.size());
    assert(page.k.lut.size() == 16);
    assert(page.v.lut.size() == 16);
    assert(page.resident_bytes() > 0);

    const auto k_symbols = chromofold::gpu_fixture::decode_symbols(page.k, tokens * dim);
    const auto v_symbols = chromofold::gpu_fixture::decode_symbols(page.v, tokens * dim);
    for (std::size_t i = 0; i < k_symbols.size(); ++i) {
        assert(k_symbols[i] <= 15);
        assert(v_symbols[i] <= 15);
        assert(std::isfinite(page.dequantized_k[i]));
        assert(std::isfinite(page.dequantized_v[i]));
    }

    const auto repeated = chromofold::gpu_fixture::encode_int4_page(keys, values, 1024, tokens, dim, 3, 64);
    assert(page.k.words == repeated.k.words);
    assert(page.v.words == repeated.v.words);
    assert(page.k.block_offsets == repeated.k.block_offsets);
    assert(page.v.block_offsets == repeated.v.block_offsets);
    assert(page.k.lut == repeated.k.lut);
    assert(page.v.lut == repeated.v.lut);

    std::cout << "M9 GPU fixture tests passed; resident_bytes=" << page.resident_bytes() << '\n';
    return 0;
}
