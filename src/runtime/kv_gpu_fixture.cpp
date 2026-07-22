#include "chromofold/kv_gpu_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace chromofold::gpu_fixture {
namespace {

// Symmetric fixed-width quantization. `bits` in {4,8}; qmax = 2^(bits-1)-1, zero_point = 2^(bits-1).
// int4: clamp [-8,7], symbol = q+8. int8: clamp [-128,127], symbol = q+128.
std::uint8_t quantize(float value, float scale, int qmax, int zero_point) {
    if (!std::isfinite(value) || !(scale > 0.0f)) throw std::invalid_argument("invalid quantization input");
    const int signed_q = std::clamp(static_cast<int>(std::lrint(value / scale)), -(qmax + 1), qmax);
    return static_cast<std::uint8_t>(signed_q + zero_point);
}

// Pack fixed-width symbols MSB-first at `bits` bits each (bits divides 32 so no symbol crosses a word). The LUT is
// the degenerate canonical-Huffman table lut[s] = s | (bits<<8); the device decoder reads `bits` bits and hits it.
EncodedStream encode_symbols(const std::vector<std::uint8_t>& symbols, std::uint32_t block_size, std::uint32_t bits) {
    if (symbols.empty() || block_size == 0 || (bits != 4u && bits != 8u)) throw std::invalid_argument("empty stream or block size");
    EncodedStream out;
    out.block_size = block_size;
    out.max_code_length = bits;
    out.fixed_width = bits;  // this fixture emits fixed-width codes -> enables the kernel's direct-decode fast path
    const std::uint32_t levels = 1u << bits;
    const std::uint32_t mask = levels - 1u;
    out.lut.resize(levels);
    for (std::uint32_t symbol = 0; symbol < levels; ++symbol) out.lut[symbol] = static_cast<std::int32_t>(symbol | (bits << 8));

    const std::uint64_t bit_count = static_cast<std::uint64_t>(symbols.size()) * bits;
    out.words.assign(static_cast<std::size_t>((bit_count + 31u) / 32u), 0u);
    out.block_offsets.reserve((symbols.size() + block_size - 1u) / block_size);

    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if (i % block_size == 0) out.block_offsets.push_back(static_cast<std::int32_t>(static_cast<std::uint64_t>(i) * bits));
        const std::uint64_t bit = static_cast<std::uint64_t>(i) * bits;
        const std::uint32_t word = static_cast<std::uint32_t>(bit >> 5u);
        const std::uint32_t shift = (32u - bits) - static_cast<std::uint32_t>(bit & 31u);
        out.words[word] |= static_cast<std::uint32_t>(symbols[i] & mask) << shift;
    }
    return out;
}

// Shared page encoder for both bit widths. K uses a per-channel scale, V a per-token scale, both maxabs/qmax.
EncodedKvPage encode_page(const std::vector<float>& keys, const std::vector<float>& values,
                          std::uint32_t token_begin, std::uint32_t token_count, std::uint32_t head_dim,
                          std::uint32_t kv_head, std::uint32_t block_size, std::uint32_t bits) {
    const std::uint64_t count = static_cast<std::uint64_t>(token_count) * head_dim;
    if (token_count == 0 || head_dim == 0 || keys.size() != count || values.size() != count || block_size == 0 ||
        (bits != 4u && bits != 8u)) {
        throw std::invalid_argument("invalid page shape");
    }
    const int qmax = (1 << (bits - 1)) - 1;
    const int zero_point = 1 << (bits - 1);

    EncodedKvPage page;
    page.token_begin = token_begin;
    page.token_count = token_count;
    page.head_dim = head_dim;
    page.kv_head = kv_head;
    page.zero_point = static_cast<std::uint32_t>(zero_point);
    page.k_scales.resize(head_dim);
    page.v_scales.resize(token_count);
    page.dequantized_k.resize(count);
    page.dequantized_v.resize(count);
    std::vector<std::uint8_t> k_symbols(count);
    std::vector<std::uint8_t> v_symbols(count);

    for (std::uint32_t d = 0; d < head_dim; ++d) {
        float maximum = 0.0f;
        for (std::uint32_t t = 0; t < token_count; ++t) maximum = std::max(maximum, std::abs(keys[static_cast<std::size_t>(t) * head_dim + d]));
        page.k_scales[d] = maximum == 0.0f ? 1.0f : maximum / static_cast<float>(qmax);
    }
    for (std::uint32_t t = 0; t < token_count; ++t) {
        float maximum = 0.0f;
        for (std::uint32_t d = 0; d < head_dim; ++d) maximum = std::max(maximum, std::abs(values[static_cast<std::size_t>(t) * head_dim + d]));
        page.v_scales[t] = maximum == 0.0f ? 1.0f : maximum / static_cast<float>(qmax);
    }

    for (std::uint32_t t = 0; t < token_count; ++t) {
        for (std::uint32_t d = 0; d < head_dim; ++d) {
            const std::size_t index = static_cast<std::size_t>(t) * head_dim + d;
            k_symbols[index] = quantize(keys[index], page.k_scales[d], qmax, zero_point);
            v_symbols[index] = quantize(values[index], page.v_scales[t], qmax, zero_point);
            page.dequantized_k[index] = (static_cast<int>(k_symbols[index]) - zero_point) * page.k_scales[d];
            page.dequantized_v[index] = (static_cast<int>(v_symbols[index]) - zero_point) * page.v_scales[t];
        }
    }

    page.k = encode_symbols(k_symbols, block_size, bits);
    page.v = encode_symbols(v_symbols, block_size, bits);
    validate_page(page);
    return page;
}

}  // namespace

std::uint64_t EncodedKvPage::resident_bytes() const noexcept {
    return (k.words.size() + v.words.size()) * sizeof(std::uint32_t) +
           (k.block_offsets.size() + v.block_offsets.size() + k.lut.size() + v.lut.size()) * sizeof(std::int32_t) +
           (k_scales.size() + v_scales.size()) * sizeof(float);
}

std::vector<std::uint8_t> decode_symbols(const EncodedStream& stream, std::uint64_t symbol_count) {
    const std::uint32_t bits = stream.max_code_length;
    if ((bits != 4u && bits != 8u) || stream.lut.size() != (1u << bits)) throw std::invalid_argument("unsupported fixture codebook");
    const std::uint32_t mask = (1u << bits) - 1u;
    std::vector<std::uint8_t> result(static_cast<std::size_t>(symbol_count));
    for (std::uint64_t i = 0; i < symbol_count; ++i) {
        const std::uint64_t bit = i * bits;
        const std::uint32_t word = static_cast<std::uint32_t>(bit >> 5u);
        const std::uint32_t shift = (32u - bits) - static_cast<std::uint32_t>(bit & 31u);
        if (word >= stream.words.size()) throw std::invalid_argument("truncated fixture stream");
        result[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((stream.words[word] >> shift) & mask);
    }
    return result;
}

EncodedKvPage encode_int4_page(const std::vector<float>& keys, const std::vector<float>& values,
                               std::uint32_t token_begin, std::uint32_t token_count, std::uint32_t head_dim,
                               std::uint32_t kv_head, std::uint32_t block_size) {
    return encode_page(keys, values, token_begin, token_count, head_dim, kv_head, block_size, 4u);
}

EncodedKvPage encode_int8_page(const std::vector<float>& keys, const std::vector<float>& values,
                               std::uint32_t token_begin, std::uint32_t token_count, std::uint32_t head_dim,
                               std::uint32_t kv_head, std::uint32_t block_size) {
    return encode_page(keys, values, token_begin, token_count, head_dim, kv_head, block_size, 8u);
}

void validate_page(const EncodedKvPage& page) {
    const std::uint64_t values = static_cast<std::uint64_t>(page.token_count) * page.head_dim;
    const std::uint32_t bits = page.k.max_code_length;
    if (page.token_count == 0 || page.head_dim == 0 ||
        page.k_scales.size() != page.head_dim || page.v_scales.size() != page.token_count ||
        page.k.block_size == 0 || page.v.block_size != page.k.block_size ||
        (bits != 4u && bits != 8u) || page.v.max_code_length != bits ||
        page.k.lut.size() != (1u << bits) || page.v.lut.size() != (1u << bits)) {
        throw std::invalid_argument("invalid encoded KV page");
    }
    if (decode_symbols(page.k, values).size() != values || decode_symbols(page.v, values).size() != values) {
        throw std::invalid_argument("fixture round trip failed");
    }
}

}  // namespace chromofold::gpu_fixture
