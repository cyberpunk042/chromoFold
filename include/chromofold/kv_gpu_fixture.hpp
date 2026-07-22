#ifndef CHROMOFOLD_KV_GPU_FIXTURE_HPP
#define CHROMOFOLD_KV_GPU_FIXTURE_HPP

#include <cstdint>
#include <vector>

namespace chromofold::gpu_fixture {

struct EncodedStream {
    std::vector<std::uint32_t> words;
    std::vector<std::int32_t> block_offsets;
    std::vector<std::int32_t> lut;
    std::uint32_t max_code_length = 4;
    std::uint32_t block_size = 64;
    std::uint32_t fixed_width = 0;  // >0: every code is exactly this many bits (fast direct decode); 0: variable
};

struct EncodedKvPage {
    std::uint32_t token_begin = 0;
    std::uint32_t token_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t kv_head = 0;
    std::uint32_t zero_point = 8;
    EncodedStream k;
    EncodedStream v;
    std::vector<float> k_scales;
    std::vector<float> v_scales;
    std::vector<float> dequantized_k;
    std::vector<float> dequantized_v;

    std::uint64_t resident_bytes() const noexcept;
};

EncodedKvPage encode_int4_page(const std::vector<float>& keys,
                               const std::vector<float>& values,
                               std::uint32_t token_begin,
                               std::uint32_t token_count,
                               std::uint32_t head_dim,
                               std::uint32_t kv_head,
                               std::uint32_t block_size = 64);

// Same scheme at 8-bit (qmax 127, zero_point 128) — ~18x less attention error than int4 (gpu-quantizer-frontier),
// at 2x the bits. Kernel/descriptor are bit-width-agnostic; only the codebook width differs.
EncodedKvPage encode_int8_page(const std::vector<float>& keys,
                               const std::vector<float>& values,
                               std::uint32_t token_begin,
                               std::uint32_t token_count,
                               std::uint32_t head_dim,
                               std::uint32_t kv_head,
                               std::uint32_t block_size = 64);

std::vector<std::uint8_t> decode_symbols(const EncodedStream& stream,
                                         std::uint64_t symbol_count);

void validate_page(const EncodedKvPage& page);

}  // namespace chromofold::gpu_fixture

#endif
