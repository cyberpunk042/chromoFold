// Capacity proof (P10 flavor): how much KV actually fits at equal VRAM. Builds a real CompressedKvCache, appends
// a long sequence so pages seal, and measures BOTH the logical compressed size (stats) AND the actual device VRAM
// (cudaMemGetInfo) — vs a dense f16 KV cache for the same tokens. Honest about allocation overhead: the current
// per-page cudaMalloc strategy inflates real VRAM above the logical bytes, which the report surfaces explicitly.
// Engine-level (not through llama's allocator, which this system doesn't let us replace). Qwen2.5-0.5B dims.
#include "chromofold/compressed_kv_cache.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace chromofold;

static void ck(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}
static double MB(std::uint64_t b) { return (double) b / (1024.0 * 1024.0); }

static void measure(KvCodecMode codec, const char* name, std::uint32_t T) {
    const std::uint32_t layers = 24, kvh = 2, hd = 64, page_size = 128;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K((std::size_t) T * hd), V((std::size_t) T * hd);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    ck(cudaFree(nullptr), "ctx init");
    std::size_t free_before = 0, total = 0;
    ck(cudaMemGetInfo(&free_before, &total), "meminfo before");

    KvCacheConfig cfg;
    cfg.layer_count = layers; cfg.kv_head_count = kvh; cfg.query_head_count = kvh;
    cfg.head_dim = hd; cfg.page_size = page_size; cfg.gqa_group_size = 1; cfg.codec = codec;
    CompressedKvCache cache(cfg);
    for (std::uint32_t l = 0; l < layers; ++l)
        for (std::uint32_t h = 0; h < kvh; ++h)
            cache.append(l, h, 0, K.data(), V.data(), T, nullptr);
    ck(cudaDeviceSynchronize(), "sync");

    std::size_t free_after = 0;
    ck(cudaMemGetInfo(&free_after, &total), "meminfo after");
    const std::uint64_t actual_vram = (std::uint64_t) (free_before - free_after);

    KvCacheStats st = cache.stats();
    const std::uint64_t logical = st.compressed_bytes + st.descriptor_bytes + st.dense_active_bytes;
    // Dense f16 baseline: T tokens x layers x kv_heads x head_dim x (K+V) x 2 bytes.
    const std::uint64_t dense_f16 = (std::uint64_t) T * layers * kvh * hd * 2ull * 2ull;

    std::printf("[%s] tokens=%u  sealed_pages=%llu\n", name, T, (unsigned long long) st.sealed_pages);
    std::printf("  dense f16            : %8.2f MB\n", MB(dense_f16));
    std::printf("  compressed (logical) : %8.2f MB   -> %.2fx smaller than dense f16\n",
                MB(logical), (double) dense_f16 / (double) logical);
    std::printf("  compressed (real VRAM): %8.2f MB   -> %.2fx smaller  (alloc overhead %.2fx over logical)\n",
                MB(actual_vram), (double) dense_f16 / (double) actual_vram, (double) actual_vram / (double) logical);
    std::printf("  at equal VRAM, context fits: %.2fx (logical) / %.2fx (real) more tokens than dense f16\n\n",
                (double) dense_f16 / (double) logical, (double) dense_f16 / (double) actual_vram);
}

int main() {
    const std::uint32_t T = 4096;
    std::printf("Capacity at equal VRAM (Qwen2.5-0.5B dims: 24 layers, 2 kv-heads, head_dim 64, page 128)\n\n");
    measure(KvCodecMode::fixed_int4_huffman, "int4", T);
    measure(KvCodecMode::fixed_int8_huffman, "int8", T);
    return 0;
}
