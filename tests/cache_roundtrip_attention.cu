// Round-trip proof for the compressed KV cache: append real K/V into CompressedKvCache, attend through
// attention_view() (sealed pages + the newly-exposed active fp32 tail) with the verified paged-attention
// kernel, and compare to a dense CPU reference over the SAME values the cache stored (int4-dequantized for
// sealed tokens, raw fp32 for the active tail). This closes the append-step gap (the appended data really
// round-trips) and de-risks the layer-2 replace step, which stands on exactly this path.
#include "chromofold/compressed_kv_cache.hpp"
#include "chromofold/kv_gpu_fixture.hpp"
#include "chromofold/kv_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace chromofold;

static void ck(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

int main() {
    const std::uint32_t hd = 64, page_size = 32, N = 70;  // 70 = 2 sealed pages (64) + 6 active
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K(N * hd), V(N * hd), Q(hd);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);
    for (auto& x : Q) x = nd(rng);

    KvCacheConfig cfg;
    cfg.layer_count = 1; cfg.kv_head_count = 1; cfg.query_head_count = 1;
    cfg.head_dim = hd; cfg.page_size = page_size; cfg.gqa_group_size = 1;
    CompressedKvCache cache(cfg);
    for (std::uint32_t t = 0; t < N; ++t)  // token-by-token, as in decode
        cache.append(0, 0, t, &K[t * hd], &V[t * hd], 1, nullptr);
    cudaDeviceSynchronize();

    KvAttentionView view = cache.attention_view(0, nullptr);
    cudaDeviceSynchronize();

    float *dQ = nullptr, *dOut = nullptr;
    ck(cudaMalloc(&dQ, hd * sizeof(float)), "malloc Q");
    ck(cudaMalloc(&dOut, hd * sizeof(float)), "malloc out");
    ck(cudaMemcpy(dQ, Q.data(), hd * sizeof(float), cudaMemcpyHostToDevice), "cp Q");

    cf_kv_paged_attention_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CF_KV_CUDA_ABI_VERSION;
    desc.pages = view.device_descriptors;
    desc.page_count = view.page_count;
    desc.kv_head_count = 1; desc.query_head_count = 1; desc.gqa_group_size = 1;
    desc.active_k = view.active_k; desc.active_v = view.active_v;
    desc.active_token_begin = view.active_token_begin; desc.active_token_count = view.active_token_count;
    desc.queries = dQ; desc.output = dOut;
    desc.query_token_begin = N - 1; desc.query_count = 1; desc.head_dim = hd;
    desc.causal_window = 0; desc.softmax_scale = 1.f / std::sqrt((float) hd);
    if (cf_kv_validate_paged_attention_desc(&desc) != CF_OK) { std::printf("desc invalid\n"); return 2; }
    if (cf_kv_paged_attention_async(&desc, nullptr) != CF_OK) { std::printf("launch rejected\n"); return 2; }
    cudaDeviceSynchronize();
    std::vector<float> actual(hd);
    ck(cudaMemcpy(actual.data(), dOut, hd * sizeof(float), cudaMemcpyDeviceToHost), "cp out");

    // Reference: dequantized K/V for the sealed pages (same encode as the cache) + raw K/V for the active tail.
    const std::uint32_t sealed = (N / page_size) * page_size;  // 64
    std::vector<float> refK, refV;
    for (std::uint32_t tb = 0; tb < sealed; tb += page_size) {
        std::vector<float> pk(K.begin() + tb * hd, K.begin() + (tb + page_size) * hd);
        std::vector<float> pv(V.begin() + tb * hd, V.begin() + (tb + page_size) * hd);
        auto enc = gpu_fixture::encode_int4_page(pk, pv, tb, page_size, hd, 0, 64);
        refK.insert(refK.end(), enc.dequantized_k.begin(), enc.dequantized_k.end());
        refV.insert(refV.end(), enc.dequantized_v.begin(), enc.dequantized_v.end());
    }
    for (std::uint32_t t = sealed; t < N; ++t) {
        refK.insert(refK.end(), K.begin() + t * hd, K.begin() + (t + 1) * hd);
        refV.insert(refV.end(), V.begin() + t * hd, V.begin() + (t + 1) * hd);
    }

    const float scale = 1.f / std::sqrt((float) hd);
    std::vector<float> score(N);
    float mx = -1e30f;
    for (std::uint32_t t = 0; t < N; ++t) {
        float s = 0.f;
        for (std::uint32_t d = 0; d < hd; ++d) s += Q[d] * refK[t * hd + d];
        s *= scale; score[t] = s; mx = std::max(mx, s);
    }
    float den = 0.f;
    for (std::uint32_t t = 0; t < N; ++t) { score[t] = std::exp(score[t] - mx); den += score[t]; }
    std::vector<float> expected(hd, 0.f);
    for (std::uint32_t t = 0; t < N; ++t) {
        const float w = score[t] / den;
        for (std::uint32_t d = 0; d < hd; ++d) expected[d] += w * refV[t * hd + d];
    }

    float maxabs = 0.f; double mse = 0.0;
    for (std::uint32_t d = 0; d < hd; ++d) {
        const float e = std::fabs(actual[d] - expected[d]);
        maxabs = std::max(maxabs, e); mse += (double) e * e;
    }
    mse /= hd;
    std::printf("{\"tokens\":%u,\"sealed\":%u,\"active\":%u,\"page_count\":%u,\"max_abs_error\":%g,\"mse\":%g}\n",
                N, sealed, view.active_token_count, view.page_count, maxabs, mse);
    cudaFree(dQ); cudaFree(dOut);
    return maxabs <= 2e-4f ? 0 : 1;
}
