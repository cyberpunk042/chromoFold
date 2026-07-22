// Prove the adapter-level replace primitive: cf_llama_kv_create -> cf_llama_kv_append -> cf_llama_kv_attention,
// with the live model's GQA shape (2 kv heads, 14 query heads, group 7, head_dim 64). Compares the served
// attention to a dense CPU reference over the values the cache stored (int4-dequant sealed + raw active tail).
// This is exactly the call the layer-2 replace callback will make on kqv_out; verifying it here (no server)
// isolates the remaining work to graph plumbing.
#include "chromofold_kv_adapter.h"
#include "chromofold/kv_gpu_fixture.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using chromofold::gpu_fixture::encode_int4_page;

static void ck(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static int run(uint32_t bits) {
    const uint32_t hd = 64, page_size = 32, N = 70, kvh = 2, qh = 14, gqa = 7;
    std::mt19937 rng(9001);
    std::normal_distribution<float> nd(0.f, 1.f);

    // per kv-head K/V [N, hd]
    std::vector<std::vector<float>> K(kvh, std::vector<float>(N * hd)), V(kvh, std::vector<float>(N * hd));
    for (uint32_t h = 0; h < kvh; ++h) { for (auto& x : K[h]) x = nd(rng); for (auto& x : V[h]) x = nd(rng); }
    std::vector<float> Q(qh * hd);
    for (auto& x : Q) x = nd(rng);

    cf_llama_kv_options o{};
    o.struct_size = sizeof(o); o.backend = CF_LLAMA_KV_BACKEND_CHROMOFOLD;
    o.layer_count = 1; o.kv_head_count = kvh; o.query_head_count = qh;
    o.head_dim = hd; o.page_size = page_size; o.gqa_group_size = gqa; o.kv_bits = bits;
    cf_llama_kv_adapter* a = cf_llama_kv_create(&o);
    if (a == nullptr) { std::printf("create failed\n"); return 2; }

    for (uint32_t h = 0; h < kvh; ++h)
        for (uint32_t t = 0; t < N; ++t)
            if (cf_llama_kv_append(a, 0, h, t, &K[h][t * hd], &V[h][t * hd], 1, nullptr) != 0) {
                std::printf("append failed: %s\n", cf_llama_kv_last_error(a)); return 2;
            }
    cudaDeviceSynchronize();

    float *dQ = nullptr, *dOut = nullptr;
    ck(cudaMalloc(&dQ, qh * hd * sizeof(float)), "malloc Q");
    ck(cudaMalloc(&dOut, qh * hd * sizeof(float)), "malloc out");
    ck(cudaMemcpy(dQ, Q.data(), qh * hd * sizeof(float), cudaMemcpyHostToDevice), "cp Q");

    const float scale = 1.f / std::sqrt((float) hd);
    if (cf_llama_kv_attention(a, 0, dQ, dOut, /*query_count*/ 1, qh, gqa,
                              /*query_token_begin*/ N - 1, scale, /*causal_window*/ 0, nullptr) != 0) {
        std::printf("attention failed: %s\n", cf_llama_kv_last_error(a)); return 2;
    }
    cudaDeviceSynchronize();
    std::vector<float> actual(qh * hd);
    ck(cudaMemcpy(actual.data(), dOut, qh * hd * sizeof(float), cudaMemcpyDeviceToHost), "cp out");

    // dense reference: per kv head, dequantized sealed + raw active; per query head h -> kv head h/gqa
    const uint32_t sealed = (N / page_size) * page_size;
    std::vector<std::vector<float>> refK(kvh), refV(kvh);
    for (uint32_t h = 0; h < kvh; ++h) {
        for (uint32_t tb = 0; tb < sealed; tb += page_size) {
            std::vector<float> pk(K[h].begin() + tb * hd, K[h].begin() + (tb + page_size) * hd);
            std::vector<float> pv(V[h].begin() + tb * hd, V[h].begin() + (tb + page_size) * hd);
            auto enc = (bits == 8) ? chromofold::gpu_fixture::encode_int8_page(pk, pv, tb, page_size, hd, h, 64)
                                   : encode_int4_page(pk, pv, tb, page_size, hd, h, 64);
            refK[h].insert(refK[h].end(), enc.dequantized_k.begin(), enc.dequantized_k.end());
            refV[h].insert(refV[h].end(), enc.dequantized_v.begin(), enc.dequantized_v.end());
        }
        for (uint32_t t = sealed; t < N; ++t) {
            refK[h].insert(refK[h].end(), K[h].begin() + t * hd, K[h].begin() + (t + 1) * hd);
            refV[h].insert(refV[h].end(), V[h].begin() + t * hd, V[h].begin() + (t + 1) * hd);
        }
    }

    float maxabs = 0.f; double mse = 0.0;
    for (uint32_t qhi = 0; qhi < qh; ++qhi) {
        const uint32_t kv = qhi / gqa;
        std::vector<float> sc(N); float mx = -1e30f;
        for (uint32_t t = 0; t < N; ++t) {
            float s = 0.f;
            for (uint32_t d = 0; d < hd; ++d) s += Q[qhi * hd + d] * refK[kv][t * hd + d];
            s *= scale; sc[t] = s; mx = std::max(mx, s);
        }
        float den = 0.f;
        for (uint32_t t = 0; t < N; ++t) { sc[t] = std::exp(sc[t] - mx); den += sc[t]; }
        for (uint32_t d = 0; d < hd; ++d) {
            float acc = 0.f;
            for (uint32_t t = 0; t < N; ++t) acc += (sc[t] / den) * refV[kv][t * hd + d];
            const float e = std::fabs(actual[qhi * hd + d] - acc);
            maxabs = std::max(maxabs, e); mse += (double) e * e;
        }
    }
    mse /= (qh * hd);
    std::printf("{\"bits\":%u,\"tokens\":%u,\"kv_heads\":%u,\"query_heads\":%u,\"gqa\":%u,\"max_abs_error\":%g,\"mse\":%g}\n",
                bits, N, kvh, qh, gqa, maxabs, mse);
    cf_llama_kv_destroy(a);
    cudaFree(dQ); cudaFree(dOut);
    return maxabs <= 2e-4f ? 0 : 1;
}

int main() {
    // Verify the paged-attention kernel is bit-exact to a dequantized reference at BOTH codec widths.
    return run(4) | run(8);
}
