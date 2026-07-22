// P10 latency: is compressed-KV attention equal-or-better latency than dense? The P1 thesis predicts a crossover
// — compressed does more compute (decode int4 inline) but reads ~4x less memory, so it should lose at low
// occupancy / short context (decode-bound) and win at high occupancy / long context (bandwidth-bound). Races the
// verified cf_kv_paged_attention_async (over real sealed int4 pages) against a fair dense-f16 attention kernel
// (same online-softmax, raw reads) at several (context, query_count) points. Reports both, honestly.
#include "chromofold/compressed_kv_cache.hpp"
#include "chromofold/kv_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace chromofold;
static void ck(cudaError_t e, const char* m) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); std::exit(2); } }

#ifndef CF_KV_MAX_HEAD_DIM
#define CF_KV_MAX_HEAD_DIM 256
#endif

// FAIR dense baseline: warp-cooperative f16 K/V, same structure as the compressed warp kernel (one warp per
// query·head, head_dim strided across lanes, butterfly-shuffle QK reduce, register-resident state). The ONLY
// difference vs compressed is the read: raw f16 here vs int4-decode there. DPL = head_dim/32.
template <int DPL>
__global__ void dense_attention_warp(const __half* K, const __half* V, const float* Q, float* out,
                                     int N, int query_count, int qh, int kvh, int gqa, float scale, int qtb) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= query_count * qh) return;
    const int query_row = warp / qh, query_head = warp % qh, kv_head = query_head / gqa;
    const int query_token = qtb + query_row, hd = DPL * 32;
    float q[DPL], w[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) { q[i] = Q[(long)(query_row * qh + query_head) * hd + lane + 32 * i]; w[i] = 0.f; }
    float maxi = -1e30f, sum = 0.f; int init = 0;
    for (int t = 0; t <= query_token && t < N; ++t) {
        const __half* kt = K + ((long) t * kvh + kv_head) * hd;
        float partial = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) partial += q[i] * __half2float(kt[lane + 32 * i]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, o);
        const float score = partial * scale;
        const __half* vt = V + ((long) t * kvh + kv_head) * hd;
        const float nm = init ? fmaxf(maxi, score) : score;
        const float a = init ? __expf(maxi - nm) : 0.f, b = init ? __expf(score - nm) : 1.f;
        sum = init ? sum * a + b : 1.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) { const float vv = __half2float(vt[lane + 32 * i]); w[i] = init ? w[i] * a + vv * b : vv; }
        maxi = nm; init = 1;
    }
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[(long)(query_row * qh + query_head) * hd + lane + 32 * i] = w[i] / (sum > 0.f ? sum : 1.f);
}

static float time_ms(void (*launch)(void*), void* ctx, int iters) {
    for (int i = 0; i < 10; ++i) launch(ctx);   // warmup
    cudaEvent_t s, e; ck(cudaEventCreate(&s), "es"); ck(cudaEventCreate(&e), "ee");
    ck(cudaEventRecord(s), "rs");
    for (int i = 0; i < iters; ++i) launch(ctx);
    ck(cudaEventRecord(e), "re"); ck(cudaEventSynchronize(e), "sync");
    float ms = 0; ck(cudaEventElapsedTime(&ms, s, e), "el");
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

struct DenseCtx { const __half *K, *V; const float* Q; float* out; int N, Q_, qh, kvh, gqa, hd, qtb; float scale; };
static void launch_dense(void* p) {
    DenseCtx* c = (DenseCtx*) p;
    const int work = c->Q_ * c->qh, warps_per_block = 4, threads = warps_per_block * 32;
    const int blocks = (work + warps_per_block - 1) / warps_per_block;
    // bench fixes head_dim=64 -> DPL=2 (warp-cooperative, matched to the compressed kernel)
    dense_attention_warp<2><<<blocks, threads>>>(c->K, c->V, c->Q, c->out, c->N, c->Q_, c->qh, c->kvh, c->gqa, c->scale, c->qtb);
}
static void launch_paged(void* p) { cf_kv_paged_attention_async((cf_kv_paged_attention_desc*) p, nullptr); }

static void bench(std::uint32_t N, std::uint32_t Qn) {
    const std::uint32_t hd = 64, kvh = 2, qh = 14, gqa = 7, page = 128, iters = 50;
    const float scale = 1.f / std::sqrt((float) hd);
    std::vector<float> K((std::size_t) N * hd), V((std::size_t) N * hd), Q((std::size_t) Qn * qh * hd);
    for (std::size_t i = 0; i < K.size(); ++i) { K[i] = std::sin(0.01f * i); V[i] = std::cos(0.013f * i); }
    for (std::size_t i = 0; i < Q.size(); ++i) Q[i] = std::sin(0.017f * i);

    // compressed: real sealed int4 pages via the cache
    KvCacheConfig cfg; cfg.layer_count = 1; cfg.kv_head_count = kvh; cfg.query_head_count = qh;
    cfg.head_dim = hd; cfg.page_size = page; cfg.gqa_group_size = gqa; cfg.codec = KvCodecMode::fixed_int4_huffman;
    CompressedKvCache cache(cfg);
    for (std::uint32_t h = 0; h < kvh; ++h) cache.append(0, h, 0, K.data(), V.data(), N, nullptr);
    KvAttentionView view = cache.attention_view(0, nullptr);
    ck(cudaDeviceSynchronize(), "view");

    float *dQ = nullptr, *dOutC = nullptr, *dOutD = nullptr;
    ck(cudaMalloc(&dQ, Q.size() * sizeof(float)), "q");
    ck(cudaMalloc(&dOutC, (std::size_t) Qn * qh * hd * sizeof(float)), "oc");
    ck(cudaMalloc(&dOutD, (std::size_t) Qn * qh * hd * sizeof(float)), "od");
    ck(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice), "cq");

    cf_kv_paged_attention_desc desc{};
    desc.struct_size = sizeof(desc); desc.abi_version = CF_KV_CUDA_ABI_VERSION;
    desc.pages = view.device_descriptors; desc.page_count = view.page_count;
    desc.kv_head_count = kvh; desc.query_head_count = qh; desc.gqa_group_size = gqa;
    desc.active_k = view.active_k; desc.active_v = view.active_v;
    desc.active_token_begin = view.active_token_begin; desc.active_token_count = view.active_token_count;
    desc.queries = dQ; desc.output = dOutC; desc.query_token_begin = N - Qn; desc.query_count = Qn;
    desc.head_dim = hd; desc.causal_window = 0; desc.softmax_scale = scale;

    // dense f16 K/V [N, kvh, hd]
    std::vector<__half> Kh((std::size_t) N * kvh * hd), Vh((std::size_t) N * kvh * hd);
    for (std::uint32_t t = 0; t < N; ++t)
        for (std::uint32_t h = 0; h < kvh; ++h)
            for (std::uint32_t d = 0; d < hd; ++d) {
                Kh[((std::size_t) t * kvh + h) * hd + d] = __float2half(K[(std::size_t) t * hd + d]);
                Vh[((std::size_t) t * kvh + h) * hd + d] = __float2half(V[(std::size_t) t * hd + d]);
            }
    __half *dK = nullptr, *dV = nullptr;
    ck(cudaMalloc(&dK, Kh.size() * sizeof(__half)), "dk"); ck(cudaMalloc(&dV, Vh.size() * sizeof(__half)), "dv");
    ck(cudaMemcpy(dK, Kh.data(), Kh.size() * sizeof(__half), cudaMemcpyHostToDevice), "ck");
    ck(cudaMemcpy(dV, Vh.data(), Vh.size() * sizeof(__half), cudaMemcpyHostToDevice), "cv");
    DenseCtx dc{dK, dV, dQ, dOutD, (int) N, (int) Qn, (int) qh, (int) kvh, (int) gqa, (int) hd, (int) (N - Qn), scale};

    const double comp = time_ms(launch_paged, &desc, iters) * 1000.0 / iters;
    const double dens = time_ms(launch_dense, &dc, iters) * 1000.0 / iters;
    std::printf("N=%-6u Qn=%-4u | compressed %8.2f us | dense-f16 %8.2f us | ratio %.2fx %s\n",
                N, Qn, comp, dens, dens / comp, comp <= dens ? "(compressed faster)" : "(dense faster)");
    std::fflush(stdout);
    cudaFree(dQ); cudaFree(dOutC); cudaFree(dOutD); cudaFree(dK); cudaFree(dV);
}

int main() {
    std::printf("Compressed int4 paged attention vs dense f16 (Qwen2.5-0.5B dims, RTX 2080 Ti)\n");
    std::printf("decode (Qn=1) and prefill/batch (Qn=256), short and long context:\n");
    bench(512, 1); bench(4096, 1);
    bench(512, 256); bench(4096, 256);
    return 0;
}
