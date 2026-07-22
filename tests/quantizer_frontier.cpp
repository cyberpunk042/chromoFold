// Quantizer frontier (host-only): confirm that the KV-attention error against a full-precision reference is the
// QUANTIZER's cost and shrinks with bit width — the reducibility claim behind the live M9 finding (int4 sealed
// pages gave mean 0.034 vs llama dense, while the raw/addressing path gave 1.2e-4). Replicates the engine's exact
// scheme (src/runtime/kv_gpu_fixture.cpp): symmetric per-channel K scale, per-token V scale, scale = maxabs/qmax.
// No CUDA — this isolates the quantizer numerics, not the kernel. Build: g++ -O2 -std=c++17.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
constexpr uint32_t HD = 64, T = 32, NQ = 256;  // one int4 page worth of tokens, many queries

float dequant(float x, float scale, int qmax) {
    const int q = std::max(-qmax - 1, std::min(qmax, (int) std::lrint(x / scale)));
    return (float) q * scale;
}

// Attention output for one query over (dequantized) K/V; scale = 1/sqrt(HD), full causal (query attends all T).
void attend(const float* Q, const std::vector<float>& K, const std::vector<float>& V, float* out) {
    const float s = 1.0f / std::sqrt((float) HD);
    std::vector<float> sc(T);
    float mx = -1e30f;
    for (uint32_t t = 0; t < T; ++t) {
        float d = 0.f;
        for (uint32_t i = 0; i < HD; ++i) d += Q[i] * K[t * HD + i];
        sc[t] = d * s; mx = std::max(mx, sc[t]);
    }
    float den = 0.f;
    for (uint32_t t = 0; t < T; ++t) { sc[t] = std::exp(sc[t] - mx); den += sc[t]; }
    for (uint32_t i = 0; i < HD; ++i) {
        float a = 0.f;
        for (uint32_t t = 0; t < T; ++t) a += (sc[t] / den) * V[t * HD + i];
        out[i] = a;
    }
}

// Quantize K (per-channel) and V (per-token) at the given qmax, returning dequantized buffers.
void quantize_kv(const std::vector<float>& K, const std::vector<float>& V, int qmax,
                 std::vector<float>& Kq, std::vector<float>& Vq) {
    Kq.resize(K.size()); Vq.resize(V.size());
    for (uint32_t d = 0; d < HD; ++d) {
        float m = 0.f;
        for (uint32_t t = 0; t < T; ++t) m = std::max(m, std::fabs(K[t * HD + d]));
        const float sc = m == 0.f ? 1.f : m / (float) qmax;
        for (uint32_t t = 0; t < T; ++t) Kq[t * HD + d] = dequant(K[t * HD + d], sc, qmax);
    }
    for (uint32_t t = 0; t < T; ++t) {
        float m = 0.f;
        for (uint32_t d = 0; d < HD; ++d) m = std::max(m, std::fabs(V[t * HD + d]));
        const float sc = m == 0.f ? 1.f : m / (float) qmax;
        for (uint32_t d = 0; d < HD; ++d) Vq[t * HD + d] = dequant(V[t * HD + d], sc, qmax);
    }
}
}  // namespace

int main() {
    std::mt19937 rng(20260722);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K(T * HD), V(T * HD);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    struct Acc { double sum = 0, max = 0; uint64_t n = 0; };
    Acc a4, a8;
    std::vector<float> K4, V4, K8, V8;
    quantize_kv(K, V, 7, K4, V4);
    quantize_kv(K, V, 127, K8, V8);

    for (uint32_t q = 0; q < NQ; ++q) {
        std::vector<float> Q(HD);
        for (auto& x : Q) x = nd(rng);
        float ref[HD], o4[HD], o8[HD];
        attend(Q.data(), K, V, ref);
        attend(Q.data(), K4, V4, o4);
        attend(Q.data(), K8, V8, o8);
        for (uint32_t i = 0; i < HD; ++i) {
            const double e4 = std::fabs(o4[i] - ref[i]), e8 = std::fabs(o8[i] - ref[i]);
            a4.sum += e4; a4.max = std::max(a4.max, e4); a4.n++;
            a8.sum += e8; a8.max = std::max(a8.max, e8); a8.n++;
        }
    }
    const double m4 = a4.sum / a4.n, m8 = a8.sum / a8.n;
    std::printf("{\"tokens\":%u,\"head_dim\":%u,\"queries\":%u,"
                "\"int4\":{\"mean\":%.6f,\"max\":%.6f},\"int8\":{\"mean\":%.6f,\"max\":%.6f},"
                "\"int4_over_int8_mean\":%.1f}\n",
                T, HD, NQ, m4, a4.max, m8, a8.max, m8 > 0 ? m4 / m8 : 0.0);
    // The point: int8 error must be markedly smaller than int4 — the quantizer's cost is reducible.
    return (m8 < m4 * 0.25) ? 0 : 1;
}
