// Quantizer frontier (host-only): map the KV-attention accuracy/size tradeoff to guide the engine's quantizer.
// Confirms the live M9 finding (int4 sealed pages: mean 0.034 vs dense; int8: 0.0024) and asks the deployment
// question it raised: can FINER-GROUPED int4 reach int8 accuracy near int4's size? Replicates the engine's
// symmetric scheme (per-channel K, per-token V) and sweeps the quantization group size. No CUDA — isolates the
// quantizer numerics. Build: g++ -O2 -std=c++17.  make -f m9-gpu.mk gpu-quantizer-frontier
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
constexpr uint32_t HD = 64, T = 32, NQ = 256;

float dequant(float x, float scale, int qmax) {
    const int q = std::max(-qmax - 1, std::min(qmax, (int) std::lrint(x / scale)));
    return (float) q * scale;
}

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

// K quantized per (channel, token-group of kg); V per (token, dim-group of vg). Engine default = (kg=T, vg=HD).
void quantize_kv(const std::vector<float>& K, const std::vector<float>& V, int qmax, uint32_t kg, uint32_t vg,
                 std::vector<float>& Kq, std::vector<float>& Vq) {
    Kq.resize(K.size()); Vq.resize(V.size());
    for (uint32_t d = 0; d < HD; ++d)
        for (uint32_t t0 = 0; t0 < T; t0 += kg) {
            const uint32_t t1 = std::min(T, t0 + kg);
            float m = 0.f;
            for (uint32_t t = t0; t < t1; ++t) m = std::max(m, std::fabs(K[t * HD + d]));
            const float sc = m == 0.f ? 1.f : m / (float) qmax;
            for (uint32_t t = t0; t < t1; ++t) Kq[t * HD + d] = dequant(K[t * HD + d], sc, qmax);
        }
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t d0 = 0; d0 < HD; d0 += vg) {
            const uint32_t d1 = std::min(HD, d0 + vg);
            float m = 0.f;
            for (uint32_t d = d0; d < d1; ++d) m = std::max(m, std::fabs(V[t * HD + d]));
            const float sc = m == 0.f ? 1.f : m / (float) qmax;
            for (uint32_t d = d0; d < d1; ++d) Vq[t * HD + d] = dequant(V[t * HD + d], sc, qmax);
        }
}

double bytes_per_token(uint32_t bits, uint32_t kg, uint32_t vg) {
    const double packed = 2.0 * T * HD * bits / 8.0;                       // K + V codes
    const double kscales = (double) HD * ((T + kg - 1) / kg);              // per (channel, token-group)
    const double vscales = (double) T * ((HD + vg - 1) / vg);             // per (token, dim-group)
    return (packed + (kscales + vscales) * 4.0) / T;
}
}  // namespace

int main() {
    std::mt19937 rng(20260722);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ur(0.f, 1.f);
    // Calibrated to the LIVE model: near-N(0,1) reproduces the observed int4 error (~0.034 vs dense). A light
    // channel-outlier component (why per-channel K is the right axis) keeps it representative without inflating.
    std::vector<float> K(T * HD), V(T * HD);
    std::vector<float> chan_gain(HD, 1.f);
    for (uint32_t d = 0; d < HD; ++d) if (ur(rng) < 0.04f) chan_gain[d] = 3.f;  // light outlier channels
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t d = 0; d < HD; ++d) {
            K[t * HD + d] = nd(rng) * chan_gain[d];
            V[t * HD + d] = nd(rng);
        }

    struct Cfg { uint32_t bits, kg, vg; const char* note; };
    const Cfg grid[] = {
        {4, T, HD, "int4 engine-default (per-channel K, per-token V)"},
        {4, 8, HD, "int4 K-group 8"},
        {4, 8, 16, "int4 K-group 8, V-group 16"},
        {4, 4, 8,  "int4 K-group 4, V-group 8"},
        {8, T, HD, "int8 engine-default (reference)"},
    };
    std::printf("dense f16 = %.0f bytes/token\n", 4.0 * HD);
    std::printf("%-46s %6s %10s %10s %8s\n", "config", "bits", "mean_err", "max_err", "B/token");
    for (const Cfg& c : grid) {
        std::vector<float> Kq, Vq;
        quantize_kv(K, V, (1 << (c.bits - 1)) - 1, c.kg, c.vg, Kq, Vq);
        double sum = 0, mx = 0; uint64_t n = 0;
        for (uint32_t q = 0; q < NQ; ++q) {
            std::vector<float> Q(HD);
            for (auto& x : Q) x = nd(rng);
            float ref[HD], o[HD];
            attend(Q.data(), K, V, ref);
            attend(Q.data(), Kq, Vq, o);
            for (uint32_t i = 0; i < HD; ++i) { const double e = std::fabs(o[i] - ref[i]); sum += e; mx = std::max(mx, e); n++; }
        }
        std::printf("%-46s %6u %10.5f %10.4f %8.1f\n", c.note, c.bits, sum / n, mx, bytes_per_token(c.bits, c.kg, c.vg));
    }
    return 0;
}
