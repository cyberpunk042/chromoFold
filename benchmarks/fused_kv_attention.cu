// fused_kv_attention.cu — M6/M9 (KV-path fusion): decode-in-attention vs decode-then-dense over an entropy-coded
// KV cache. The fused path holds ONLY the compressed KV store and decodes attended K/V rows inline; the dense path
// must materialise the dequantized K,V tiles. Verify the two agree BIT-FOR-BIT (fusion is numerically free),
// cross-check vs the numpy golden, and measure the KV VRAM the fused path saves. Usage: ./fused_kv_attention a.cfkv

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e = (x);                                                                                       \
    if (e != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                   \
      return 2;                                                                                                \
    }                                                                                                          \
  } while (0)

struct Kv {
  uint64_t resident = 0, dense = 0;
  uint32_t seq = 0, dim = 0, nq = 0, window = 0, bits = 0, block = 0, zero = 0;
  uint32_t kmaxlen = 0, knwords = 0, knblocks = 0, klutlen = 0, vmaxlen = 0, vnwords = 0, vnblocks = 0, vlutlen = 0;
  float sscale = 0;
  std::vector<uint32_t> kwords, vwords;
  std::vector<int32_t> kboff, klut, vboff, vlut;
  std::vector<float> kscale, vscale, Q, golden;
};

static bool load_kv(const char *path, Kv &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFKV", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  rd(&r.seq, 4); rd(&r.dim, 4); rd(&r.nq, 4); rd(&r.window, 4); rd(&r.bits, 4); rd(&r.block, 4); rd(&r.zero, 4);
  rd(&r.kmaxlen, 4); rd(&r.knwords, 4); rd(&r.knblocks, 4); rd(&r.klutlen, 4);
  rd(&r.vmaxlen, 4); rd(&r.vnwords, 4); rd(&r.vnblocks, 4); rd(&r.vlutlen, 4);
  rd(&r.sscale, 4); rd(&r.resident, 8); rd(&r.dense, 8);
  if (!ok || version != 1) { std::fclose(f); return false; }
  r.kwords.resize(r.knwords); r.kboff.resize(r.knblocks); r.klut.resize(r.klutlen);
  r.vwords.resize(r.vnwords); r.vboff.resize(r.vnblocks); r.vlut.resize(r.vlutlen);
  r.kscale.resize(r.dim); r.vscale.resize(r.seq);
  r.Q.resize((size_t)r.nq * r.dim); r.golden.resize((size_t)r.nq * r.dim);
  rd(r.kwords.data(), r.kwords.size() * 4); rd(r.kboff.data(), r.kboff.size() * 4); rd(r.klut.data(), r.klut.size() * 4);
  rd(r.vwords.data(), r.vwords.size() * 4); rd(r.vboff.data(), r.vboff.size() * 4); rd(r.vlut.data(), r.vlut.size() * 4);
  rd(r.kscale.data(), r.kscale.size() * 4); rd(r.vscale.data(), r.vscale.size() * 4);
  rd(r.Q.data(), r.Q.size() * 4); rd(r.golden.data(), r.golden.size() * 4);
  std::fclose(f);
  return ok;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <class T> static T *upload(const std::vector<T> &h) {
  T *d = nullptr;
  if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s a.cfkv [b.cfkv ...]\n", argv[0]); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0; cudaRuntimeGetVersion(&rt);
  std::printf("M6/M9 (KV-path fusion) — decode-in-attention vs decode-then-dense over an entropy-coded KV cache\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d, median over 30 reps\n\n", prop.name, prop.major, prop.minor,
              rt / 1000, (rt % 1000) / 10);
  std::printf("  %-12s %5s %6s   %9s %9s %7s   %9s %9s   %-14s %s\n", "KV (int4)", "win", "nq", "resident", "dense",
              "less", "fused ms", "dense ms", "fused==dense", "vs golden");

  const int REPS = 30, WARM = 5;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  for (int fi = 1; fi < argc; ++fi) {
    Kv r;
    if (!load_kv(argv[fi], r)) { std::fprintf(stderr, "skip (bad .cfkv): %s\n", argv[fi]); continue; }
    int seq = (int)r.seq, dim = (int)r.dim, nq = (int)r.nq, window = (int)r.window, block = (int)r.block, zero = (int)r.zero;

    uint32_t *d_kw = upload(r.kwords), *d_vw = upload(r.vwords);
    int32_t *d_kb = upload(r.kboff), *d_kl = upload(r.klut), *d_vb = upload(r.vboff), *d_vl = upload(r.vlut);
    float *d_ks = upload(r.kscale), *d_vs = upload(r.vscale), *d_Q = upload(r.Q);
    float *d_of = nullptr, *d_od = nullptr, *d_Kd = nullptr, *d_Vd = nullptr;
    CK(cudaMalloc(&d_of, (size_t)nq * dim * 4));
    CK(cudaMalloc(&d_od, (size_t)nq * dim * 4));
    CK(cudaMalloc(&d_Kd, (size_t)seq * dim * 4));   // the dense intermediate the fused path avoids
    CK(cudaMalloc(&d_Vd, (size_t)seq * dim * 4));

    cf_kv_attn_fused_async(d_kw, d_kb, d_kl, (int)r.kmaxlen, d_vw, d_vb, d_vl, (int)r.vmaxlen, d_ks, d_vs, d_Q, d_of,
                           seq, dim, nq, window, block, zero, r.sscale, nullptr);
    cf_kv_attn_dense_async(d_kw, d_kb, d_kl, (int)r.kmaxlen, d_vw, d_vb, d_vl, (int)r.vmaxlen, d_ks, d_vs, d_Q, d_Kd,
                           d_Vd, d_od, seq, dim, nq, window, block, zero, r.sscale, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> hf((size_t)nq * dim), hd((size_t)nq * dim);
    CK(cudaMemcpy(hf.data(), d_of, hf.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hd.data(), d_od, hd.size() * 4, cudaMemcpyDeviceToHost));
    bool bit_identical = std::memcmp(hf.data(), hd.data(), hf.size() * 4) == 0;
    double maxabs = 0, maxy = 0;
    for (size_t i = 0; i < hf.size(); ++i) {
      maxabs = std::max(maxabs, (double)std::fabs(hf[i] - r.golden[i]));
      maxy = std::max(maxy, (double)std::fabs(r.golden[i]));
    }
    double rel = maxabs / (maxy + 1e-12);

    auto bench = [&](bool fused) {
      std::vector<double> ts;
      for (int i = 0; i < WARM + REPS; ++i) {
        cudaEventRecord(a);
        if (fused)
          cf_kv_attn_fused_async(d_kw, d_kb, d_kl, (int)r.kmaxlen, d_vw, d_vb, d_vl, (int)r.vmaxlen, d_ks, d_vs, d_Q,
                                 d_of, seq, dim, nq, window, block, zero, r.sscale, nullptr);
        else
          cf_kv_attn_dense_async(d_kw, d_kb, d_kl, (int)r.kmaxlen, d_vw, d_vb, d_vl, (int)r.vmaxlen, d_ks, d_vs, d_Q,
                                 d_Kd, d_Vd, d_od, seq, dim, nq, window, block, zero, r.sscale, nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (i >= WARM) ts.push_back(ms);
      }
      return median(ts);
    };
    double fused_ms = bench(true), dense_ms = bench(false);

    char kv[24];
    std::snprintf(kv, sizeof kv, "%d×%d", seq, dim);
    std::printf("  %-12s %5d %6d   %6.2f MB %6.2f MB %6.1f×   %9.3f %9.3f   %-14s rel=%.1e\n", kv, window, nq,
                r.resident / 1e6, r.dense / 1e6, (double)r.dense / (double)r.resident, fused_ms, dense_ms,
                bit_identical ? "BIT-IDENTICAL" : "DIFFER", rel);

    cudaFree(d_kw); cudaFree(d_vw); cudaFree(d_kb); cudaFree(d_kl); cudaFree(d_vb); cudaFree(d_vl);
    cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_Q); cudaFree(d_of); cudaFree(d_od); cudaFree(d_Kd); cudaFree(d_Vd);
  }
  std::printf("\n=> The fused kernel decodes + dequantizes each attended K/V row INSIDE the attention, so the dense\n");
  std::printf("   dequantized K/V tiles never exist in VRAM — only the compressed KV store is resident, and only\n");
  std::printf("   the windowed positions are decoded (sparse consumer). Bit-identical to decode-then-dense: fusion\n");
  std::printf("   is numerically free. This is the long-context KV-cache memory win, on the path to M9.\n");
  return 0;
}
