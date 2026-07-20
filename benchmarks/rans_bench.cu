// rans_bench.cu — M4: native block-rANS decode verified BIT-IDENTICAL to the Warp golden, and the honest crossover
// vs block-Huffman. rANS approaches H0; Huffman carries up to ~1 bit/symbol. Usage: ./rans_bench a.cfrs [b.cfrs ...]

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e = (x);                                                                                       \
    if (e != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                   \
      return 2;                                                                                                \
    }                                                                                                          \
  } while (0)

extern "C" cf_status cf_rans_decode_async(const uint8_t *data, const int32_t *byte_off, const uint32_t *state0,
                                          const int32_t *slot2sym, const int32_t *freq, const int32_t *cum,
                                          int block, int n, int32_t *out, void *stream);

struct Rs {
  uint64_t n = 0, rans_bits = 0, huff_bits = 0;
  uint32_t block = 0, V = 0, M = 0, nblocks = 0, data_bytes = 0, slotlen = 0;
  float h0 = 0;
  std::vector<uint8_t> data;
  std::vector<int32_t> byte_off, slot2sym, freq, cum, golden;
  std::vector<uint32_t> state0;
};

static bool load_rs(const char *path, Rs &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFRS", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  rd(&r.n, 8);
  rd(&r.block, 4); rd(&r.V, 4); rd(&r.M, 4); rd(&r.nblocks, 4); rd(&r.data_bytes, 4); rd(&r.slotlen, 4);
  rd(&r.h0, 4); rd(&r.rans_bits, 8); rd(&r.huff_bits, 8);
  if (!ok || version != 1) { std::fclose(f); return false; }
  r.data.resize(r.data_bytes);
  r.byte_off.resize(r.nblocks); r.state0.resize(r.nblocks);
  r.slot2sym.resize(r.slotlen); r.freq.resize(r.V); r.cum.resize(r.V);
  r.golden.resize(r.n);
  rd(r.data.data(), r.data_bytes);
  rd(r.byte_off.data(), r.byte_off.size() * 4);
  rd(r.state0.data(), r.state0.size() * 4);
  rd(r.slot2sym.data(), r.slot2sym.size() * 4);
  rd(r.freq.data(), r.freq.size() * 4);
  rd(r.cum.data(), r.cum.size() * 4);
  rd(r.golden.data(), r.golden.size() * 4);
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
  if (argc < 2) { std::fprintf(stderr, "usage: %s a.cfrs [b.cfrs ...]\n", argv[0]); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0; cudaRuntimeGetVersion(&rt);
  std::printf("M4 — block-rANS decode: native CUDA, verified vs Warp golden; the near-entropy coder vs Huffman\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d,  decode median over 30 reps\n\n", prop.name, prop.major, prop.minor,
              rt / 1000, (rt % 1000) / 10);
  std::printf("  %-10s %6s %6s   %8s %8s %8s %8s   %11s   %s\n", "n", "block", "H0", "rANS b/v", "Huff b/v",
              "winner", "vs H0", "decode M/s", "correct");

  const int REPS = 30, WARM = 5;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  for (int fi = 1; fi < argc; ++fi) {
    Rs r;
    if (!load_rs(argv[fi], r)) { std::fprintf(stderr, "skip (bad .cfrs): %s\n", argv[fi]); continue; }
    int n = (int)r.n, block = (int)r.block;

    uint8_t *d_data = upload(r.data);
    int32_t *d_boff = upload(r.byte_off), *d_slot = upload(r.slot2sym), *d_freq = upload(r.freq), *d_cum = upload(r.cum);
    uint32_t *d_st = upload(r.state0);
    int32_t *d_out = nullptr;
    CK(cudaMalloc(&d_out, (size_t)n * 4));

    cf_rans_decode_async(d_data, d_boff, d_st, d_slot, d_freq, d_cum, block, n, d_out, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<int32_t> h(n);
    CK(cudaMemcpy(h.data(), d_out, (size_t)n * 4, cudaMemcpyDeviceToHost));
    bool ok = std::equal(h.begin(), h.end(), r.golden.begin());

    std::vector<double> ts;
    for (int i = 0; i < WARM + REPS; ++i) {
      cudaEventRecord(a);
      cf_rans_decode_async(d_data, d_boff, d_st, d_slot, d_freq, d_cum, block, n, d_out, nullptr);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      if (i >= WARM) ts.push_back(ms);
    }
    double mps = (double)n / (median(ts) / 1e3) / 1e6;
    double rbv = r.rans_bits * 1.0 / n, hbv = r.huff_bits * 1.0 / n;
    char nbuf[16]; std::snprintf(nbuf, sizeof nbuf, "%d", n);
    std::printf("  %-10s %6d %6.2f   %8.3f %8.3f %8s %8.2f×   %11.0f   %s\n", nbuf, block, r.h0, rbv, hbv,
                rbv < hbv ? "rANS" : "Huffman", rbv / r.h0, mps, ok ? "BIT-IDENTICAL ✓" : "FAIL");

    cudaFree(d_data); cudaFree(d_boff); cudaFree(d_slot); cudaFree(d_freq); cudaFree(d_cum); cudaFree(d_st); cudaFree(d_out);
  }
  std::printf("\n=> Honest crossover: rANS approaches H0 (no per-symbol overhead), but carries a fixed 32-bit state\n");
  std::printf("   PER BLOCK — so at small blocks it loses to Huffman, and for multi-bit streams Huffman is already\n");
  std::printf("   near-optimal. rANS wins on LOW-entropy streams with LARGE blocks (bulk/archival decode). Both\n");
  std::printf("   decode one-thread-per-block on the GPU; pick the coder per data + access pattern.\n");
  return 0;
}
