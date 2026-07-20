// rrr_bench.cu — M4: native RRR rank1 verified vs the Warp golden, and the entropy memory-latency frontier.
// RRR compresses skewed planes far below 1 bit/bit; the price is a combinatorial in-register block decode.
// Usage: ./rrr_bench a.cfrr [b.cfrr ...]  (one per density)

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

#include <algorithm>
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

extern "C" cf_status cf_rrr_rank_async(const uint32_t *classes, const uint32_t *offsets, const int32_t *sbrank,
                                       const int32_t *sboff, const int *width, const int *binom,
                                       const uint32_t *positions, uint32_t *out, size_t count, void *stream);

struct Rrr {
  uint64_t n = 0, class_bits = 0, offset_bits = 0;
  uint32_t T = 0, SB = 0, nblocks = 0, nsb = 0, cwords = 0, owords = 0, nqueries = 0;
  float density = 0;
  std::vector<uint32_t> classes, offsets, pos, golden;
  std::vector<int32_t> sbrank, sboff;
};

static bool load_rrr(const char *path, Rrr &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFRR", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  rd(&r.n, 8);
  rd(&r.T, 4); rd(&r.SB, 4); rd(&r.nblocks, 4); rd(&r.nsb, 4);
  rd(&r.cwords, 4); rd(&r.owords, 4); rd(&r.nqueries, 4);
  rd(&r.density, 4); rd(&r.class_bits, 8); rd(&r.offset_bits, 8);
  if (!ok || version != 1) { std::fclose(f); return false; }
  r.classes.resize(r.cwords); r.offsets.resize(r.owords);
  r.sbrank.resize(r.nsb + 1); r.sboff.resize(r.nsb + 1);
  r.pos.resize(r.nqueries); r.golden.resize(r.nqueries);
  rd(r.classes.data(), r.cwords * 4);
  rd(r.offsets.data(), r.owords * 4);
  rd(r.sbrank.data(), (r.nsb + 1) * 4);
  rd(r.sboff.data(), (r.nsb + 1) * 4);
  rd(r.pos.data(), r.nqueries * 4);
  rd(r.golden.data(), r.nqueries * 4);
  std::fclose(f);
  return ok;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s a.cfrr [b.cfrr ...]\n", argv[0]); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0; cudaRuntimeGetVersion(&rt);

  // binom[16x16] (Pascal) + width[16] (offset bit-width per class), computed once on the host
  std::vector<int> binom(256, 0), width(16, 0);
  for (int nn = 0; nn < 16; ++nn) {
    binom[nn * 16] = 1;
    for (int kk = 1; kk <= nn; ++kk) binom[nn * 16 + kk] = binom[(nn - 1) * 16 + kk - 1] + binom[(nn - 1) * 16 + kk];
  }
  for (int k = 0; k < 16; ++k) {
    int c = binom[15 * 16 + k];
    int w = 0; while ((1 << w) < c) ++w;
    width[k] = (c > 1) ? w : 0;
  }
  int *d_binom, *d_width;
  CK(cudaMalloc(&d_binom, 256 * 4)); CK(cudaMalloc(&d_width, 16 * 4));
  CK(cudaMemcpy(d_binom, binom.data(), 256 * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_width, width.data(), 16 * 4, cudaMemcpyHostToDevice));

  std::printf("M4 — RRR bitvector rank1: native CUDA, verified vs Warp golden; the entropy frontier\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d,  kernel-only median over 30 reps\n\n", prop.name, prop.major,
              prop.minor, rt / 1000, (rt % 1000) / 10);
  std::printf("  %8s %8s   %10s %8s   %-11s\n", "density", "H0", "RRR b/bit", "vs pack", "rank1 correct");

  const int REPS = 30, WARM = 5, threads = 256;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  for (int fi = 1; fi < argc; ++fi) {
    Rrr r;
    if (!load_rrr(argv[fi], r)) { std::fprintf(stderr, "skip (bad .cfrr): %s\n", argv[fi]); continue; }
    size_t nq = r.nqueries;

    uint32_t *d_cls, *d_off, *d_pos, *d_out;
    int32_t *d_sbr, *d_sbo;
    CK(cudaMalloc(&d_cls, r.cwords * 4)); CK(cudaMalloc(&d_off, r.owords * 4));
    CK(cudaMalloc(&d_sbr, (r.nsb + 1) * 4)); CK(cudaMalloc(&d_sbo, (r.nsb + 1) * 4));
    CK(cudaMalloc(&d_pos, nq * 4)); CK(cudaMalloc(&d_out, nq * 4));
    CK(cudaMemcpy(d_cls, r.classes.data(), r.cwords * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_off, r.offsets.data(), r.owords * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sbr, r.sbrank.data(), (r.nsb + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sbo, r.sboff.data(), (r.nsb + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, r.pos.data(), nq * 4, cudaMemcpyHostToDevice));

    cf_rrr_rank_async(d_cls, d_off, d_sbr, d_sbo, d_width, d_binom, d_pos, d_out, nq, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> h(nq);
    CK(cudaMemcpy(h.data(), d_out, nq * 4, cudaMemcpyDeviceToHost));
    bool ok = std::equal(h.begin(), h.end(), r.golden.begin());

    std::vector<double> ts;
    for (int i = 0; i < WARM + REPS; ++i) {
      cudaEventRecord(a);
      cf_rrr_rank_async(d_cls, d_off, d_sbr, d_sbo, d_width, d_binom, d_pos, d_out, nq, nullptr);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      if (i >= WARM) ts.push_back(ms);
    }
    double ns = median(ts) * 1e6 / (double)nq;
    double rrr_bpb = (r.class_bits + r.offset_bits + (double)(r.nsb + 1) * 8 * 8) / (double)r.n;
    double p = r.density, h0 = (p > 0 && p < 1) ? -(p * std::log2(p) + (1 - p) * std::log2(1 - p)) : 0.0;
    std::printf("  %8.4f %8.3f   %10.3f %7.2f×   %s (%.2f ns)\n", p, h0, rrr_bpb, 1.0 / rrr_bpb,
                ok ? "BIT-IDENTICAL ✓" : "FAIL", ns);

    cudaFree(d_cls); cudaFree(d_off); cudaFree(d_sbr); cudaFree(d_sbo); cudaFree(d_pos); cudaFree(d_out);
  }
  std::printf("\n=> RRR rank1 runs GPU-resident over the entropy-sized index: a superblock jump + a short class\n");
  std::printf("   scan + ONE combinatorial block decode in registers. Skewed planes cost far below 1 bit/bit and\n");
  std::printf("   approach H0 — the memory win the FM-index rides on; the price is the in-register decode.\n");
  return 0;
}
