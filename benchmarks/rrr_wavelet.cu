// rrr_wavelet.cu — M4 (wavelet wiring): native RRR-backed wavelet `access` + `rank`, verified BIT-IDENTICAL to the
// Warp golden, and the entropy-size win measured against the packed wavelet on a BWT. The resident index is
// entropy-sized (every level an RRR bitvector) yet still GPU-searchable: one object, compact AND navigable.
// Usage: ./rrr_wavelet a.cfrw [b.cfrw ...]

#include "chromofold/detail/rrr_wavelet_device.cuh"

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

struct Rrw {
  uint64_t n = 0, rrr_bytes = 0, packed_bytes = 0;
  uint32_t bits = 0, vocab = 0, nblocks = 0, nsb = 0, cwords = 0, na = 0, owords = 0, nqueries = 0, nrank = 0;
  float h0 = 0;
  std::vector<uint32_t> classes, offsets, pos, acc_golden, rank_c, rank_i, rank_golden;
  std::vector<int32_t> rank_a, off_a, offbase, zeros;
  std::vector<uint16_t> rank_d, off_d;
};

static bool load_rrw(const char *path, Rrw &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFRW", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  rd(&r.n, 8);
  rd(&r.bits, 4); rd(&r.vocab, 4); rd(&r.nblocks, 4); rd(&r.nsb, 4); rd(&r.cwords, 4); rd(&r.na, 4); rd(&r.owords, 4);
  rd(&r.nqueries, 4); rd(&r.nrank, 4);
  rd(&r.rrr_bytes, 8); rd(&r.packed_bytes, 8); rd(&r.h0, 4);
  if (!ok || version != 1) { std::fclose(f); return false; }
  const uint32_t nsb1 = r.nsb + 1;
  r.classes.resize((size_t)r.bits * r.cwords);
  r.offsets.resize(r.owords);
  r.rank_a.resize((size_t)r.bits * r.na);
  r.rank_d.resize((size_t)r.bits * nsb1);
  r.off_a.resize((size_t)r.bits * r.na);
  r.off_d.resize((size_t)r.bits * nsb1);
  r.offbase.resize(r.bits);
  r.zeros.resize(r.bits);
  r.pos.resize(r.nqueries);
  r.acc_golden.resize(r.nqueries);
  r.rank_c.resize(r.nrank);
  r.rank_i.resize(r.nrank);
  r.rank_golden.resize(r.nrank);
  rd(r.classes.data(), r.classes.size() * 4);
  rd(r.offsets.data(), r.offsets.size() * 4);
  rd(r.rank_a.data(), r.rank_a.size() * 4);
  rd(r.rank_d.data(), r.rank_d.size() * 2);
  rd(r.off_a.data(), r.off_a.size() * 4);
  rd(r.off_d.data(), r.off_d.size() * 2);
  rd(r.offbase.data(), r.offbase.size() * 4);
  rd(r.zeros.data(), r.zeros.size() * 4);
  rd(r.pos.data(), r.pos.size() * 4);
  rd(r.acc_golden.data(), r.acc_golden.size() * 4);
  rd(r.rank_c.data(), r.rank_c.size() * 4);
  rd(r.rank_i.data(), r.rank_i.size() * 4);
  rd(r.rank_golden.data(), r.rank_golden.size() * 4);
  std::fclose(f);
  return ok;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class T> static T *upload(const std::vector<T> &h) {
  T *d = nullptr;
  if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s a.cfrw [b.cfrw ...]\n", argv[0]); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0; cudaRuntimeGetVersion(&rt);

  // binom[16x16] (Pascal) + width[16] (offset bit-width per class), computed once on the host (constants).
  std::vector<int> binom(256, 0), width(16, 0);
  for (int nn = 0; nn < 16; ++nn) {
    binom[nn * 16] = 1;
    for (int kk = 1; kk <= nn; ++kk) binom[nn * 16 + kk] = binom[(nn - 1) * 16 + kk - 1] + binom[(nn - 1) * 16 + kk];
  }
  for (int k = 0; k < 16; ++k) {
    int c = binom[15 * 16 + k], w = 0;
    while ((1 << w) < c) ++w;
    width[k] = (c > 1) ? w : 0;
  }
  int *d_binom = upload(binom), *d_width = upload(width);
  CK(d_binom && d_width ? cudaSuccess : cudaErrorMemoryAllocation);

  std::printf("M4 (wavelet wiring) — RRR-backed wavelet access + rank: native CUDA, verified vs Warp golden\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d,  kernel-only median over 30 reps\n\n", prop.name, prop.major,
              prop.minor, rt / 1000, (rt % 1000) / 10);
  std::printf("  %6s %5s   %8s %8s %7s   %10s %10s   %-16s\n", "vocab", "bits", "RRR b/t", "pack b/t", "smaller",
              "access ns", "rank ns", "correct (access|rank)");

  const int REPS = 30, WARM = 5;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  for (int fi = 1; fi < argc; ++fi) {
    Rrw r;
    if (!load_rrw(argv[fi], r)) { std::fprintf(stderr, "skip (bad .cfrw): %s\n", argv[fi]); continue; }

    cf_rrrw_view v;
    v.classes = upload(r.classes); v.offsets = upload(r.offsets);
    v.rank_a = upload(r.rank_a);   v.rank_d = upload(r.rank_d);
    v.off_a = upload(r.off_a);     v.off_d = upload(r.off_d);
    v.offbase = upload(r.offbase); v.zeros = upload(r.zeros);
    v.width = d_width;             v.binom = d_binom;
    v.bits = (int)r.bits; v.cwords = (int)r.cwords; v.nsb = (int)r.nsb; v.na = (int)r.na;

    uint32_t *d_pos = upload(r.pos), *d_rc = upload(r.rank_c), *d_ri = upload(r.rank_i);
    uint32_t *d_aout = nullptr, *d_rout = nullptr;
    CK(cudaMalloc(&d_aout, (size_t)r.nqueries * 4));
    CK(cudaMalloc(&d_rout, (size_t)r.nrank * 4));

    // correctness first (M0 gate): both access and rank must be BIT-IDENTICAL to the golden before any timing.
    cf_rrrw_access_async(v, d_pos, d_aout, r.nqueries, nullptr);
    cf_rrrw_rank_async(v, d_rc, d_ri, d_rout, r.nrank, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> ha(r.nqueries), hr(r.nrank);
    CK(cudaMemcpy(ha.data(), d_aout, (size_t)r.nqueries * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(), d_rout, (size_t)r.nrank * 4, cudaMemcpyDeviceToHost));
    bool ok_a = std::equal(ha.begin(), ha.end(), r.acc_golden.begin());
    bool ok_r = std::equal(hr.begin(), hr.end(), r.rank_golden.begin());

    auto time_ns = [&](bool is_access) {
      std::vector<double> ts;
      for (int i = 0; i < WARM + REPS; ++i) {
        cudaEventRecord(a);
        if (is_access) cf_rrrw_access_async(v, d_pos, d_aout, r.nqueries, nullptr);
        else           cf_rrrw_rank_async(v, d_rc, d_ri, d_rout, r.nrank, nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (i >= WARM) ts.push_back(ms);
      }
      return median(ts) * 1e6 / (double)(is_access ? r.nqueries : r.nrank);
    };
    double acc_ns = time_ns(true), rnk_ns = time_ns(false);

    double rrr_bt = r.rrr_bytes * 8.0 / (double)r.n, pack_bt = r.packed_bytes * 8.0 / (double)r.n;
    std::printf("  %6u %5u   %8.3f %8.3f %6.2f×   %10.2f %10.2f   %s | %s\n", r.vocab, r.bits, rrr_bt, pack_bt,
                (double)r.packed_bytes / (double)r.rrr_bytes, acc_ns, rnk_ns,
                ok_a ? "BIT-IDENTICAL ✓" : "FAIL", ok_r ? "✓" : "FAIL");

    for (auto p : {v.classes, v.offsets}) cudaFree((void *)p);
    cudaFree((void *)v.rank_a); cudaFree((void *)v.rank_d); cudaFree((void *)v.off_a); cudaFree((void *)v.off_d);
    cudaFree((void *)v.offbase); cudaFree((void *)v.zeros);
    cudaFree(d_pos); cudaFree(d_rc); cudaFree(d_ri); cudaFree(d_aout); cudaFree(d_rout);
  }
  std::printf("\n=> Every wavelet level is an RRR bitvector decoded in registers (superblock jump + class scan +\n");
  std::printf("   one combinatorial block decode), with two-level superblock samples. The index shrinks toward the\n");
  std::printf("   BWT's entropy while access/rank stay GPU-resident — the compact, searchable self-index the\n");
  std::printf("   FM-index (M7) will ride on. Price: RRR decode makes access/rank slower than the packed wavelet.\n");
  return 0;
}
