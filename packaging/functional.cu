// functional.cu — GPU functional test THROUGH libchromofold.so (the hardware-gated half of the SDD-500 seam
// split). Loads a .cffm FM-index fixture, builds a cf_fm_view, calls cf_fm_count / cf_fm_ranges / cf_fm_locate
// via the shared library, and verifies count + locate positions bit-identical to the fixture's golden. Proves
// the packaged .so does compressed-domain search correctly — not just that its symbols link.
//
// Usage: ./functional fixtures/tiny.cffm   (needs an NVIDIA GPU)

#include "chromofold/chromofold_search.h" // the public ABI only (no detail headers -> no struct redefinition)

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

struct Fm {
  uint64_t n = 0, rrr_bytes = 0;
  uint32_t bits = 0, vocab = 0, sigma = 0, nblocks = 0, nsb = 0, cwords = 0, na = 0, owords = 0, sa_sample = 0;
  uint32_t mwords_len = 0, msb_len = 0, nsval = 0, npat = 0, patflat = 0, nloc = 0;
  std::vector<uint32_t> classes, offsets, mwords, count_golden;
  std::vector<int32_t> rank_a, off_a, offbase, zeros, C, msb, sval, pat, pstart, plen, locoff, locpos;
  std::vector<uint16_t> rank_d, off_d;
};

static bool load_fm(const char *path, Fm &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFFM", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  rd(&r.n, 8);
  rd(&r.bits, 4); rd(&r.vocab, 4); rd(&r.sigma, 4); rd(&r.nblocks, 4); rd(&r.nsb, 4); rd(&r.cwords, 4);
  rd(&r.na, 4); rd(&r.owords, 4); rd(&r.sa_sample, 4); rd(&r.mwords_len, 4); rd(&r.msb_len, 4);
  rd(&r.nsval, 4); rd(&r.npat, 4); rd(&r.patflat, 4); rd(&r.nloc, 4);
  rd(&r.rrr_bytes, 8);
  if (!ok || version != 1) { std::fclose(f); return false; }
  const uint32_t nsb1 = r.nsb + 1;
  r.classes.resize((size_t)r.bits * r.cwords); r.offsets.resize(r.owords);
  r.rank_a.resize((size_t)r.bits * r.na);      r.rank_d.resize((size_t)r.bits * nsb1);
  r.off_a.resize((size_t)r.bits * r.na);       r.off_d.resize((size_t)r.bits * nsb1);
  r.offbase.resize(r.bits);                    r.zeros.resize(r.bits);
  r.C.resize(r.sigma); r.mwords.resize(r.mwords_len); r.msb.resize(r.msb_len); r.sval.resize(r.nsval);
  r.pat.resize(r.patflat); r.pstart.resize(r.npat); r.plen.resize(r.npat); r.count_golden.resize(r.npat);
  r.locoff.resize(r.npat + 1); r.locpos.resize(r.nloc);
  rd(r.classes.data(), r.classes.size() * 4);  rd(r.offsets.data(), r.offsets.size() * 4);
  rd(r.rank_a.data(), r.rank_a.size() * 4);    rd(r.rank_d.data(), r.rank_d.size() * 2);
  rd(r.off_a.data(), r.off_a.size() * 4);      rd(r.off_d.data(), r.off_d.size() * 2);
  rd(r.offbase.data(), r.offbase.size() * 4);  rd(r.zeros.data(), r.zeros.size() * 4);
  rd(r.C.data(), r.C.size() * 4);              rd(r.mwords.data(), r.mwords.size() * 4);
  rd(r.msb.data(), r.msb.size() * 4);          rd(r.sval.data(), r.sval.size() * 4);
  rd(r.pat.data(), r.pat.size() * 4);          rd(r.pstart.data(), r.pstart.size() * 4);
  rd(r.plen.data(), r.plen.size() * 4);        rd(r.count_golden.data(), r.count_golden.size() * 4);
  rd(r.locoff.data(), r.locoff.size() * 4);    rd(r.locpos.data(), r.locpos.size() * 4);
  std::fclose(f);
  return ok;
}

template <class T> static T *up(const std::vector<T> &h) {
  T *d = nullptr;
  cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
  if (!h.empty()) cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "fixtures/tiny.cffm";
  Fm r;
  if (!load_fm(path, r)) { std::fprintf(stderr, "cannot load .cffm: %s\n", path); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

  // constant tables (offset width per class + Pascal), rebuilt on host like the engine's benchmark.
  std::vector<int> binom(256, 0), width(16, 0);
  for (int nn = 0; nn < 16; ++nn) { binom[nn * 16] = 1; for (int kk = 1; kk <= nn; ++kk)
      binom[nn * 16 + kk] = binom[(nn - 1) * 16 + kk - 1] + binom[(nn - 1) * 16 + kk]; }
  for (int k = 0; k < 16; ++k) { int c = binom[15 * 16 + k], w = 0; while ((1 << w) < c) ++w; width[k] = c > 1 ? w : 0; }

  cf_fm_view v;
  v.w.classes = up(r.classes); v.w.offsets = up(r.offsets);
  v.w.rank_a = up(r.rank_a);   v.w.rank_d = up(r.rank_d);
  v.w.off_a = up(r.off_a);     v.w.off_d = up(r.off_d);
  v.w.offbase = up(r.offbase); v.w.zeros = up(r.zeros);
  v.w.width = up(width);       v.w.binom = up(binom);
  v.w.bits = (int)r.bits; v.w.cwords = (int)r.cwords; v.w.nsb = (int)r.nsb; v.w.na = (int)r.na;
  v.C = up(r.C); v.mwords = up(r.mwords); v.msb = up(r.msb); v.sval = up(r.sval);
  v.sigma = (int)r.sigma; v.n = (int)r.n; v.sa_sample = (int)r.sa_sample;

  int32_t *d_pat = up(r.pat), *d_ps = up(r.pstart), *d_pl = up(r.plen);
  uint32_t *d_cnt = nullptr; int32_t *d_lo = nullptr, *d_hi = nullptr;
  CK(cudaMalloc(&d_cnt, (size_t)r.npat * 4));
  CK(cudaMalloc(&d_lo, (size_t)r.npat * 4)); CK(cudaMalloc(&d_hi, (size_t)r.npat * 4));

  std::printf("libchromofold functional (through the .so) — %s, device %s\n", path, prop.name);
  std::printf("  index: n=%llu bits=%u sigma=%u  patterns=%u\n", (unsigned long long)r.n, r.bits, r.sigma, r.npat);

  // count
  if (cf_fm_count_async(v, d_pat, d_ps, d_pl, d_cnt, r.npat, nullptr) != CF_OK) { std::fprintf(stderr, "count status\n"); return 2; }
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> hcnt(r.npat);
  CK(cudaMemcpy(hcnt.data(), d_cnt, (size_t)r.npat * 4, cudaMemcpyDeviceToHost));
  bool ok_count = std::equal(hcnt.begin(), hcnt.end(), r.count_golden.begin());

  // locate: ranges -> flatten SA rows -> locate -> sorted-per-pattern compare to golden positions
  cf_fm_ranges_async(v, d_pat, d_ps, d_pl, d_lo, d_hi, r.npat, nullptr);
  CK(cudaDeviceSynchronize());
  std::vector<int32_t> hlo(r.npat), hhi(r.npat);
  CK(cudaMemcpy(hlo.data(), d_lo, (size_t)r.npat * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hhi.data(), d_hi, (size_t)r.npat * 4, cudaMemcpyDeviceToHost));
  std::vector<int32_t> rows; std::vector<uint32_t> off(r.npat + 1, 0);
  for (uint32_t i = 0; i < r.npat; ++i) { for (int rr = hlo[i]; rr < hhi[i]; ++rr) rows.push_back(rr); off[i + 1] = (uint32_t)rows.size(); }
  bool ok_locate = true;
  if (!rows.empty()) {
    int32_t *d_r = up(rows), *d_pos = nullptr;
    CK(cudaMalloc(&d_pos, rows.size() * 4));
    cf_fm_locate_async(v, d_r, d_pos, rows.size(), nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<int32_t> pos(rows.size());
    CK(cudaMemcpy(pos.data(), d_pos, rows.size() * 4, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < r.npat && ok_locate; ++i) {
      std::vector<int32_t> mine(pos.begin() + off[i], pos.begin() + off[i + 1]);
      std::vector<int32_t> gold(r.locpos.begin() + r.locoff[i], r.locpos.begin() + r.locoff[i + 1]);
      std::sort(mine.begin(), mine.end()); std::sort(gold.begin(), gold.end());
      if (mine != gold) ok_locate = false;
    }
  }

  std::printf("  cf_fm_count  vs golden : %s\n", ok_count ? "BIT-IDENTICAL ✓" : "FAIL");
  std::printf("  cf_fm_locate vs golden : %s (%zu occurrences)\n", ok_locate ? "MATCH ✓" : "FAIL", rows.size());
  bool ok = ok_count && ok_locate;
  std::printf("%s\n", ok ? "PASS — libchromofold performs FM-index search correctly through the shared library."
                         : "FAIL");
  return ok ? 0 : 3;
}
