// delta_bench.cu — M8: reference/delta cluster decode verified BIT-IDENTICAL to the golden originals, with the
// cross-sequence dedup footprint and the cost to add one turn. Reconstructs every token of every member (base
// overridden by its sparse deltas) and checks against the frozen originals. Usage: ./delta_bench a.cfdc

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e = (x);                                                                                       \
    if (e != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                   \
      return 2;                                                                                                \
    }                                                                                                          \
  } while (0)

extern "C" cf_status cf_delta_fetch_async(const int32_t *base, int nbase, const int32_t *dpos, const int32_t *dval,
                                          const int32_t *dstart, const int32_t *dlen, const int32_t *device_leaf,
                                          const int32_t *device_pos, int32_t *out, size_t count, void *stream);

struct Dc {
  uint64_t total = 0;
  uint32_t N = 0, nbase = 0, ndelta = 0, vocab = 0;
  std::vector<int32_t> base, lengths, dstart, dlen, dpos, dval, ostart, originals;
};

static bool load_dc(const char *path, Dc &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFDC", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  rd(&r.N, 4); rd(&r.nbase, 4); rd(&r.ndelta, 4); rd(&r.vocab, 4); rd(&r.total, 8);
  if (!ok || version != 1) { std::fclose(f); return false; }
  r.base.resize(r.nbase); r.lengths.resize(r.N); r.dstart.resize(r.N); r.dlen.resize(r.N);
  r.dpos.resize(r.ndelta); r.dval.resize(r.ndelta); r.ostart.resize(r.N + 1); r.originals.resize(r.total);
  rd(r.base.data(), r.base.size() * 4);
  rd(r.lengths.data(), r.lengths.size() * 4);
  rd(r.dstart.data(), r.dstart.size() * 4);
  rd(r.dlen.data(), r.dlen.size() * 4);
  rd(r.dpos.data(), r.dpos.size() * 4);
  rd(r.dval.data(), r.dval.size() * 4);
  rd(r.ostart.data(), r.ostart.size() * 4);
  rd(r.originals.data(), r.originals.size() * 4);
  std::fclose(f);
  return ok;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <class T> static T *upload(const std::vector<T> &h) {
  T *d = nullptr;
  if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
  if (!h.empty()) cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s a.cfdc\n", argv[0]); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0; cudaRuntimeGetVersion(&rt);
  Dc r;
  if (!load_dc(argv[1], r)) { std::fprintf(stderr, "bad .cfdc\n"); return 1; }

  int32_t *d_base = upload(r.base), *d_dpos = upload(r.dpos), *d_dval = upload(r.dval);
  int32_t *d_dstart = upload(r.dstart), *d_dlen = upload(r.dlen);

  // decode-all: one query per token of every member -> compare to the frozen originals (full bit-identical check).
  size_t total = r.total;
  std::vector<int32_t> leaf(total), pos(total);
  for (uint32_t i = 0; i < r.N; ++i) {
    int off = r.ostart[i], L = r.lengths[i];
    for (int p = 0; p < L; ++p) { leaf[off + p] = (int)i; pos[off + p] = p; }
  }
  int32_t *d_leaf = upload(leaf), *d_pos = upload(pos), *d_out = nullptr;
  CK(cudaMalloc(&d_out, total * 4));
  cf_delta_fetch_async(d_base, (int)r.nbase, d_dpos, d_dval, d_dstart, d_dlen, d_leaf, d_pos, d_out, total, nullptr);
  CK(cudaDeviceSynchronize());
  std::vector<int32_t> got(total);
  CK(cudaMemcpy(got.data(), d_out, total * 4, cudaMemcpyDeviceToHost));
  bool ok = std::equal(got.begin(), got.end(), r.originals.begin());

  // throughput: a big random-access batch (leaf, pos) reconstructed in VRAM.
  size_t Q = std::min<size_t>(total, 1u << 20);
  std::vector<int32_t> ql(Q), qp(Q);
  std::mt19937_64 rng(7);
  std::uniform_int_distribution<int> RL(0, (int)r.N - 1);
  for (size_t i = 0; i < Q; ++i) { int l = RL(rng); ql[i] = l; qp[i] = (int)(rng() % r.lengths[l]); }
  int32_t *d_ql = upload(ql), *d_qp = upload(qp), *d_qo = nullptr;
  CK(cudaMalloc(&d_qo, Q * 4));
  const int REPS = 30, WARM = 5;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<double> ts;
  for (int i = 0; i < WARM + REPS; ++i) {
    cudaEventRecord(a);
    cf_delta_fetch_async(d_base, (int)r.nbase, d_dpos, d_dval, d_dstart, d_dlen, d_ql, d_qp, d_qo, Q, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    if (i >= WARM) ts.push_back(ms);
  }
  double mtps = (double)Q / (median(ts) / 1e3) / 1e6;

  int suffix = (r.N ? r.lengths[0] - (int)r.nbase : 0);
  double cluster = r.nbase * 4.0 + r.ndelta * 8.0 + r.N * 8.0;
  double dup = r.total * 4.0;
  double add_turn = suffix * 8.0;                          // one appended turn = its suffix stored as deltas
  double add_dup = (r.nbase + suffix) * 4.0;               // duplicated: a whole new copy

  std::printf("M8 — reference/delta cluster decode: native CUDA, verified vs golden originals (cross-seq dedup)\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d\n\n", prop.name, prop.major, prop.minor, rt / 1000, (rt % 1000) / 10);
  std::printf("  cluster: %u members × (base %u + suffix %d) = %llu tokens\n", r.N, r.nbase, suffix,
              (unsigned long long)r.total);
  std::printf("  resident (base + sparse deltas): %.1f KB   vs   duplicated %.2f MB   => %.1f× less VRAM\n",
              cluster / 1e3, dup / 1e6, dup / cluster);
  std::printf("  reconstruct: %s (%llu / %llu tokens)   random-access fetch %.0f M tok/s\n",
              ok ? "BIT-IDENTICAL ✓" : "FAIL", (unsigned long long)(ok ? total : 0), (unsigned long long)total,
              mtps);
  std::printf("  add one turn: +%.0f B (suffix as deltas) vs +%.0f B (a duplicated copy) => %.0f× cheaper/turn\n",
              add_turn, add_dup, add_dup / add_turn);
  std::printf("\n=> The cluster stores the shared reference ONCE + each member's sparse diff, so N near-duplicate\n");
  std::printf("   requests cost base + small per-request deltas, not N full copies — reconstructed on the GPU by a\n");
  std::printf("   binary search over each leaf's deltas (base[pos] overridden by the deepest match). Adding a\n");
  std::printf("   conversation turn appends its suffix as deltas — no base recopy. Honest: sparse (pos,val) deltas\n");
  std::printf("   cost 8 B/token, so a LONG contiguous suffix is better stored as a plain array; the win is the\n");
  std::printf("   shared prefix amortised across the batch (P2/P10: cross-sequence dedup).\n");
  return ok ? 0 : 1;
}
