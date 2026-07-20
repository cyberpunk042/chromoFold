// rank_bench.cu — M3: verify native wavelet `rank` bit-for-bit vs the frozen Warp golden, and benchmark the
// two rank-directory designs head-to-head (linear word-scan vs two-level coarse+per-word-prefix). Answers the
// architecture §6 / brief §5.2 question: does the two-level directory's extra memory buy lower rank latency?
//
// Usage: ./rank_bench ref_V256.cfwv [ref_V32768.cfwv ...]

#include "chromofold/chromofold.h"
#include "reference_io.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e = (x);                                                                                       \
    if (e != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                   \
      return 2;                                                                                                \
    }                                                                                                          \
  } while (0)

extern "C" cf_status cf_rank2_async(cf_wavelet_view index, const int32_t *coarse, const uint16_t *fine,
                                    int ncoarse, int CB, const uint32_t *sym, const uint32_t *pos, uint32_t *out,
                                    size_t count, void *stream);

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s ref1.cfwv [ref2.cfwv ...]\n", argv[0]);
    return 1;
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int rt = 0;
  cudaRuntimeGetVersion(&rt);
  const int CB = 64;  // coarse superblock = 64 words = 2048 bits; per-word uint16 prefix within it

  std::printf("M3 — wavelet rank: native CUDA, verified vs Warp golden; linear vs two-level rank directory\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d,  kernel-only median over 30 reps\n\n", prop.name, prop.major,
              prop.minor, rt / 1000, (rt % 1000) / 10);
  std::printf("  %6s %5s %8s   %-11s %-11s %8s   %-9s %-9s\n", "vocab", "bits", "correct", "linear ns",
              "2-level ns", "speedup", "lin dir", "2lvl dir");

  const int REPS = 30, WARM = 5, threads = 256;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));

  for (int f = 1; f < argc; ++f) {
    Ref r;
    if (!cf_load_reference(argv[f], r)) {
      std::fprintf(stderr, "skip (need .cfwv v3): %s\n", argv[f]);
      continue;
    }
    size_t nq = r.nrank;
    int nw = (int)r.nwords, levels = (int)r.levels;
    int ncoarse = (nw + CB - 1) / CB + 1;  // +1 sentinel for a pos=n boundary query (word == nwords)

    // build the two-level directory on the CPU (previews the C++ builder, M5). fine has an nwords+1 sentinel.
    std::vector<int32_t> coarse((size_t)levels * ncoarse);
    std::vector<uint16_t> fine((size_t)levels * (nw + 1));
    for (int lvl = 0; lvl < levels; ++lvl) {
      int running = 0, within = 0;
      const uint32_t *w = &r.words[(size_t)lvl * nw];
      for (int word = 0; word <= nw; ++word) {  // inclusive: fill the sentinel at word == nwords
        if (word % CB == 0) { coarse[(size_t)lvl * ncoarse + word / CB] = running; within = 0; }
        fine[(size_t)lvl * (nw + 1) + word] = (uint16_t)within;
        if (word < nw) {
          int pc = __builtin_popcount(w[word]);
          within += pc;
          running += pc;
        }
      }
    }

    uint32_t *d_words, *d_c, *d_i, *d_out, *d_out2;
    int32_t *d_sb, *d_zeros, *d_coarse;
    uint16_t *d_fine;
    CK(cudaMalloc(&d_words, r.words.size() * 4));
    CK(cudaMalloc(&d_sb, r.sb.size() * 4));
    CK(cudaMalloc(&d_zeros, r.zeros.size() * 4));
    CK(cudaMalloc(&d_coarse, coarse.size() * 4));
    CK(cudaMalloc(&d_fine, fine.size() * 2));
    CK(cudaMalloc(&d_c, nq * 4));
    CK(cudaMalloc(&d_i, nq * 4));
    CK(cudaMalloc(&d_out, nq * 4));
    CK(cudaMalloc(&d_out2, nq * 4));
    CK(cudaMemcpy(d_words, r.words.data(), r.words.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sb, r.sb.data(), r.sb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_zeros, r.zeros.data(), r.zeros.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_coarse, coarse.data(), coarse.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_fine, fine.data(), fine.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_c, r.rank_c.data(), nq * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_i, r.rank_i.data(), nq * 4, cudaMemcpyHostToDevice));
    cf_wavelet_view view{d_words, d_sb, d_zeros, r.n, r.levels, r.nwords, r.nblocks};

    // correctness: both rank paths bit-identical to the golden counts
    cf_rank_async(view, d_c, d_i, d_out, nq, nullptr);
    cf_rank2_async(view, d_coarse, d_fine, ncoarse, CB, d_c, d_i, d_out2, nq, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> h1(nq), h2(nq);
    CK(cudaMemcpy(h1.data(), d_out, nq * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2.data(), d_out2, nq * 4, cudaMemcpyDeviceToHost));
    bool ok = std::equal(h1.begin(), h1.end(), r.rank_golden.begin()) &&
              std::equal(h2.begin(), h2.end(), r.rank_golden.begin());
    if (!ok) {
      size_t mm = 0;
      for (size_t k = 0; k < nq; ++k) mm += (h1[k] != r.rank_golden[k]);
      std::fprintf(stderr, "  [dbg] c=%u i=%u  linear=%u twolvl=%u golden=%u  native-agree=%d  mism=%zu/%zu\n",
                   r.rank_c[0], r.rank_i[0], h1[0], h2[0], r.rank_golden[0], (int)(h1 == h2), mm, nq);
    }

    auto timeit = [&](auto fn) -> double {
      std::vector<double> ts;
      for (int i = 0; i < WARM + REPS; ++i) {
        cudaEventRecord(a);
        fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (i >= WARM) ts.push_back(ms);
      }
      return median(ts) * 1e6 / (double)nq;  // ns/query
    };
    double lin = timeit([&] { cf_rank_async(view, d_c, d_i, d_out, nq, nullptr); });
    double two = timeit([&] { cf_rank2_async(view, d_coarse, d_fine, ncoarse, CB, d_c, d_i, d_out2, nq, nullptr); });

    double lin_dir = r.sb.size() * 4.0 / ((double)levels * nw);          // directory bytes per bitplane word
    double two_dir = (coarse.size() * 4.0 + fine.size() * 2.0) / ((double)levels * nw);
    std::printf("  %6u %5u %8s   %9.2f %9.2f   %7.2f×   %6.2fB/w %6.2fB/w\n", r.vocab, r.levels,
                ok ? "✓" : "FAIL", lin, two, lin / two, lin_dir, two_dir);

    cudaFree(d_words); cudaFree(d_sb); cudaFree(d_zeros); cudaFree(d_coarse); cudaFree(d_fine);
    cudaFree(d_c); cudaFree(d_i); cudaFree(d_out); cudaFree(d_out2);
  }
  std::printf("\n=> two-level trades directory memory (per bitplane word) for eliminating the in-superblock word\n");
  std::printf("   scan. Whether it wins depends on whether rank is latency-bound on the scan or bandwidth-bound\n");
  std::printf("   on the larger directory — a benchmark-driven decision, per constitution P7.\n");
  return 0;
}
