// frontier.cu — Experiment A: the price of addressability. Raw GPU gather vs wavelet `access`, across
// vocabulary widths and query patterns, with the reproducibility envelope (P7). Answers the brief's #1 gap:
// what does keeping the data *addressable and searchable* cost against a plain uint8/16/32 gather?
//
// Usage: ./frontier ref_V4.cfwv ref_V256.cfwv ... (one frozen .cfwv per vocabulary)

#include "chromofold/chromofold.h"
#include "reference_io.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e = (x);                                                                                       \
    if (e != cudaSuccess) {                                                                                    \
      std::fprintf(stderr, "CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);                   \
      return 2;                                                                                                \
    }                                                                                                          \
  } while (0)

template <typename T>
__global__ void gather_kernel(const T *tokens, const uint32_t *pos, uint32_t *out, size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t < count) out[t] = (uint32_t)tokens[pos[t]];
}

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
  std::printf("Experiment A — price of addressability: raw GPU gather vs wavelet access\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d,  n=1M, 100K queries, kernel-only median over 30 reps\n\n",
              prop.name, prop.major, prop.minor, rt / 1000, (rt % 1000) / 10);
  std::printf("  %5s %5s %8s %9s  %-11s %9s %9s %8s\n", "vocab", "bits", "raw B/tk", "wav b/tk", "pattern",
              "gather ns", "wavelet ns", "×cost");

  const int threads = 256, REPS = 30, WARM = 5;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));

  for (int f = 1; f < argc; ++f) {
    Ref r;
    if (!cf_load_reference(argv[f], r)) {
      std::fprintf(stderr, "skip (bad file): %s\n", argv[f]);
      continue;
    }
    size_t nq = r.nqueries, blocks = (nq + threads - 1) / threads;

    uint32_t *d_words, *d_out;
    int32_t *d_sb, *d_zeros;
    void *d_raw;
    uint32_t *d_pos;
    CK(cudaMalloc(&d_words, r.words.size() * 4));
    CK(cudaMalloc(&d_sb, r.sb.size() * 4));
    CK(cudaMalloc(&d_zeros, r.zeros.size() * 4));
    CK(cudaMalloc(&d_raw, r.raw.size()));
    CK(cudaMalloc(&d_pos, nq * 4));
    CK(cudaMalloc(&d_out, nq * 4));
    CK(cudaMemcpy(d_words, r.words.data(), r.words.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sb, r.sb.data(), r.sb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_zeros, r.zeros.data(), r.zeros.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_raw, r.raw.data(), r.raw.size(), cudaMemcpyHostToDevice));
    cf_wavelet_view view{d_words, d_sb, d_zeros, r.n, r.levels, r.nwords, r.nblocks};

    // three query patterns from the same positions: uniform (as frozen), sorted (coalesced), contiguous
    std::vector<uint32_t> uniform = r.pos, sorted = r.pos, contig(nq);
    std::sort(sorted.begin(), sorted.end());
    std::iota(contig.begin(), contig.end(), 0u);
    const char *pat_name[3] = {"uniform", "sorted", "contiguous"};
    std::vector<uint32_t> *pats[3] = {&uniform, &sorted, &contig};

    // one-time correctness check on the uniform pattern (golden matches r.pos)
    CK(cudaMemcpy(d_pos, uniform.data(), nq * 4, cudaMemcpyHostToDevice));
    cf_access_async(view, d_pos, d_out, nq, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> chk(nq);
    CK(cudaMemcpy(chk.data(), d_out, nq * 4, cudaMemcpyDeviceToHost));
    bool okc = std::equal(chk.begin(), chk.end(), r.golden.begin());

    auto time_kernel = [&](auto launch) -> double {
      std::vector<double> ts;
      for (int i = 0; i < WARM + REPS; ++i) {
        cudaEventRecord(a);
        launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (i >= WARM) ts.push_back((double)ms);
      }
      return median(ts) * 1e6 / (double)nq;  // ns per access
    };

    std::printf("  %5u %5u %8u %9.2f%s\n", r.vocab, r.levels, r.token_bytes, cf_index_mb(r) * 8e6 / r.n,
                okc ? "  [access==golden ✓]" : "  [MISMATCH]");
    for (int pi = 0; pi < 3; ++pi) {
      CK(cudaMemcpy(d_pos, pats[pi]->data(), nq * 4, cudaMemcpyHostToDevice));
      double g_ns = time_kernel([&] {
        if (r.token_bytes == 1)
          gather_kernel<uint8_t><<<blocks, threads>>>((const uint8_t *)d_raw, d_pos, d_out, nq);
        else if (r.token_bytes == 2)
          gather_kernel<uint16_t><<<blocks, threads>>>((const uint16_t *)d_raw, d_pos, d_out, nq);
        else
          gather_kernel<uint32_t><<<blocks, threads>>>((const uint32_t *)d_raw, d_pos, d_out, nq);
      });
      double w_ns = time_kernel([&] { cf_access_async(view, d_pos, d_out, nq, nullptr); });
      std::printf("  %5s %5s %8s %9s  %-11s %9.2f %9.2f %7.1f×\n", "", "", "", "", pat_name[pi], g_ns, w_ns,
                  w_ns / g_ns);
    }

    cudaFree(d_words);
    cudaFree(d_sb);
    cudaFree(d_zeros);
    cudaFree(d_raw);
    cudaFree(d_pos);
    cudaFree(d_out);
  }
  std::printf("\n=> ×cost is wavelet-access ÷ raw-gather: the price of keeping the data addressable AND searchable\n");
  std::printf("   (rank/select/FM-search) in the same footprint. Raw gather has no rank, no search, no entropy\n");
  std::printf("   coding, and its B/token is fixed by the type width; the wavelet's is ~the vocabulary's bits.\n");
  return 0;
}
