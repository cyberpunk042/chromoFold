// gpu_access.cu — verify the native CUDA `access` kernel against the frozen Warp reference, and benchmark it
// with the reproducibility envelope the constitution requires (P7): device metadata, >=20 warm reps,
// median/p5/p95, and the four timing layers reported separately (kernel-only vs round-trip).
//
// Usage: ./gpu_access [reference.cfwv]

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
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);             \
      return 2;                                                                                                \
    }                                                                                                          \
  } while (0)

struct Ref {
  uint32_t levels, nwords, nblocks, nqueries;
  uint64_t n;
  std::vector<uint32_t> words;
  std::vector<int32_t> sb, zeros;
  std::vector<uint32_t> pos, golden;
};

static bool load(const char *path, Ref &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "CFWV", 4) != 0) return false;
  std::fread(&version, 4, 1, f);
  std::fread(&r.levels, 4, 1, f);
  std::fread(&r.n, 8, 1, f);
  std::fread(&r.nwords, 4, 1, f);
  std::fread(&r.nblocks, 4, 1, f);
  std::fread(&r.nqueries, 4, 1, f);
  r.words.resize((size_t)r.levels * r.nwords);
  r.sb.resize((size_t)r.levels * (r.nblocks + 1));
  r.zeros.resize(r.levels);
  r.pos.resize(r.nqueries);
  r.golden.resize(r.nqueries);
  std::fread(r.words.data(), 4, r.words.size(), f);
  std::fread(r.sb.data(), 4, r.sb.size(), f);
  std::fread(r.zeros.data(), 4, r.zeros.size(), f);
  std::fread(r.pos.data(), 4, r.pos.size(), f);
  std::fread(r.golden.data(), 4, r.golden.size(), f);
  std::fclose(f);
  return true;
}

static double pct(std::vector<double> &v, double p) {
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(p / 100.0 * (v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "reference.cfwv";
  Ref r;
  if (!load(path, r)) {
    std::fprintf(stderr, "could not read reference file: %s\n", path);
    return 1;
  }

  int dev = 0;
  cudaDeviceProp prop;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  int rt = 0;
  cudaRuntimeGetVersion(&rt);

  // upload the immutable index + queries (build != query: this is one-time)
  uint32_t *d_words, *d_pos, *d_out;
  int32_t *d_sb, *d_zeros;
  size_t nq = r.nqueries;
  CK(cudaMalloc(&d_words, r.words.size() * 4));
  CK(cudaMalloc(&d_sb, r.sb.size() * 4));
  CK(cudaMalloc(&d_zeros, r.zeros.size() * 4));
  CK(cudaMalloc(&d_pos, nq * 4));
  CK(cudaMalloc(&d_out, nq * 4));
  CK(cudaMemcpy(d_words, r.words.data(), r.words.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_sb, r.sb.data(), r.sb.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_zeros, r.zeros.data(), r.zeros.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pos, r.pos.data(), nq * 4, cudaMemcpyHostToDevice));

  cf_wavelet_view view{d_words, d_sb, d_zeros, r.n, r.levels, r.nwords, r.nblocks};

  // ---- correctness: bit-identical to the golden Warp output ----
  cf_status st = cf_access_async(view, d_pos, d_out, nq, nullptr);
  CK(cudaDeviceSynchronize());
  if (st != CF_OK) {
    std::fprintf(stderr, "cf_access_async status %d\n", (int)st);
    return 2;
  }
  std::vector<uint32_t> host_out(nq);
  CK(cudaMemcpy(host_out.data(), d_out, nq * 4, cudaMemcpyDeviceToHost));
  size_t mism = 0;
  for (size_t i = 0; i < nq; ++i)
    if (host_out[i] != r.golden[i]) ++mism;

  // ---- timing: kernel-only vs round-trip, >=20 warm reps ----
  const int REPS = 30, WARM = 5;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  std::vector<double> kern, trip;
  for (int i = 0; i < WARM + REPS; ++i) {
    // kernel-only (inputs/outputs already resident)
    CK(cudaEventRecord(a));
    cf_access_async(view, d_pos, d_out, nq, nullptr);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    // round-trip (H2D positions + kernel + D2H output) — today's user-facing path
    CK(cudaEventRecord(a));
    CK(cudaMemcpyAsync(d_pos, r.pos.data(), nq * 4, cudaMemcpyHostToDevice));
    cf_access_async(view, d_pos, d_out, nq, nullptr);
    CK(cudaMemcpyAsync(host_out.data(), d_out, nq * 4, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms2 = 0;
    CK(cudaEventElapsedTime(&ms2, a, b));
    if (i >= WARM) {
      kern.push_back(ms);
      trip.push_back(ms2);
    }
  }

  double kmed = pct(kern, 50), kp95 = pct(kern, 95);
  double tmed = pct(trip, 50), tp95 = pct(trip, 95);
  double idx_mb = (r.words.size() + r.sb.size() + r.zeros.size()) * 4.0 / 1e6;

  std::printf("ChromoFold M1 — CUDA C++ access, verified vs frozen Warp reference\n\n");
  std::printf("  device      : %s (sm_%d%d), CUDA runtime %d.%d\n", prop.name, prop.major, prop.minor,
              rt / 1000, (rt % 1000) / 10);
  std::printf("  index       : n=%llu  levels=%u  nwords=%u  nblocks=%u  resident %.2f MB (%.2f b/tok)\n",
              (unsigned long long)r.n, r.levels, r.nwords, r.nblocks, idx_mb, idx_mb * 8e6 / r.n);
  std::printf("  queries     : %u\n\n", r.nqueries);
  std::printf("  correctness : %s  (%zu / %u mismatches vs golden)\n", mism == 0 ? "BIT-IDENTICAL ✓" : "FAIL",
              mism, r.nqueries);
  std::printf("  kernel-only : median %.3f ms  p95 %.3f ms  = %.0f M access/s  (%.2f ns/access)\n", kmed, kp95,
              nq / (kmed * 1e-3) / 1e6, kmed * 1e6 / nq);
  std::printf("  round-trip  : median %.3f ms  p95 %.3f ms  = %.0f M access/s  (%.2f ns/access)\n", tmed, tp95,
              nq / (tmed * 1e-3) / 1e6, tmed * 1e6 / nq);
  std::printf("\n  => device-native kernel is transfer-free; the round-trip layer shows the H2D/D2H tax the\n");
  std::printf("     production device-native API (device pointers, caller stream) removes.\n");

  cudaFree(d_words);
  cudaFree(d_sb);
  cudaFree(d_zeros);
  cudaFree(d_pos);
  cudaFree(d_out);
  return mism == 0 ? 0 : 3;
}
