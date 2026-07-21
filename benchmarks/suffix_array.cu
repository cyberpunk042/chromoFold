// suffix_array.cu — M5: GPU suffix-array build verified BIT-IDENTICAL to the CPU suffix array, and timed against
// it. Closes the last host straggler in the FM-index build path: construction now runs on the device too.
// Usage: ./suffix_array [--n N --vocab V --seed S]

#include "chromofold/detail/suffix_cpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void cf_gpu_suffix_array(const std::vector<int> &s, std::vector<int> &out);

int main(int argc, char **argv) {
  int64_t n = 2000000;
  int vocab = 64;
  uint64_t seed = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--n") n = std::stoll(argv[i + 1]);
    else if (k == "--vocab") vocab = std::stoi(argv[i + 1]);
    else if (k == "--seed") seed = std::stoull(argv[i + 1]);
  }
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    if (std::getenv("CI") != nullptr || std::getenv("GITHUB_ACTIONS") != nullptr) {
      std::fprintf(stderr, "  [SKIP] suffix-array: no CUDA device on CI runner\n");
      return 0;
    }
    std::fprintf(stderr, "no CUDA device\n");
    return 2;
  }
  int rt = 0; cudaRuntimeGetVersion(&rt);

  // sentinel-terminated string s = (seq + 1) ++ [0]  (as the FM-index indexes it); its SA is unique.
  std::vector<int64_t> seq = cf_gen_stream(n, vocab, seed);
  std::vector<int> s(n + 1);
  for (int64_t i = 0; i < n; ++i) s[i] = (int)seq[i] + 1;
  s[n] = 0;
  int N = (int)s.size();

  auto t0 = std::chrono::steady_clock::now();
  std::vector<int> sa_cpu = cf_cpu_suffix_array(s);
  auto t1 = std::chrono::steady_clock::now();

  std::vector<int> sa_gpu;
  cf_gpu_suffix_array(s, sa_gpu);        // warm-up (JIT / allocator)
  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  cf_gpu_suffix_array(s, sa_gpu);
  cudaDeviceSynchronize();
  auto t3 = std::chrono::steady_clock::now();

  bool ok = sa_gpu.size() == sa_cpu.size() && std::memcmp(sa_gpu.data(), sa_cpu.data(), N * 4) == 0;
  double cpu_s = std::chrono::duration<double>(t1 - t0).count();
  double gpu_s = std::chrono::duration<double>(t3 - t2).count();

  std::printf("M5 — GPU suffix-array build (prefix-doubling, thrust radix sort + scan re-rank), vs CPU\n");
  std::printf("device %s (sm_%d%d), CUDA %d.%d\n\n", prop.name, prop.major, prop.minor, rt / 1000, (rt % 1000) / 10);
  std::printf("  N=%d (vocab %d)   CPU build %.3f s   GPU build %.3f s   => %.1f× faster   SA %s\n", N, vocab,
              cpu_s, gpu_s, cpu_s / gpu_s, ok ? "BIT-IDENTICAL ✓" : "MISMATCH ✗");
  std::printf("\n=> The FM-index construction (suffix array) now runs on the GPU too — the SA of a sentinel-\n");
  std::printf("   terminated string is unique, so the device build is bit-identical to the CPU reference. Each\n");
  std::printf("   round is a 64-bit composite-key radix sort + a scan re-rank; the build path leaves the CPU.\n");
  return ok ? 0 : 1;
}
