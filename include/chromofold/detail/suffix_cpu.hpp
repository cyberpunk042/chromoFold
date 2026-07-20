// suffix_cpu.hpp — host reference for the suffix-array build: a deterministic structured stream generator + a
// CPU prefix-doubling suffix array. The CPU SA is the golden the GPU build (src/cuda/suffix_array.cu) must match
// bit-for-bit (the SA of a sentinel-terminated string is unique), and its build time is the honest baseline the
// GPU build is measured against. Header-only so the benchmark and any host tool share one definition.
#ifndef CHROMOFOLD_DETAIL_SUFFIX_CPU_HPP
#define CHROMOFOLD_DETAIL_SUFFIX_CPU_HPP

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

// A deterministic order-1 chain with runs (compressible; a realistic FM-index input). Returns seq in [0, vocab).
static inline std::vector<int64_t> cf_gen_stream(int64_t n, int vocab, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::uniform_int_distribution<int> Rv(0, vocab - 1);
  std::vector<int64_t> seq(n);
  seq[0] = 0;
  for (int64_t i = 1; i < n; ++i) {
    double u = U(rng);
    if (u < 0.60) seq[i] = seq[i - 1];
    else if (u < 0.80) seq[i] = (seq[i - 1] + 1) % vocab;
    else if (u < 0.90) seq[i] = (seq[i - 1] + vocab - 1) % vocab;
    else seq[i] = Rv(rng);
  }
  return seq;
}

// CPU suffix array by prefix doubling (O(n log^2 n)): the exact algorithm the GPU build parallelises.
static inline std::vector<int> cf_cpu_suffix_array(const std::vector<int> &s) {
  int n = (int)s.size();
  std::vector<int> sa(n), rank(n), tmp(n);
  for (int i = 0; i < n; ++i) { sa[i] = i; rank[i] = s[i]; }
  for (int k = 1;; k <<= 1) {
    auto key2 = [&](int i) { return i + k < n ? rank[i + k] : -1; };
    auto cmp = [&](int a, int b) {
      if (rank[a] != rank[b]) return rank[a] < rank[b];
      return key2(a) < key2(b);
    };
    std::sort(sa.begin(), sa.end(), cmp);
    tmp[sa[0]] = 0;
    for (int i = 1; i < n; ++i) tmp[sa[i]] = tmp[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
    for (int i = 0; i < n; ++i) rank[i] = tmp[i];
    if (rank[sa[n - 1]] == n - 1) break;
  }
  return sa;
}

#endif // CHROMOFOLD_DETAIL_SUFFIX_CPU_HPP
