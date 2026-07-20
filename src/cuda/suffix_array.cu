// suffix_array.cu — build the suffix array on the GPU so the FM-index *construction* is resident too (M5, the
// last host straggler). Prefix-doubling, exactly the CPU algorithm parallelised: each round forms a 64-bit
// composite key = (rank[i] << 32) | (rank[i+k] + 1), radix-sorts the suffixes by it (thrust::sort_by_key), and
// re-ranks by adjacent-key differences (a scan) — O(n log n) sorts on the device. Faithful to
// warp_compress.gpu_suffix. The SA of a sentinel-terminated string is unique, so the result is bit-identical to
// the CPU suffix array (the verification gate).

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdint>
#include <vector>

using u64 = unsigned long long;

// key[i] = (rank[i] << 32) | (rank[i+k] + 1); past-the-end second half = 0 so it sorts first.
struct KeyFn {
  const int *rank;
  int n, k;
  __host__ __device__ u64 operator()(int i) const {
    u64 hi = (u64)(unsigned)rank[i];
    u64 lo = (i + k < n) ? (u64)(unsigned)(rank[i + k] + 1) : 0ULL;
    return (hi << 32) | lo;
  }
};

// new-rank boundary flag: 1 where the sorted key differs from its predecessor (a new equivalence class starts).
struct NeqFn {
  const u64 *g;
  __host__ __device__ int operator()(int j) const { return g[j] != g[j - 1] ? 1 : 0; }
};

// Build the suffix array of `s` on the GPU. `out` receives the n suffix start positions in sorted order.
void cf_gpu_suffix_array(const std::vector<int> &s, std::vector<int> &out) {
  int n = (int)s.size();
  thrust::device_vector<int> rank(s.begin(), s.end());   // rank_0 = the symbols themselves
  thrust::device_vector<int> sa(n), flag(n), pos(n);
  thrust::device_vector<u64> key(n), gkey(n);
  thrust::sequence(sa.begin(), sa.end());

  auto cnt0 = thrust::counting_iterator<int>(0);
  auto cnt1 = thrust::counting_iterator<int>(1);
  for (int k = 1; k < n; k <<= 1) {
    KeyFn kf{thrust::raw_pointer_cast(rank.data()), n, k};
    thrust::transform(cnt0, cnt0 + n, key.begin(), kf);                 // key per suffix start
    thrust::gather(sa.begin(), sa.end(), key.begin(), gkey.begin());    // gkey[j] = key[sa[j]]
    thrust::sort_by_key(gkey.begin(), gkey.end(), sa.begin());          // radix sort suffixes by key
    thrust::fill(flag.begin(), flag.begin() + 1, 0);
    NeqFn nf{thrust::raw_pointer_cast(gkey.data())};
    thrust::transform(cnt1, cnt1 + (n - 1), flag.begin() + 1, nf);      // class boundaries in sorted order
    thrust::inclusive_scan(flag.begin(), flag.end(), pos.begin());      // pos[j] = new rank of sa[j]
    thrust::scatter(pos.begin(), pos.end(), sa.begin(), rank.begin());  // rank[sa[j]] = pos[j]
    if ((int)pos[n - 1] == n - 1) break;                               // all distinct -> done
  }
  out.resize(n);
  thrust::copy(sa.begin(), sa.end(), out.begin());
}
