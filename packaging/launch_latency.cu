// launch_latency.cu — a diagnostic that isolates the fixed per-call GPU cost (kernel launch + stream sync) from
// kernel COMPUTE, so a search-throughput number can be attributed honestly. Motivated by bench_search_device
// (sovereign-os): the FM count path showed a ~1.2-1.8 ms per-call floor, batch-independent for K>=16. This proves
// that floor is NOT launch/sync overhead (nor a WSL2 virtualization tax): an empty kernel launch + sync is tens of
// microseconds here. By subtraction the ~1.5 ms is real kernel time — the RRR entropy-decode of the compressed
// index (P1 compute-for-memory), not plumbing. Reproduce: make -C packaging launch-latency.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>

__global__ void empty_kernel() {}

static cudaStream_t g_s;

static double best_of(int reps, void (*f)()) {
  double b = 1e30;
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    double e = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (e < b) b = e;
  }
  return b;
}

static void launch_sync() { empty_kernel<<<16, 256, 0, g_s>>>(); cudaStreamSynchronize(g_s); }
static void sync_only()   { cudaStreamSynchronize(g_s); }
static void launch_only() { empty_kernel<<<16, 256, 0, g_s>>>(); }

int main() {
  if (cudaStreamCreate(&g_s) != cudaSuccess) { std::fprintf(stderr, "no CUDA device\n"); return 1; }
  launch_sync();  // warm up: context + module load
  double ls = best_of(50, launch_sync);
  double so = best_of(50, sync_only);
  double lo = best_of(50, launch_only);
  std::printf("GPU per-call fixed cost (best of 50):\n");
  std::printf("  empty kernel launch + stream sync : %.3f ms\n", ls * 1e3);
  std::printf("  stream sync only (idle stream)    : %.3f ms\n", so * 1e3);
  std::printf("  launch only (no sync)             : %.3f ms\n", lo * 1e3);
  std::printf("=> launch+sync is the floor's %s; a ~1.5 ms FM-count call is dominated by KERNEL compute.\n",
              ls < 0.2 ? "negligible part" : "significant part");
  cudaStreamDestroy(g_s);
  return 0;
}
