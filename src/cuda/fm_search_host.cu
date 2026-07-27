// fm_search_host.cu — host-pointer convenience layer over the device-native FM-index search (chromofold_search.h).
// Hides all CUDA device marshalling inside the engine so a caller with a .cffm blob + host query arrays (e.g. a
// Rust `-sys` FFI) can count/locate without linking cudart or doing cudaMalloc/Memcpy itself. Build the device
// index ONCE (P9 build != query); each query uploads the small pattern arrays, runs the verified device kernels,
// downloads the results. NOT included by a TU that already has the detail structs (single-definition rule) — this
// IS the engine, so it includes the detail header for cf_fm_view + cf_fm_count/ranges/locate_async.
#include "chromofold/chromofold.h"
#include "chromofold/detail/fm_search_device.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

struct cf_fm_host_index {
    cf_fm_view v;
    std::vector<void*> allocs;   // every device buffer owned by this index, freed at destroy
};

extern "C" void cf_fm_host_free(cf_fm_host_index* ix);  // fwd decl (header not included: single-definition rule)

namespace {

struct Reader {
    const uint8_t* p; const uint8_t* end; bool ok = true;
    void raw(void* dst, std::size_t n) { if (!ok || static_cast<std::size_t>(end - p) < n) { ok = false; return; }
        std::memcpy(dst, p, n); p += n; }
    template <class T> void arr(std::vector<T>& out, std::size_t count) { out.resize(count); raw(out.data(), count * sizeof(T)); }
};

template <class T> T* upload(std::vector<void*>& allocs, const std::vector<T>& h, bool& ok) {
    void* d = nullptr;
    if (cudaMalloc(&d, std::max<std::size_t>(h.size(), 1) * sizeof(T)) != cudaSuccess) { ok = false; return nullptr; }
    allocs.push_back(d);
    if (!h.empty() && cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    return static_cast<T*>(d);
}

// query scratch: upload host int32 arrays, run `fn`, download outputs. Frees its own scratch. Returns fn's status.
struct Scratch { std::vector<void*> a; ~Scratch() { for (void* p : a) cudaFree(p); }
    int32_t* up_i32(const int32_t* h, std::size_t n) { void* d = nullptr; if (cudaMalloc(&d, n * 4) != cudaSuccess) return nullptr;
        a.push_back(d); if (cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr; return static_cast<int32_t*>(d); }
    void* dev(std::size_t bytes) { void* d = nullptr; if (cudaMalloc(&d, bytes) != cudaSuccess) return nullptr; a.push_back(d); return d; } };

std::size_t flat_len(const int32_t* pstart, const int32_t* plen, std::uint32_t npat) {
    std::size_t m = 0;
    for (std::uint32_t i = 0; i < npat; ++i) m = std::max(m, static_cast<std::size_t>(pstart[i]) + static_cast<std::size_t>(plen[i]));
    return m;
}

}  // namespace

extern "C" cf_status cf_fm_host_load(const uint8_t* cffm, size_t nbytes, cf_fm_host_index** out) {
    if (cffm == nullptr || out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    *out = nullptr;
    Reader R{cffm, cffm + nbytes};
    char magic[4]; R.raw(magic, 4);
    std::uint32_t version = 0; R.raw(&version, 4);
    std::uint64_t n = 0; R.raw(&n, 8);
    std::uint32_t bits, vocab, sigma, nblocks, nsb, cwords, na, owords, sa_sample, mwords_len, msb_len, nsval, npat, patflat, nloc;
    R.raw(&bits, 4); R.raw(&vocab, 4); R.raw(&sigma, 4); R.raw(&nblocks, 4); R.raw(&nsb, 4); R.raw(&cwords, 4);
    R.raw(&na, 4); R.raw(&owords, 4); R.raw(&sa_sample, 4); R.raw(&mwords_len, 4); R.raw(&msb_len, 4);
    R.raw(&nsval, 4); R.raw(&npat, 4); R.raw(&patflat, 4); R.raw(&nloc, 4);
    std::uint64_t rrr_bytes = 0; R.raw(&rrr_bytes, 8);
    if (!R.ok || std::memcmp(magic, "CFFM", 4) != 0 || version != 1) return CF_ERR_INVALID_ARGUMENT;

    const std::uint32_t nsb1 = nsb + 1;
    std::vector<std::uint32_t> classes, offsets, mwords;
    std::vector<std::int32_t> rank_a, off_a, offbase, zeros, C, msb, sval;
    std::vector<std::uint16_t> rank_d, off_d;
    R.arr(classes, static_cast<std::size_t>(bits) * cwords); R.arr(offsets, owords);
    R.arr(rank_a, static_cast<std::size_t>(bits) * na);     R.arr(rank_d, static_cast<std::size_t>(bits) * nsb1);
    R.arr(off_a, static_cast<std::size_t>(bits) * na);      R.arr(off_d, static_cast<std::size_t>(bits) * nsb1);
    R.arr(offbase, bits); R.arr(zeros, bits); R.arr(C, sigma);
    R.arr(mwords, mwords_len); R.arr(msb, msb_len); R.arr(sval, nsval);
    if (!R.ok) return CF_ERR_INVALID_ARGUMENT;  // truncated blob

    // constant tables (offset bit-width per class + Pascal), rebuilt on host exactly like the benchmark/fixture.
    std::vector<int> binom(256, 0), width(16, 0);
    for (int nn = 0; nn < 16; ++nn) { binom[nn * 16] = 1; for (int kk = 1; kk <= nn; ++kk)
        binom[nn * 16 + kk] = binom[(nn - 1) * 16 + kk - 1] + binom[(nn - 1) * 16 + kk]; }
    for (int k = 0; k < 16; ++k) { int c = binom[15 * 16 + k], w = 0; while ((1 << w) < c) ++w; width[k] = c > 1 ? w : 0; }

    cf_fm_host_index* ix = new cf_fm_host_index();
    bool ok = true;
    auto& al = ix->allocs;
    ix->v.w.classes = upload(al, classes, ok); ix->v.w.offsets = upload(al, offsets, ok);
    ix->v.w.rank_a = upload(al, rank_a, ok);   ix->v.w.rank_d = upload(al, rank_d, ok);
    ix->v.w.off_a = upload(al, off_a, ok);     ix->v.w.off_d = upload(al, off_d, ok);
    ix->v.w.offbase = upload(al, offbase, ok); ix->v.w.zeros = upload(al, zeros, ok);
    ix->v.w.width = upload(al, width, ok);     ix->v.w.binom = upload(al, binom, ok);
    ix->v.w.bits = static_cast<int>(bits); ix->v.w.cwords = static_cast<int>(cwords);
    ix->v.w.nsb = static_cast<int>(nsb);   ix->v.w.na = static_cast<int>(na);
    ix->v.C = upload(al, C, ok); ix->v.mwords = upload(al, mwords, ok); ix->v.msb = upload(al, msb, ok); ix->v.sval = upload(al, sval, ok);
    ix->v.sigma = static_cast<int>(sigma); ix->v.n = static_cast<int>(n); ix->v.sa_sample = static_cast<int>(sa_sample);
    if (!ok) { cf_fm_host_free(ix); return CF_ERR_CUDA; }
    *out = ix;
    return CF_OK;
}

extern "C" cf_status cf_fm_host_count(const cf_fm_host_index* ix, const int32_t* pat, const int32_t* pstart,
                                      const int32_t* plen, uint32_t npat, uint32_t* counts_out) {
    if (ix == nullptr || pat == nullptr || pstart == nullptr || plen == nullptr || counts_out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    if (npat == 0) return CF_OK;
    Scratch s;
    int32_t* d_pat = s.up_i32(pat, flat_len(pstart, plen, npat));
    int32_t* d_ps = s.up_i32(pstart, npat); int32_t* d_pl = s.up_i32(plen, npat);
    uint32_t* d_cnt = static_cast<uint32_t*>(s.dev(static_cast<std::size_t>(npat) * 4));
    if (!d_pat || !d_ps || !d_pl || !d_cnt) return CF_ERR_CUDA;
    cf_status st = cf_fm_count_async(ix->v, d_pat, d_ps, d_pl, d_cnt, npat, nullptr);
    if (st != CF_OK) return st;
    if (cudaDeviceSynchronize() != cudaSuccess) return CF_ERR_CUDA;
    return cudaMemcpy(counts_out, d_cnt, static_cast<std::size_t>(npat) * 4, cudaMemcpyDeviceToHost) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_fm_host_ranges(const cf_fm_host_index* ix, const int32_t* pat, const int32_t* pstart,
                                       const int32_t* plen, uint32_t npat, int32_t* lo_out, int32_t* hi_out) {
    if (ix == nullptr || pat == nullptr || pstart == nullptr || plen == nullptr || lo_out == nullptr || hi_out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    if (npat == 0) return CF_OK;
    Scratch s;
    int32_t* d_pat = s.up_i32(pat, flat_len(pstart, plen, npat));
    int32_t* d_ps = s.up_i32(pstart, npat); int32_t* d_pl = s.up_i32(plen, npat);
    int32_t* d_lo = static_cast<int32_t*>(s.dev(static_cast<std::size_t>(npat) * 4));
    int32_t* d_hi = static_cast<int32_t*>(s.dev(static_cast<std::size_t>(npat) * 4));
    if (!d_pat || !d_ps || !d_pl || !d_lo || !d_hi) return CF_ERR_CUDA;
    cf_status st = cf_fm_ranges_async(ix->v, d_pat, d_ps, d_pl, d_lo, d_hi, npat, nullptr);
    if (st != CF_OK) return st;
    if (cudaDeviceSynchronize() != cudaSuccess) return CF_ERR_CUDA;
    if (cudaMemcpy(lo_out, d_lo, static_cast<std::size_t>(npat) * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return CF_ERR_CUDA;
    return cudaMemcpy(hi_out, d_hi, static_cast<std::size_t>(npat) * 4, cudaMemcpyDeviceToHost) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_fm_host_locate(const cf_fm_host_index* ix, const int32_t* rows, uint32_t nocc, int32_t* pos_out) {
    if (ix == nullptr || rows == nullptr || pos_out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    if (nocc == 0) return CF_OK;
    Scratch s;
    int32_t* d_r = s.up_i32(rows, nocc);
    int32_t* d_pos = static_cast<int32_t*>(s.dev(static_cast<std::size_t>(nocc) * 4));
    if (!d_r || !d_pos) return CF_ERR_CUDA;
    cf_status st = cf_fm_locate_async(ix->v, d_r, d_pos, nocc, nullptr);
    if (st != CF_OK) return st;
    if (cudaDeviceSynchronize() != cudaSuccess) return CF_ERR_CUDA;
    return cudaMemcpy(pos_out, d_pos, static_cast<std::size_t>(nocc) * 4, cudaMemcpyDeviceToHost) == cudaSuccess ? CF_OK : CF_ERR_CUDA;
}

extern "C" void cf_fm_host_free(cf_fm_host_index* ix) {
    if (ix == nullptr) return;
    for (void* p : ix->allocs) cudaFree(p);
    delete ix;
}

// Device-native path: hand back the resident device view so the caller can drive cf_fm_*_async directly.
// The view's pointers stay owned by `ix` (valid until cf_fm_host_free); the caller copies the POD by value.
extern "C" cf_status cf_fm_host_view(const cf_fm_host_index* ix, cf_fm_view* out) {
    if (ix == nullptr || out == nullptr) return CF_ERR_INVALID_ARGUMENT;
    *out = ix->v;
    return CF_OK;
}
