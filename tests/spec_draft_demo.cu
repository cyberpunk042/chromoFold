// Searchable-thesis demonstration (see docs/SEARCHABLE_WORKLOADS.md): use the verified GPU FM-index as a
// COMPRESSED, GPU-resident n-gram / prompt-lookup speculative-draft oracle, and compare to an UNCOMPRESSED hash
// n-gram table on draft hit-rate + memory. The point: same predictions (same n-gram statistics) at a fraction of
// the memory, one index for ANY context length — a capability llama's quantized KV cannot provide.
//   build: nvcc ... tests/spec_draft_demo.cu src/cuda/fm_search.cu -o spec_draft_demo
//   run:   spec_draft_demo corpus.toks index.cffm [L]
#include "chromofold/detail/fm_search_device.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA %s: %s\n", #x, cudaGetErrorString(e_)); return 2; } } while (0)

template <class T> static T *upload(const std::vector<T> &h) {
    T *d = nullptr; if (cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)) != cudaSuccess) return nullptr;
    if (!h.empty()) cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

struct Fm {
    uint64_t n = 0, rrr_bytes = 0;
    uint32_t bits = 0, vocab = 0, sigma = 0, nblocks = 0, nsb = 0, cwords = 0, na = 0, owords = 0, sa_sample = 0;
    uint32_t mwords_len = 0, msb_len = 0, nsval = 0, npat = 0, patflat = 0, nloc = 0;
    std::vector<uint32_t> classes, offsets, mwords, count_golden;
    std::vector<int32_t> rank_a, off_a, offbase, zeros, C, msb, sval, pat, pstart, plen, locoff, locpos;
    std::vector<uint16_t> rank_d, off_d;
};

static bool load_fm(const char *path, Fm &r) {
    FILE *f = std::fopen(path, "rb"); if (!f) return false;
    char magic[4]; uint32_t version = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFFM", 4) == 0;
    auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
    rd(&version, 4); rd(&r.n, 8);
    rd(&r.bits, 4); rd(&r.vocab, 4); rd(&r.sigma, 4); rd(&r.nblocks, 4); rd(&r.nsb, 4); rd(&r.cwords, 4);
    rd(&r.na, 4); rd(&r.owords, 4); rd(&r.sa_sample, 4); rd(&r.mwords_len, 4); rd(&r.msb_len, 4);
    rd(&r.nsval, 4); rd(&r.npat, 4); rd(&r.patflat, 4); rd(&r.nloc, 4); rd(&r.rrr_bytes, 8);
    if (!ok || version != 1) { std::fclose(f); return false; }
    const uint32_t nsb1 = r.nsb + 1;
    r.classes.resize((size_t) r.bits * r.cwords); r.offsets.resize(r.owords);
    r.rank_a.resize((size_t) r.bits * r.na); r.rank_d.resize((size_t) r.bits * nsb1);
    r.off_a.resize((size_t) r.bits * r.na); r.off_d.resize((size_t) r.bits * nsb1);
    r.offbase.resize(r.bits); r.zeros.resize(r.bits); r.C.resize(r.sigma);
    r.mwords.resize(r.mwords_len); r.msb.resize(r.msb_len); r.sval.resize(r.nsval);
    r.pat.resize(r.patflat); r.pstart.resize(r.npat); r.plen.resize(r.npat);
    r.count_golden.resize(r.npat); r.locoff.resize(r.npat + 1); r.locpos.resize(r.nloc);
    rd(r.classes.data(), r.classes.size() * 4); rd(r.offsets.data(), r.offsets.size() * 4);
    rd(r.rank_a.data(), r.rank_a.size() * 4);   rd(r.rank_d.data(), r.rank_d.size() * 2);
    rd(r.off_a.data(), r.off_a.size() * 4);     rd(r.off_d.data(), r.off_d.size() * 2);
    rd(r.offbase.data(), r.offbase.size() * 4); rd(r.zeros.data(), r.zeros.size() * 4);
    rd(r.C.data(), r.C.size() * 4); rd(r.mwords.data(), r.mwords.size() * 4);
    rd(r.msb.data(), r.msb.size() * 4); rd(r.sval.data(), r.sval.size() * 4);
    rd(r.pat.data(), r.pat.size() * 4); rd(r.pstart.data(), r.pstart.size() * 4);
    rd(r.plen.data(), r.plen.size() * 4); rd(r.count_golden.data(), r.count_golden.size() * 4);
    rd(r.locoff.data(), r.locoff.size() * 4); rd(r.locpos.data(), r.locpos.size() * 4);
    std::fclose(f); return ok;
}

static std::vector<int32_t> load_corpus(const char *path) {
    std::vector<int32_t> c; FILE *f = std::fopen(path, "rb"); if (!f) return c;
    int64_t n = 0; if (std::fread(&n, 8, 1, f) == 1 && n > 0) { c.resize((size_t) n); if (std::fread(c.data(), 4, c.size(), f) != c.size()) c.clear(); }
    std::fclose(f); return c;
}

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s corpus.toks index.cffm [L]\n", argv[0]); return 1; }
    const int L = argc > 3 ? std::atoi(argv[3]) : 4;
    std::vector<int32_t> corpus = load_corpus(argv[1]);
    Fm r;
    if (corpus.empty() || !load_fm(argv[2], r)) { std::fprintf(stderr, "load failed\n"); return 1; }
    const int n = (int) corpus.size(), V = (int) r.vocab;

    // FM view (identical to the verified fm_search path)
    std::vector<int> binom(256, 0), width(16, 0);
    for (int nn = 0; nn < 16; ++nn) { binom[nn * 16] = 1; for (int kk = 1; kk <= nn; ++kk) binom[nn * 16 + kk] = binom[(nn - 1) * 16 + kk - 1] + binom[(nn - 1) * 16 + kk]; }
    for (int k = 0; k < 16; ++k) { int cc = binom[15 * 16 + k], w = 0; while ((1 << w) < cc) ++w; width[k] = (cc > 1) ? w : 0; }
    cf_fm_view v;
    v.w.classes = upload(r.classes); v.w.offsets = upload(r.offsets);
    v.w.rank_a = upload(r.rank_a); v.w.rank_d = upload(r.rank_d);
    v.w.off_a = upload(r.off_a); v.w.off_d = upload(r.off_d);
    v.w.offbase = upload(r.offbase); v.w.zeros = upload(r.zeros);
    v.w.width = upload(width); v.w.binom = upload(binom);
    v.w.bits = (int) r.bits; v.w.cwords = (int) r.cwords; v.w.nsb = (int) r.nsb; v.w.na = (int) r.na;
    v.C = upload(r.C); v.mwords = upload(r.mwords); v.msb = upload(r.msb); v.sval = upload(r.sval);
    v.sigma = (int) r.sigma; v.n = (int) r.n; v.sa_sample = (int) r.sa_sample;

    // Queries: evenly spaced positions; each pattern = corpus[p-L..p-1] shifted +1 into the sentinel alphabet.
    const int M = 4000;
    std::vector<int> qp;
    for (int i = 0; i < M; ++i) { int p = L + (int) ((long) i * (n - L - 1) / M); if (p + 1 < n) qp.push_back(p); }
    const int Q = (int) qp.size();
    std::vector<int32_t> pat, pstart(Q), plen(Q, L);
    for (int i = 0; i < Q; ++i) { pstart[i] = (int) pat.size(); for (int j = 0; j < L; ++j) pat.push_back(corpus[qp[i] - L + j] + 1); }

    int32_t *d_pat = upload(pat), *d_ps = upload(pstart), *d_pl = upload(plen), *d_lo = nullptr, *d_hi = nullptr;
    CK(cudaMalloc(&d_lo, (size_t) Q * 4)); CK(cudaMalloc(&d_hi, (size_t) Q * 4));
    cf_fm_ranges_async(v, d_pat, d_ps, d_pl, d_lo, d_hi, Q, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<int32_t> lo(Q), hi(Q); CK(cudaMemcpy(lo.data(), d_lo, Q * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hi.data(), d_hi, Q * 4, cudaMemcpyDeviceToHost));
    std::vector<int32_t> rflat; std::vector<uint32_t> occoff(Q + 1, 0);
    for (int i = 0; i < Q; ++i) { for (int rr = lo[i]; rr < hi[i]; ++rr) rflat.push_back(rr); occoff[i + 1] = (uint32_t) rflat.size(); }
    std::vector<int32_t> pos(rflat.size());
    if (!rflat.empty()) {
        int32_t *d_r = upload(rflat), *d_pos = nullptr; CK(cudaMalloc(&d_pos, rflat.size() * 4));
        cf_fm_locate_async(v, d_r, d_pos, rflat.size(), nullptr); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pos.data(), d_pos, rflat.size() * 4, cudaMemcpyDeviceToHost));
    }

    // Correctness gate (M0): for the first queries, FM occurrence set must equal a brute-force corpus scan.
    int gate = std::min(Q, 40), gate_ok = 0;
    for (int i = 0; i < gate; ++i) {
        std::vector<int> fm(pos.begin() + occoff[i], pos.begin() + occoff[i + 1]);
        std::vector<int> bf;
        for (int t = 0; t + L <= n; ++t) { bool m = true; for (int j = 0; j < L; ++j) if (corpus[t + j] != corpus[qp[i] - L + j]) { m = false; break; } if (m) bf.push_back(t); }
        std::sort(fm.begin(), fm.end());
        if (fm == bf) ++gate_ok;
    }
    if (gate_ok != gate) { std::printf("CORRECTNESS GATE FAILED: %d/%d FM occurrence sets match brute force\n", gate_ok, gate); return 1; }

    // FM draft = argmax next-token over PRIOR occurrences; hit if it equals the true next token.
    auto argmax_next = [&](int qi, int self_pos) {
        std::vector<int> h(V, 0); int best = -1, bc = -1;
        for (uint32_t k = occoff[qi]; k < occoff[qi + 1]; ++k) { int q = pos[k]; if (q == self_pos || q + L >= n) continue; int nt = corpus[q + L]; if (++h[nt] > bc) { bc = h[nt]; best = nt; } }
        return best;
    };
    int fm_hit = 0, fm_have = 0;
    for (int i = 0; i < Q; ++i) { int d = argmax_next(i, qp[i] - L); if (d >= 0) { ++fm_have; if (d == corpus[qp[i]]) ++fm_hit; } }

    // Uncompressed hash n-gram baseline (order L): key = packed L-gram -> next-token histogram (built once).
    std::unordered_map<uint64_t, std::vector<uint32_t>> ng;
    ng.reserve((size_t) n);
    auto key = [&](int t) { uint64_t k = 0; for (int j = 0; j < L; ++j) k = k * (uint64_t) V + (uint64_t) corpus[t + j]; return k; };
    for (int t = 0; t + L < n; ++t) { auto &hv = ng[key(t)]; if (hv.empty()) hv.assign(V, 0); hv[corpus[t + L]]++; }
    int hash_hit = 0, hash_have = 0;
    for (int i = 0; i < Q; ++i) {
        auto it = ng.find(key(qp[i] - L)); if (it == ng.end()) continue;
        const int truth = corpus[qp[i]];
        int best = -1, bc = -1;
        for (int t = 0; t < V; ++t) {
            int cnt = (int) it->second[t] - (t == truth ? 1 : 0);  // exclude the query's OWN occurrence (fair vs FM)
            if (cnt > bc) { bc = cnt; best = t; }
        }
        if (bc <= 0) continue;  // no prior occurrence of this L-gram -> like FM's fm_have filter
        ++hash_have; if (best == truth) ++hash_hit;
    }

    // Memory. FM: the full GPU-resident searchable index. Hash: minimal packed (distinct L-grams x (L+1) bytes,
    // predicting only the argmax) — charitable to the baseline; and it needs a SEPARATE table per order L.
    const double fm_bytes = (double) (r.classes.size() * 4 + r.offsets.size() * 4 + r.rank_a.size() * 4 + r.rank_d.size() * 2
        + r.off_a.size() * 4 + r.off_d.size() * 2 + r.C.size() * 4 + r.mwords.size() * 4 + r.msb.size() * 4 + r.sval.size() * 4);
    const double hash_min_bytes = (double) ng.size() * (L + 1);        // charitable: argmax-only, no overhead
    const double hash_full_bytes = (double) ng.size() * (double) V * 4; // what the histogram map actually stores

    std::printf("Searchable spec-draft: FM-index (compressed, GPU) vs uncompressed hash n-gram (corpus=%d, vocab=%d, L=%d)\n", n, V, L);
    std::printf("  queries: %d  | FM occurrences located: %zu\n", Q, rflat.size());
    std::printf("  correctness gate: FM occurrence sets == brute force on %d/%d queries\n", gate_ok, gate);
    std::printf("  draft hit-rate   FM: %.3f (%d/%d)   hash: %.3f (%d/%d)\n",
                fm_have ? (double) fm_hit / fm_have : 0.0, fm_hit, fm_have,
                hash_have ? (double) hash_hit / hash_have : 0.0, hash_hit, hash_have);
    std::printf("  memory   FM-index (ONE index, ANY L): %.0f KB\n", fm_bytes / 1024.0);
    std::printf("           hash order-%d: %.0f KB (argmax-only, no overhead) .. %.0f KB (full histograms) | %zu distinct %d-grams\n",
                L, hash_min_bytes / 1024.0, hash_full_bytes / 1024.0, ng.size(), L);
    std::printf("  => FM vs charitable hash %.2fx; vs real histogram map %.2fx smaller. FM also gives locate + any-L.\n",
                fm_bytes / hash_min_bytes, hash_full_bytes / fm_bytes);
    return 0;
}
