// build_index.cpp — the native C++20 offline builder for the RRR-backed wavelet index (milestone M5). This is the
// "build ≠ query" half (constitution P9): a pure-CPU builder that takes a token stream, builds its suffix array →
// BWT → RRR-coded wavelet (every level an RRR bitvector with two-level superblock samples), and serialises the
// exact .cfrw v1 format the GPU query kernels consume — WITHOUT the Warp prototype. A scalar CPU access/rank
// backend doubles as the correctness oracle (it computes the golden and self-verifies that access reconstructs the
// BWT) and as an honest CPU baseline (not a Python loop). Verification: feed this builder's .cfrw to the existing
// build/rrr_wavelet GPU benchmark — GPU access/rank must be bit-identical to the CPU golden written here.
//
// Faithful CPU port of warp_compress.gpu_rrr.rrr_encode / _two_level and gpu_rrr_wavelet.RRRWaveletGPU. Build:
//   g++ -O3 -std=c++17 tools/build_index.cpp -o build/build_index
// Run:  build/build_index out.cfrw --n 2000000 --vocab 64 --queries 100000 --rank 100000

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static constexpr int T = 15;   // RRR block size (bits)
static constexpr int S = 64;   // blocks per superblock
static constexpr int K = 32;   // superblocks per anchor (two-level rank)

static int64_t BINOM[16][16];
static int WIDTH[16];

static void init_tables() {
  for (int n = 0; n < 16; ++n) {
    BINOM[n][0] = 1;
    for (int k = 1; k <= n; ++k) BINOM[n][k] = BINOM[n - 1][k - 1] + BINOM[n - 1][k];
  }
  for (int k = 0; k < 16; ++k) {
    int64_t c = BINOM[T][k];
    int w = 0;
    while ((1LL << w) < c) ++w;
    WIDTH[k] = (c > 1) ? w : 0;
  }
}

// ---- deterministic structured stream: an order-1 chain with runs, so its BWT bitplanes are skewed (RRR wins) --
static std::vector<int64_t> markov_stream(int64_t n, int vocab, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::uniform_int_distribution<int> Rv(0, vocab - 1);
  std::vector<int64_t> seq(n);
  seq[0] = 0;
  for (int64_t i = 1; i < n; ++i) {
    double u = U(rng);
    if (u < 0.60) seq[i] = seq[i - 1];                       // a run (highly compressible)
    else if (u < 0.80) seq[i] = (seq[i - 1] + 1) % vocab;
    else if (u < 0.90) seq[i] = (seq[i - 1] + vocab - 1) % vocab;
    else seq[i] = Rv(rng);
  }
  return seq;
}

// ---- suffix array by prefix doubling (O(n log^2 n)); offline build cost is acceptable (P9) ----
static std::vector<int> suffix_array(const std::vector<int> &s) {
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

// ---- RRR encode one 0/1 plane -> class stream (4b/block), variable-width offset stream, superblock samples ----
struct RrrPlane {
  std::vector<uint32_t> cpk, opk;
  std::vector<int32_t> sbrank, sboff;  // length nsb+1
  int nblocks = 0, nsb = 0, cwords = 0;
  int64_t class_bits = 0, offset_bits = 0;
  int zeros = 0;
};

static RrrPlane rrr_encode(const std::vector<uint8_t> &bits) {
  RrrPlane e;
  int64_t n = (int64_t)bits.size();
  int nblocks = (int)((n + T - 1) / T);
  e.nblocks = nblocks;
  e.cwords = (int)((nblocks * 4 + 31) / 32 + 1);
  e.cpk.assign(e.cwords, 0);

  std::vector<int> cls(nblocks);
  std::vector<int64_t> off(nblocks);
  std::vector<int64_t> ostart(nblocks + 1, 0);  // cumulative offset-bit position
  int64_t ones = 0;
  for (int blk = 0; blk < nblocks; ++blk) {
    int r = 0;
    int64_t o = 0;
    for (int c = 0; c < T; ++c) {
      int64_t idx = (int64_t)blk * T + c;
      int bit = (idx < n) ? bits[idx] : 0;
      if (bit) { ++r; o += BINOM[c][r]; }  // enumerative rank in the combinatorial number system
    }
    cls[blk] = r;
    off[blk] = o;
    ones += r;
    // pack the 4-bit class, LSB-first
    int64_t bp = (int64_t)blk * 4;
    e.cpk[bp >> 5] |= (uint32_t)(r & 15) << (bp & 31);
    ostart[blk + 1] = ostart[blk] + WIDTH[r];
  }
  e.class_bits = (int64_t)nblocks * 4;
  e.offset_bits = ostart[nblocks];
  e.zeros = (int)(n - ones);

  int64_t total = ostart[nblocks];
  int owords = (int)((total + 31) / 32 + 2);
  e.opk.assign(owords, 0);
  for (int blk = 0; blk < nblocks; ++blk) {
    int w = WIDTH[cls[blk]];
    if (!w) continue;
    int64_t start = ostart[blk];
    int64_t wi = start >> 5;
    int b = (int)(start & 31);
    uint64_t v = (uint64_t)off[blk];
    e.opk[wi] |= (uint32_t)((v << b) & 0xFFFFFFFFu);
    if (b + w > 32) e.opk[wi + 1] |= (uint32_t)(v >> (32 - b));
  }

  int nsb = (int)((nblocks + S - 1) / S);
  e.nsb = nsb;
  e.sbrank.assign(nsb + 1, 0);
  e.sboff.assign(nsb + 1, 0);
  std::vector<int64_t> cum_class(nblocks + 1, 0);
  for (int blk = 0; blk < nblocks; ++blk) cum_class[blk + 1] = cum_class[blk] + cls[blk];
  for (int i = 0; i <= nsb; ++i) {
    int sidx = std::min((int64_t)i * S, (int64_t)nblocks);
    e.sbrank[i] = (int32_t)cum_class[sidx];
    e.sboff[i] = (int32_t)ostart[sidx];
  }
  return e;
}

// two-level split of an int32 cumulative sample: int32 anchors every K + uint16 deltas (halves the sample table).
static void two_level(const std::vector<int32_t> &cum, std::vector<int32_t> &anchors, std::vector<uint16_t> &delta) {
  int m = (int)cum.size();
  int na = (m + K - 1) / K;
  anchors.assign(na, 0);
  delta.assign(m, 0);
  for (int a = 0; a < na; ++a) anchors[a] = cum[(int64_t)a * K];
  for (int i = 0; i < m; ++i) {
    int64_t d = (int64_t)cum[i] - anchors[i / K];
    delta[i] = (uint16_t)d;  // bounded < 2^16 for K<=64 (see gpu_rrr._two_level)
  }
}

// ---- the assembled RRR-wavelet index (mirrors cf_rrrw_view + the .cfrw payload) ----
struct RrrWavelet {
  int bits = 0, cwords = 0, nsb = 0, na = 0;
  std::vector<uint32_t> classes, offsets;
  std::vector<int32_t> rank_a, off_a, offbase, zeros;
  std::vector<uint16_t> rank_d, off_d;
  int64_t rrr_bytes = 0;
};

static RrrWavelet build_wavelet(const std::vector<int64_t> &bwt, int bits) {
  RrrWavelet w;
  w.bits = bits;
  int64_t n = (int64_t)bwt.size();
  std::vector<int64_t> cur = bwt;
  std::vector<int32_t> sbrank_all, sboff_all;  // (bits, nsb+1) row-major
  int64_t off_words = 0;
  int64_t bits_stored = 0;
  for (int lvl = 0; lvl < bits; ++lvl) {
    std::vector<uint8_t> plane(n);
    for (int64_t i = 0; i < n; ++i) plane[i] = (uint8_t)((cur[i] >> (bits - 1 - lvl)) & 1);
    RrrPlane e = rrr_encode(plane);
    if (lvl == 0) { w.cwords = e.cwords; w.nsb = e.nsb; }
    w.classes.insert(w.classes.end(), e.cpk.begin(), e.cpk.end());
    w.offsets.insert(w.offsets.end(), e.opk.begin(), e.opk.end());
    sbrank_all.insert(sbrank_all.end(), e.sbrank.begin(), e.sbrank.end());
    sboff_all.insert(sboff_all.end(), e.sboff.begin(), e.sboff.end());
    w.zeros.push_back(e.zeros);
    w.offbase.push_back((int32_t)(off_words * 32));
    off_words += (int64_t)e.opk.size();
    bits_stored += e.class_bits + e.offset_bits;
    // stable partition cur by the plane bit (zeros first, then ones) for the next level
    std::vector<int64_t> nxt;
    nxt.reserve(n);
    for (int64_t i = 0; i < n; ++i) if (!plane[i]) nxt.push_back(cur[i]);
    for (int64_t i = 0; i < n; ++i) if (plane[i]) nxt.push_back(cur[i]);
    cur.swap(nxt);
  }
  // two-level samples per level: rank_a/off_a are (bits, na), rank_d/off_d are (bits, nsb+1)
  int nsb1 = w.nsb + 1;
  w.na = (nsb1 + K - 1) / K;
  for (int lvl = 0; lvl < bits; ++lvl) {
    std::vector<int32_t> rc(sbrank_all.begin() + (int64_t)lvl * nsb1, sbrank_all.begin() + (int64_t)(lvl + 1) * nsb1);
    std::vector<int32_t> oc(sboff_all.begin() + (int64_t)lvl * nsb1, sboff_all.begin() + (int64_t)(lvl + 1) * nsb1);
    std::vector<int32_t> ra, oa;
    std::vector<uint16_t> rd, od;
    two_level(rc, ra, rd);
    two_level(oc, oa, od);
    w.rank_a.insert(w.rank_a.end(), ra.begin(), ra.end());
    w.rank_d.insert(w.rank_d.end(), rd.begin(), rd.end());
    w.off_a.insert(w.off_a.end(), oa.begin(), oa.end());
    w.off_d.insert(w.off_d.end(), od.begin(), od.end());
  }
  int64_t sb_bytes = (int64_t)w.rank_a.size() * 4 + w.rank_d.size() * 2 + w.off_a.size() * 4 + w.off_d.size() * 2;
  w.rrr_bytes = bits_stored / 8 + sb_bytes;
  return w;
}

// ---- CPU backend (the oracle + baseline): access/rank over the assembled RRR-wavelet, mirroring the device path ----
static inline int cpu_classat(const uint32_t *classes, int j) {
  int bp = j * 4;
  return (int)((classes[bp >> 5] >> (bp & 31)) & 15u);
}
static inline int cpu_readbits(const uint32_t *stream, int64_t bitpos, int width) {
  if (!width) return 0;
  int64_t wi = bitpos >> 5;
  int b = (int)(bitpos & 31);
  uint32_t val = stream[wi] >> b;
  if (b + width > 32) val |= stream[wi + 1] << (32 - b);
  uint32_t mask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
  return (int)(val & mask);
}
static inline uint32_t cpu_decode_word(int cl, int off) {
  uint32_t word = 0;
  int r = off, i = cl;
  while (i >= 1) {
    int c = T - 1;
    while (BINOM[c][i] > r) --c;
    word |= (1u << c);
    r -= (int)BINOM[c][i];
    --i;
  }
  return word;
}
static int cpu_rank1_lvl(const RrrWavelet &w, int lvl, int pos) {
  const uint32_t *classes = w.classes.data() + (int64_t)lvl * w.cwords;
  int nsb1 = w.nsb + 1;
  int blk = pos / T, b = pos % T, sbi = blk / S, a = sbi / K;
  int r = w.rank_a[(int64_t)lvl * w.na + a] + (int)w.rank_d[(int64_t)lvl * nsb1 + sbi];
  int obit = w.off_a[(int64_t)lvl * w.na + a] + (int)w.off_d[(int64_t)lvl * nsb1 + sbi];
  for (int j = sbi * S; j < blk; ++j) { int cl = cpu_classat(classes, j); r += cl; obit += WIDTH[cl]; }
  if (b > 0) {
    int cl = cpu_classat(classes, blk);
    int off = cpu_readbits(w.offsets.data(), (int64_t)w.offbase[lvl] + obit, WIDTH[cl]);
    uint32_t word = cpu_decode_word(cl, off);
    r += __builtin_popcount(word & ((1u << b) - 1u));
  }
  return r;
}
static uint32_t cpu_access(const RrrWavelet &w, int i) {
  int val = 0;
  for (int lvl = 0; lvl < w.bits; ++lvl) {
    int r0 = cpu_rank1_lvl(w, lvl, i), r1 = cpu_rank1_lvl(w, lvl, i + 1);
    if (r1 - r0 == 1) { val = (val << 1) | 1; i = w.zeros[lvl] + r0; }
    else { val = val << 1; i = i - r0; }
  }
  return (uint32_t)val;
}
static uint32_t cpu_rank(const RrrWavelet &w, int c, int i) {
  int p = 0, q = i;
  for (int lvl = 0; lvl < w.bits; ++lvl) {
    int bitc = (c >> (w.bits - 1 - lvl)) & 1;
    int rp = cpu_rank1_lvl(w, lvl, p), rq = cpu_rank1_lvl(w, lvl, q);
    if (bitc) { p = w.zeros[lvl] + rp; q = w.zeros[lvl] + rq; }
    else { p = p - rp; q = q - rq; }
  }
  return (uint32_t)(q - p);
}

static void wr(FILE *f, const void *p, size_t bytes) { std::fwrite(p, 1, bytes, f); }

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s out.cfrw [--n N --vocab V --queries Q --rank R --seed S]\n", argv[0]); return 1; }
  const char *out = argv[1];
  int64_t n = 2000000;
  int vocab = 64, nq = 100000, nr = 100000;
  uint64_t seed = 0;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--n") n = std::stoll(argv[i + 1]);
    else if (k == "--vocab") vocab = std::stoi(argv[i + 1]);
    else if (k == "--queries") nq = std::stoi(argv[i + 1]);
    else if (k == "--rank") nr = std::stoi(argv[i + 1]);
    else if (k == "--seed") seed = std::stoull(argv[i + 1]);
  }
  init_tables();
  auto t0 = std::chrono::steady_clock::now();

  std::vector<int64_t> seq = markov_stream(n, vocab, seed);
  // BWT of s = (seq+1) ++ [0]: index the sentinel'd stream, like the FM-index.
  std::vector<int> s(n + 1);
  for (int64_t i = 0; i < n; ++i) s[i] = (int)seq[i] + 1;
  s[n] = 0;
  int64_t N = n + 1;
  std::vector<int> sa = suffix_array(s);
  std::vector<int64_t> bwt(N);
  for (int64_t i = 0; i < N; ++i) bwt[i] = s[(int64_t)(sa[i] - 1 + N) % N];
  int sigma = *std::max_element(bwt.begin(), bwt.end()) + 1;
  int bits = 1;
  while ((1 << bits) < sigma) ++bits;  // ceil(log2(sigma)) = max(1, bit_length(sigma-1))
  auto t_build0 = std::chrono::steady_clock::now();

  RrrWavelet w = build_wavelet(bwt, bits);
  auto t_build1 = std::chrono::steady_clock::now();

  // golden + self-check: CPU access must reconstruct the BWT exactly (the oracle validates itself).
  std::mt19937_64 rng(seed + 12345);
  std::uniform_int_distribution<int> Rp(0, (int)N - 1), Ri(0, (int)N), Rc(0, sigma - 1);
  std::vector<uint32_t> pos(nq), acc(nq), rc(nr), ri(nr), rg(nr);
  bool self_ok = true;
  for (int i = 0; i < nq; ++i) { pos[i] = (uint32_t)Rp(rng); acc[i] = cpu_access(w, (int)pos[i]); if (acc[i] != (uint32_t)bwt[pos[i]]) self_ok = false; }
  for (int i = 0; i < nr; ++i) { rc[i] = (uint32_t)Rc(rng); ri[i] = (uint32_t)Ri(rng); rg[i] = cpu_rank(w, (int)rc[i], (int)ri[i]); }

  // CPU baseline throughput (real, not a Python loop): median-ish via a single timed sweep over the golden set.
  auto ta = std::chrono::steady_clock::now();
  uint64_t sink = 0;
  for (int rep = 0; rep < 3; ++rep) for (int i = 0; i < nq; ++i) sink += cpu_access(w, (int)pos[i]);
  auto tb = std::chrono::steady_clock::now();
  double acc_ns = std::chrono::duration<double, std::nano>(tb - ta).count() / (3.0 * nq);
  auto tc = std::chrono::steady_clock::now();
  for (int rep = 0; rep < 3; ++rep) for (int i = 0; i < nr; ++i) sink += cpu_rank(w, (int)rc[i], (int)ri[i]);
  auto td = std::chrono::steady_clock::now();
  double rnk_ns = std::chrono::duration<double, std::nano>(td - tc).count() / (3.0 * nr);

  // packed-wavelet footprint (bitplanes + SB=8 superblocks) — matches GPUWavelet.index_bytes(), for the size line.
  int64_t nwords = (N + 31) / 32, nb = (nwords + 7) / 8;
  int64_t packed_bytes = (int64_t)bits * nwords * 4 + (int64_t)bits * (nb + 1) * 4;
  double h0 = 0.0;
  {
    std::vector<int64_t> cnt(sigma, 0);
    for (int64_t x : bwt) cnt[x]++;
    for (int64_t c : cnt) if (c) { double pph = (double)c / N; h0 -= pph * std::log2(pph); }
  }

  // serialise .cfrw v1 (byte-identical layout to tools/export_rrr_wavelet.py)
  FILE *f = std::fopen(out, "wb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", out); return 1; }
  uint32_t v1 = 1, u_bits = bits, u_vocab = vocab, u_nblocks = (uint32_t)((N + T - 1) / T);
  uint32_t u_nsb = w.nsb, u_cwords = w.cwords, u_na = w.na, u_owords = (uint32_t)w.offsets.size();
  uint32_t u_nq = nq, u_nr = nr;
  uint64_t u_n = (uint64_t)N, u_rrr = (uint64_t)w.rrr_bytes, u_pack = (uint64_t)packed_bytes;
  float f_h0 = (float)h0;
  wr(f, "CFRW", 4);
  wr(f, &v1, 4); wr(f, &u_n, 8); wr(f, &u_bits, 4); wr(f, &u_vocab, 4); wr(f, &u_nblocks, 4); wr(f, &u_nsb, 4);
  wr(f, &u_cwords, 4); wr(f, &u_na, 4); wr(f, &u_owords, 4); wr(f, &u_nq, 4); wr(f, &u_nr, 4);
  wr(f, &u_rrr, 8); wr(f, &u_pack, 8); wr(f, &f_h0, 4);
  wr(f, w.classes.data(), w.classes.size() * 4);
  wr(f, w.offsets.data(), w.offsets.size() * 4);
  wr(f, w.rank_a.data(), w.rank_a.size() * 4);
  wr(f, w.rank_d.data(), w.rank_d.size() * 2);
  wr(f, w.off_a.data(), w.off_a.size() * 4);
  wr(f, w.off_d.data(), w.off_d.size() * 2);
  wr(f, w.offbase.data(), w.offbase.size() * 4);
  wr(f, w.zeros.data(), w.zeros.size() * 4);
  wr(f, pos.data(), pos.size() * 4);
  wr(f, acc.data(), acc.size() * 4);
  wr(f, rc.data(), rc.size() * 4);
  wr(f, ri.data(), ri.size() * 4);
  wr(f, rg.data(), rg.size() * 4);
  std::fclose(f);

  double sa_s = std::chrono::duration<double>(t_build0 - t0).count();
  double enc_s = std::chrono::duration<double>(t_build1 - t_build0).count();
  std::printf("built %s  (native C++ — no Warp)\n", out);
  std::printf("  BWT length=%lld  vocab=%d  bits=%d  sigma=%d  nsb=%u  na=%u\n", (long long)N, vocab, bits, sigma,
              w.nsb, w.na);
  std::printf("  RRR wavelet %.2f MB (%.2f b/tok)  vs packed %.2f MB (%.2f b/tok)  => %.2fx smaller   H0=%.2f\n",
              w.rrr_bytes / 1e6, w.rrr_bytes * 8.0 / N, packed_bytes / 1e6, packed_bytes * 8.0 / N,
              (double)packed_bytes / w.rrr_bytes, h0);
  std::printf("  build: suffix array %.1fs + RRR encode %.1fs (offline, P9)\n", sa_s, enc_s);
  std::printf("  CPU backend baseline: access %.1f ns, rank %.1f ns  (scalar, 1 thread)   [sink %llu]\n", acc_ns,
              rnk_ns, (unsigned long long)sink);
  std::printf("  self-check: CPU access == BWT  %s\n", self_ok ? "OK" : "FAIL");
  return self_ok ? 0 : 3;
}
