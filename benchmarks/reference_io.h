// reference_io.h — load a frozen ChromoFold reference vector (.cfwv v2). Shared by the benchmarks.
#ifndef CF_REFERENCE_IO_H
#define CF_REFERENCE_IO_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Ref {
  uint32_t levels = 0, nwords = 0, nblocks = 0, nqueries = 0, nrank = 0, token_bytes = 0, vocab = 0;
  uint64_t n = 0;
  std::vector<uint32_t> words;      // [levels * nwords]
  std::vector<int32_t> sb, zeros;   // [levels*(nblocks+1)], [levels]
  std::vector<uint32_t> pos, golden;
  std::vector<uint32_t> rank_c, rank_i, rank_golden;  // rank queries + golden counts
  std::vector<uint8_t> raw;         // n * token_bytes (minimal-width raw token stream)
};

static inline bool cf_load_reference(const char *path, Ref &r) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "CFWV", 4) == 0;
  auto rd = [&](void *p, size_t bytes) { ok = ok && std::fread(p, 1, bytes, f) == bytes; };
  rd(&version, 4);
  if (version != 3) {
    std::fclose(f);
    return false;
  }
  rd(&r.levels, 4);
  rd(&r.n, 8);
  rd(&r.nwords, 4);
  rd(&r.nblocks, 4);
  rd(&r.nqueries, 4);
  rd(&r.nrank, 4);
  rd(&r.token_bytes, 4);
  rd(&r.vocab, 4);
  if (!ok) {
    std::fclose(f);
    return false;
  }
  r.words.resize((size_t)r.levels * r.nwords);
  r.sb.resize((size_t)r.levels * (r.nblocks + 1));
  r.zeros.resize(r.levels);
  r.pos.resize(r.nqueries);
  r.golden.resize(r.nqueries);
  r.rank_c.resize(r.nrank);
  r.rank_i.resize(r.nrank);
  r.rank_golden.resize(r.nrank);
  r.raw.resize((size_t)r.n * r.token_bytes);
  rd(r.words.data(), r.words.size() * 4);
  rd(r.sb.data(), r.sb.size() * 4);
  rd(r.zeros.data(), r.zeros.size() * 4);
  rd(r.pos.data(), r.pos.size() * 4);
  rd(r.golden.data(), r.golden.size() * 4);
  rd(r.rank_c.data(), r.rank_c.size() * 4);
  rd(r.rank_i.data(), r.rank_i.size() * 4);
  rd(r.rank_golden.data(), r.rank_golden.size() * 4);
  rd(r.raw.data(), r.raw.size());
  std::fclose(f);
  return ok;
}

static inline double cf_index_mb(const Ref &r) {
  return (r.words.size() + r.sb.size() + r.zeros.size()) * 4.0 / 1e6;
}

#endif // CF_REFERENCE_IO_H
