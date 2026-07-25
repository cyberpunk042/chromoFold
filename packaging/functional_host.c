/* functional_host.c — GPU functional test of the HOST-pointer FM-search API through libchromofold.so.
 * Deliberately PURE C with NO CUDA in the source (exactly the position of a Rust `-sys` FFI): read a .cffm blob,
 * cf_fm_host_load it, count the fixture's own patterns with plain host arrays, and check bit-identical to the
 * golden the fixture ships. This is the sovereign-os SDD-400 "step 7" (host<->device marshalling) done inside the
 * engine instead of the caller. Links libchromofold + cudart; the caller code never calls a CUDA function. */
#include "chromofold/chromofold_search.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *) malloc((size_t) sz);
    if (!b || fread(b, 1, (size_t) sz, f) != (size_t) sz) { free(b); fclose(f); return NULL; }
    fclose(f); *n = (size_t) sz; return b;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "fixtures/tiny.cffm";
    size_t nbytes = 0;
    uint8_t *blob = slurp(path, &nbytes);
    if (!blob) { fprintf(stderr, "cannot read %s\n", path); return 1; }

    /* Parse the header + skip the index arrays to reach the fixture's test patterns + golden counts. */
    const uint8_t *p = blob;
#define RD(dst, sz) do { memcpy((dst), p, (sz)); p += (sz); } while (0)
    char magic[4]; RD(magic, 4);
    uint32_t version; RD(&version, 4);
    uint64_t n; RD(&n, 8);
    uint32_t bits, vocab, sigma, nblocks, nsb, cwords, na, owords, sa_sample, mwords_len, msb_len, nsval, npat, patflat, nloc;
    RD(&bits, 4); RD(&vocab, 4); RD(&sigma, 4); RD(&nblocks, 4); RD(&nsb, 4); RD(&cwords, 4);
    RD(&na, 4); RD(&owords, 4); RD(&sa_sample, 4); RD(&mwords_len, 4); RD(&msb_len, 4);
    RD(&nsval, 4); RD(&npat, 4); RD(&patflat, 4); RD(&nloc, 4);
    uint64_t rrr_bytes; RD(&rrr_bytes, 8);
    if (memcmp(magic, "CFFM", 4) != 0 || version != 1) { fprintf(stderr, "not a v1 .cffm\n"); return 1; }
    uint32_t nsb1 = nsb + 1;
    p += (size_t) bits * cwords * 4; p += (size_t) owords * 4;           /* classes, offsets */
    p += (size_t) bits * na * 4;     p += (size_t) bits * nsb1 * 2;      /* rank_a, rank_d */
    p += (size_t) bits * na * 4;     p += (size_t) bits * nsb1 * 2;      /* off_a, off_d */
    p += (size_t) bits * 4; p += (size_t) bits * 4; p += (size_t) sigma * 4;  /* offbase, zeros, C */
    p += (size_t) mwords_len * 4; p += (size_t) msb_len * 4; p += (size_t) nsval * 4;  /* mwords, msb, sval */
    int32_t *pat = (int32_t *) malloc((size_t) patflat * 4); RD(pat, (size_t) patflat * 4);
    int32_t *pstart = (int32_t *) malloc((size_t) npat * 4); RD(pstart, (size_t) npat * 4);
    int32_t *plen = (int32_t *) malloc((size_t) npat * 4);   RD(plen, (size_t) npat * 4);
    uint32_t *golden = (uint32_t *) malloc((size_t) npat * 4); RD(golden, (size_t) npat * 4);

    /* The host API: load once, query with host arrays — no CUDA on this side. */
    cf_fm_host_index *ix = NULL;
    if (cf_fm_host_load(blob, nbytes, &ix) != CF_OK || !ix) { fprintf(stderr, "cf_fm_host_load failed\n"); return 2; }
    uint32_t *counts = (uint32_t *) malloc((size_t) npat * 4);
    cf_status st = cf_fm_host_count(ix, pat, pstart, plen, npat, counts);
    if (st != CF_OK) { fprintf(stderr, "cf_fm_host_count status %d\n", (int) st); return 2; }
    int mism = 0;
    for (uint32_t i = 0; i < npat; ++i) if (counts[i] != golden[i]) ++mism;
    int null_ok = (cf_fm_host_count(NULL, pat, pstart, plen, npat, counts) == CF_ERR_INVALID_ARGUMENT)
               && (cf_fm_host_load(NULL, 0, &ix) == CF_ERR_INVALID_ARGUMENT);
    cf_fm_host_free(ix);

    printf("libchromofold host FM-search (pure C caller, NO CUDA in source) — %s\n", path);
    printf("  loaded index from a %zu-byte .cffm blob; patterns=%u\n", nbytes, npat);
    printf("  cf_fm_host_count vs golden : %s (%d/%u mismatch)\n", mism == 0 ? "BIT-IDENTICAL" : "MISMATCH", mism, npat);
    printf("  null_arg_contract          : %s\n", null_ok ? "OK" : "FAIL");
    if (mism == 0 && null_ok) { printf("PASS — host API serves FM-search correctly; the caller does no device marshalling.\n"); return 0; }
    return 1;
}
