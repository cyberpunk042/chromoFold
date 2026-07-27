/* functional_device.c — GPU functional test of the DEVICE-NATIVE FM-search path through libchromofold.so.
 * Like functional_host.c it is PURE C with NO CUDA in the source (a Rust `-sys` FFI's position), but instead of the
 * host convenience layer it drives the device-native async API directly: cf_fm_host_view exposes the resident FM
 * view, cf_device_alloc/h2d put the query on the device ONCE, cf_fm_count_async runs the kernel on a stream,
 * cf_stream_sync + cf_device_d2h read the result back. The point (P5): buffers are resident and REUSED across
 * calls — no per-call malloc/memcpy/free — and a device-resident consumer could skip the download entirely.
 * Verified bit-identical to the fixture's golden. Links libchromofold + cudart; the caller never calls CUDA. */
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

    /* Parse header + skip the index arrays to reach the fixture's patterns + golden counts (as functional_host.c). */
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
    p += (size_t) bits * cwords * 4; p += (size_t) owords * 4;
    p += (size_t) bits * na * 4;     p += (size_t) bits * nsb1 * 2;
    p += (size_t) bits * na * 4;     p += (size_t) bits * nsb1 * 2;
    p += (size_t) bits * 4; p += (size_t) bits * 4; p += (size_t) sigma * 4;
    p += (size_t) mwords_len * 4; p += (size_t) msb_len * 4; p += (size_t) nsval * 4;
    int32_t *pat = (int32_t *) malloc((size_t) patflat * 4); RD(pat, (size_t) patflat * 4);
    int32_t *pstart = (int32_t *) malloc((size_t) npat * 4); RD(pstart, (size_t) npat * 4);
    int32_t *plen = (int32_t *) malloc((size_t) npat * 4);   RD(plen, (size_t) npat * 4);
    uint32_t *golden = (uint32_t *) malloc((size_t) npat * 4); RD(golden, (size_t) npat * 4);

    /* Load the index once, and take the resident device view. */
    cf_fm_host_index *ix = NULL;
    if (cf_fm_host_load(blob, nbytes, &ix) != CF_OK || !ix) { fprintf(stderr, "cf_fm_host_load failed\n"); return 2; }
    cf_fm_view view;
    if (cf_fm_host_view(ix, &view) != CF_OK) { fprintf(stderr, "cf_fm_host_view failed\n"); return 2; }

    /* Resident device buffers, allocated ONCE (P9 build != query; here also alloc != query). */
    void *d_pat = NULL, *d_pstart = NULL, *d_plen = NULL, *d_counts = NULL, *stream = NULL;
    cf_status st = CF_OK;
    st |= cf_device_alloc((size_t) patflat * 4, &d_pat);
    st |= cf_device_alloc((size_t) npat * 4, &d_pstart);
    st |= cf_device_alloc((size_t) npat * 4, &d_plen);
    st |= cf_device_alloc((size_t) npat * 4, &d_counts);
    st |= cf_stream_create(&stream);
    if (st != CF_OK) { fprintf(stderr, "device alloc/stream failed\n"); return 2; }

    /* Upload the query ONCE; then run the kernel twice reusing the SAME resident buffers (the zero-copy hot loop). */
    st |= cf_device_h2d(d_pat, pat, (size_t) patflat * 4);
    st |= cf_device_h2d(d_pstart, pstart, (size_t) npat * 4);
    st |= cf_device_h2d(d_plen, plen, (size_t) npat * 4);
    if (st != CF_OK) { fprintf(stderr, "upload failed\n"); return 2; }

    uint32_t *counts = (uint32_t *) malloc((size_t) npat * 4);
    int mism = 0;
    for (int rep = 0; rep < 2; ++rep) {
        st = cf_fm_count_async(view, (const int32_t *) d_pat, (const int32_t *) d_pstart,
                               (const int32_t *) d_plen, (uint32_t *) d_counts, npat, stream);
        if (st != CF_OK) { fprintf(stderr, "cf_fm_count_async status %d\n", (int) st); return 2; }
        if (cf_stream_sync(stream) != CF_OK) { fprintf(stderr, "stream sync failed\n"); return 2; }
        if (cf_device_d2h(counts, d_counts, (size_t) npat * 4) != CF_OK) { fprintf(stderr, "download failed\n"); return 2; }
        for (uint32_t i = 0; i < npat; ++i) if (counts[i] != golden[i]) ++mism;
    }

    /* null_arg_contract on the new device-native entry points (checked before any CUDA call). */
    int null_ok = (cf_fm_host_view(NULL, &view) == CF_ERR_INVALID_ARGUMENT)
               && (cf_device_alloc(16, NULL) == CF_ERR_INVALID_ARGUMENT)
               && (cf_device_h2d(NULL, pat, 4) == CF_ERR_INVALID_ARGUMENT)
               && (cf_stream_create(NULL) == CF_ERR_INVALID_ARGUMENT);

    cf_device_free(d_pat); cf_device_free(d_pstart); cf_device_free(d_plen); cf_device_free(d_counts);
    cf_stream_destroy(stream);
    cf_fm_host_free(ix);

    printf("libchromofold DEVICE-NATIVE FM-search (pure C caller, NO CUDA in source) — %s\n", path);
    printf("  loaded index; view exposed; query resident on device, kernel run 2x reusing buffers; patterns=%u\n", npat);
    printf("  cf_fm_count_async vs golden : %s (%d/%u mismatch over 2 reps)\n",
           mism == 0 ? "BIT-IDENTICAL" : "MISMATCH", mism, npat * 2);
    printf("  null_arg_contract           : %s\n", null_ok ? "OK" : "FAIL");
    if (mism == 0 && null_ok) { printf("PASS — device-native path correct; no host round-trip in the query loop.\n"); return 0; }
    return 1;
}
