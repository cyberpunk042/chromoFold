/* conformance.c — pure-seam ABI conformance for libchromofold (sovereign-os SDD-500 / Q-500-C).
 *
 * Validates that the shared library exports the search ABI, the C header signatures match, and the NULL-argument
 * error contract holds — WITHOUT a GPU or driver. Every call here returns before any CUDA call (the argument
 * guards fire first), so this is the "pure seam" a CI job with no SAIN-01 box can run. Exit 0 = PASS.
 */
#include "chromofold/chromofold.h"
#include "chromofold/chromofold_search.h"

#include <stdio.h>
#include <string.h>

static int check(const char *name, cf_status got) {
  int ok = (got == CF_ERR_INVALID_ARGUMENT);
  printf("  %-24s -> status %d  %s\n", name, (int)got, ok ? "OK" : "FAIL");
  return ok;
}

int main(void) {
  cf_wavelet_view wv;
  cf_rrrw_view rv;
  cf_fm_view fv;
  memset(&wv, 0, sizeof wv);
  memset(&rv, 0, sizeof rv);
  memset(&fv, 0, sizeof fv);

  int ok = 1;
  printf("libchromofold conformance — ABI v%d, structs: cf_wavelet_view=%zu cf_rrrw_view=%zu cf_fm_view=%zu B\n",
         CHROMOFOLD_ABI_VERSION, sizeof(cf_wavelet_view), sizeof(cf_rrrw_view), sizeof(cf_fm_view));
  ok &= check("cf_access_async", cf_access_async(wv, NULL, NULL, 1, NULL));
  ok &= check("cf_rank_async", cf_rank_async(wv, NULL, NULL, NULL, 1, NULL));
  ok &= check("cf_rrrw_access_async", cf_rrrw_access_async(rv, NULL, NULL, 1, NULL));
  ok &= check("cf_rrrw_rank_async", cf_rrrw_rank_async(rv, NULL, NULL, NULL, 1, NULL));
  ok &= check("cf_fm_count_async", cf_fm_count_async(fv, NULL, NULL, NULL, NULL, 1, NULL));
  ok &= check("cf_fm_ranges_async", cf_fm_ranges_async(fv, NULL, NULL, NULL, NULL, NULL, 1, NULL));
  ok &= check("cf_fm_locate_async", cf_fm_locate_async(fv, NULL, NULL, 1, NULL));
  printf("%s\n", ok ? "PASS — search ABI exported and the NULL-argument contract holds (no GPU used)."
                    : "FAIL — ABI seam mismatch.");
  return ok ? 0 : 1;
}
