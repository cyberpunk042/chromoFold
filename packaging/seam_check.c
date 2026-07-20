/* seam_check.c — the ".cfold header" pure seam (sovereign-os SDD-500 / Q-500-C).
 *
 * Reads a ChromoFold reference fixture and validates its stable header prefix (4-byte magic + u32 version)
 * without a GPU or the CUDA runtime — the header-parse seam a no-SAIN-01 CI job runs. Recognized magics:
 *   CFWV  packed-wavelet index (+ golden access)      version 3
 *   CFRR  RRR bitvector (+ golden rank1)              version 1
 *   CFRW  RRR-backed wavelet (BWT'd, + golden a/r)    version 1
 * Exit 0 = a recognized magic; the version is printed so a consumer can gate on it.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *fmt_of(const char *m) {
  if (!memcmp(m, "CFWV", 4)) return "packed-wavelet";
  if (!memcmp(m, "CFRR", 4)) return "RRR-bitvector";
  if (!memcmp(m, "CFRW", 4)) return "RRR-wavelet";
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FIXTURE.cf{wv,rr,rw}\n", argv[0]);
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    perror(argv[1]);
    return 2;
  }
  char magic[5] = {0};
  uint32_t version = 0;
  int rd = (fread(magic, 1, 4, f) == 4) && (fread(&version, 4, 1, f) == 1);
  fclose(f);
  if (!rd) {
    fprintf(stderr, "  %-28s short read (not a ChromoFold fixture)\n", argv[1]);
    return 2;
  }
  const char *fmt = fmt_of(magic);
  printf("  %-28s magic=%.4s version=%u  %s\n", argv[1], magic, version,
         fmt ? fmt : "UNKNOWN MAGIC");
  return fmt ? 0 : 1;
}
