"""export_delta.py — freeze a reference/delta-compressed cluster + golden reconstruction (.cfdc), M8. Warp-free.

Cross-sequence dedup — the shared-prefix / conversation-history / LoRA-library / near-duplicate-context path. A
cluster of near-identical sequences is folded into ONE base (the shared reference) + per-sequence **sparse
deltas** (the positions where a member diverges, plus its appended-turn suffix). fetch(seq, pos) = base[pos]
overridden by the sequence's own delta at pos (binary search). So N near-duplicate requests cost base-once +
small per-request diffs instead of N full copies. This freezes the base + per-sequence deltas + the golden
original sequences; the native kernel (src/cuda/delta_apply.cu) must reconstruct every token bit-for-bit. Pure
numpy (no Warp): the golden is the originals themselves. Run:

    python tools/export_delta.py out.cfdc --members 256 --base 8000 --divergence 0.01 --suffix 64

Binary layout v1 (little-endian):
    magic "CFDC" | u32 version=1 | u32 N | u32 nbase | u32 ndelta | u32 vocab | u64 total_tokens
    base[nbase] i32 | lengths[N] i32 | dstart[N] i32 | dlen[N] i32 | dpos[ndelta] i32 | dval[ndelta] i32
    ostart[N+1] i32 | originals[total_tokens] i32
(dpos is sorted ascending within each sequence: divergence positions in [0,nbase) then the appended suffix.)
"""
import argparse
import os
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--members", type=int, default=256, help="N near-duplicate sequences in the cluster")
    ap.add_argument("--base", type=int, default=8000, help="shared reference length")
    ap.add_argument("--divergence", type=float, default=0.01, help="fraction of prefix positions each member overrides")
    ap.add_argument("--suffix", type=int, default=64, help="unique appended-turn tokens per member")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    base = rng.integers(0, a.vocab, a.base).astype(np.int64)
    ndiv = max(0, int(round(a.divergence * a.base)))

    lengths, dstart, dlen, dpos, dval, ostart, originals = [], [], [], [], [], [0], []
    cursor = 0
    for _ in range(a.members):
        div = np.sort(rng.choice(a.base, size=ndiv, replace=False)) if ndiv else np.zeros(0, np.int64)
        divval = rng.integers(0, a.vocab, div.shape[0]).astype(np.int64)
        suf_pos = np.arange(a.base, a.base + a.suffix, dtype=np.int64)
        suf_val = rng.integers(0, a.vocab, a.suffix).astype(np.int64)
        pos = np.concatenate([div, suf_pos])                 # sorted: divergences (<nbase) then suffix (>=nbase)
        val = np.concatenate([divval, suf_val])
        dstart.append(cursor)
        dlen.append(int(pos.shape[0]))
        dpos.append(pos)
        dval.append(val)
        cursor += int(pos.shape[0])

        L = a.base + a.suffix
        lengths.append(L)
        seq = base.copy()
        seq[div] = divval                                    # apply the prefix overrides
        seq = np.concatenate([seq, suf_val])                 # append the unique suffix
        originals.append(seq)
        ostart.append(ostart[-1] + L)

    dpos = np.concatenate(dpos).astype(np.int32) if dpos else np.zeros(0, np.int32)
    dval = np.concatenate(dval).astype(np.int32) if dval else np.zeros(0, np.int32)
    originals = np.concatenate(originals).astype(np.int32)
    ndelta = int(dpos.shape[0])
    total = int(originals.shape[0])

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFDC")
        f.write(struct.pack("<IIIIIQ", 1, a.members, a.base, ndelta, a.vocab, total))
        for x in (base.astype(np.int32), np.asarray(lengths, np.int32), np.asarray(dstart, np.int32),
                  np.asarray(dlen, np.int32), dpos, dval, np.asarray(ostart, np.int32), originals):
            f.write(np.ascontiguousarray(x).tobytes())

    cluster = a.base * 4 + ndelta * 8 + a.members * 8         # base + (pos,val) deltas + per-seq directory
    dup = total * 4
    print(f"wrote {out}")
    print(f"  cluster: {a.members} members × (base {a.base} + suffix {a.suffix}), divergence {a.divergence:.1%}")
    print(f"  resident (base + sparse deltas) {cluster / 1e3:.1f} KB   vs   duplicated {dup / 1e6:.2f} MB"
          f"   => {dup / cluster:.1f}× less   golden originals frozen ✓")


if __name__ == "__main__":
    main()
