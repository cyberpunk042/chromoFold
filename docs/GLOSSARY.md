# Glossary — every term in plain language

This project borrows vocabulary from three fields: **succinct data structures** (compact indexes), **compression**
(entropy coding), and **LLM systems** (GPUs, KV caches, quantization). Below is every term you'll meet, each with a
short definition and an everyday analogy. Nothing here assumes prior knowledge.

New to the project? Read [`../README.md`](../README.md) first, then [`CONCEPTS.md`](CONCEPTS.md) for how it all
fits together, then [`RESULTS.md`](RESULTS.md) for what was measured.

---

## The big picture

**Succinct data structure.** A way to store data that is *almost as small as the information it contains* (near the
"entropy" lower bound) **but still supports fast operations directly on the compact form** — no unpacking first.
*Analogy: a phone book that's been shrunk to a fraction of its size but you can still flip straight to any name.*

**Compute-for-memory.** The project's core trade: GPUs have lots of spare arithmetic but little fast memory, so we
deliberately spend extra computation to shrink memory use. *Analogy: re-deriving a fact each time instead of
keeping a big lookup table — slower per query, but the table never has to fit in your pocket.*

**Fusion (decode-in-consumer).** Instead of "decompress everything, then compute," a single GPU kernel decodes each
value *the instant it uses it*, in fast on-chip registers, so the decompressed data never exists as a buffer in
memory. *Analogy: a chef prepping each ingredient exactly when the recipe calls for it, instead of prepping the
whole pantry onto the counter first.*

**Lossless (over the chosen precision).** The compression is exactly reversible: what you get out equals what went
in, bit for bit. If the input was already rounded (quantized) to 4-bit, we losslessly store *those* 4-bit values.
*Analogy: a perfect photocopy of an already-low-resolution photo — no new blur added.*

**Bit-identical / golden.** Before trusting any fast GPU kernel, we run it and check its output equals a
pre-computed **reference file** (the "golden") *exactly* — not "close," but every bit the same. *Analogy: checking
a translation against an official answer key, character by character.*

---

## Sequences, symbols, and indexes

**Token / symbol.** One unit in a sequence — a word-piece for an LLM, a letter for text, a value for a weight
stream. A **vocabulary** (or **alphabet**) of size `V` is the set of distinct symbols; you need `log₂(V)` bits to
name one.

**Access(i).** "What symbol is at position *i*?" The most basic query. In a plain array it's trivial; over a
*compressed* index it takes cleverness to still answer in constant time.

**Rank(c, i).** "How many times does symbol *c* appear in the first *i* positions?" A surprisingly powerful
primitive — substring search is built out of it. *Analogy: "how many times has the letter 'e' appeared so far?"*

**Select(c, k).** The inverse of rank: "where is the *k*-th occurrence of symbol *c*?"

**Suffix array (SA).** A list of every starting position in a text, sorted by the text that follows it. It's the
classic structure behind fast substring search. *Analogy: an index at the back of a book listing every position,
ordered so all similar continuations sit together.* Building it is the expensive part of construction; we do it on
the GPU (21–23× faster than CPU — milestone M5).

**BWT (Burrows–Wheeler Transform).** A reversible reshuffle of a text that groups similar contexts together, which
makes it **much more compressible** and is the backbone of the FM-index. *Analogy: sorting a deck so all the cards
that tend to follow the same card end up next to each other — suddenly there are long runs you can pack tightly.*
Crucially, the BWT's bit-patterns are **skewed** (lots of one symbol), which is exactly where entropy coders win.

**Wavelet tree / wavelet matrix.** The data structure that lets you do `access`, `rank`, and `select` over a
*multi-symbol* sequence using only operations on *bits*. It slices the sequence into `log₂(V)` layers of 0/1 bits;
answering a query walks down the layers. *Analogy: a game of twenty questions — each layer asks one yes/no
question ("is the symbol in the top half of the alphabet?") and narrows down the answer.*

**FM-index.** A full-text search index built from the BWT + rank. It answers **`count`** ("how many times does
this pattern occur?") and **`locate`** ("at which positions?") by **backward search** — walking the pattern
right-to-left, shrinking a range with each `rank` call. It's what genome aligners use to search a whole genome in a
compact index; here it searches token streams. (Milestone M7.) *Analogy: narrowing down a suspect by adding one
clue at a time, each clue cutting the candidate list.*

**Backward search.** The FM-index's search loop: start with "the whole text could match," then feed the pattern's
characters from last to first, each one narrowing the matching range via `rank`. When the range is empty, there's
no match; its size is the number of occurrences.

**LF-mapping / LF-walk.** The step that turns a position in the BWT back into a text position, used by `locate`.
Repeatedly applying it "walks" from an occurrence to a known landmark. *Analogy: following "you are here → previous
step → previous step…" breadcrumbs until you hit a signpost with real coordinates.*

---

## Compression / entropy coding

**Entropy (H₀).** The information-theoretic *minimum* average bits per symbol, given how often each symbol occurs.
A coin that's 50/50 needs 1 bit per flip; a coin that's 99/1 needs far less on average. `H₀` is the floor any
lossless coder is trying to reach. *Analogy: the shortest possible average Morse-code length for a given letter
frequency.*

**Bits per token/value (b/tok, b/val).** The measured size of a compressed representation, divided by the number
of items. Lower is smaller. We always compare it to `H₀` (the floor) and to a naive layout (the ceiling).

**Huffman coding.** A classic entropy coder giving each symbol a whole number of bits (short codes for common
symbols). Simple and fast, but it "wastes" up to ~1 bit per symbol because code lengths must be whole numbers —
which hurts a lot when the true entropy is well below 1 bit/symbol.

**Canonical Huffman + LUT decode.** A specific fast way to decode Huffman: read a fixed number of bits, look the
result up in a table, and the table tells you both the symbol and how many bits it really used. Used in the fused
weight decoder. *Analogy: a decoder ring where one glance resolves the next letter and how far to advance.*

**rANS (range Asymmetric Numeral Systems).** A modern entropy coder (used in Zstandard/FSE and NVIDIA's nvCOMP)
that can spend *fractional* bits per symbol, so it reaches much closer to `H₀`. The catch: it carries a small fixed
"state" per block, so it only pays off on very skewed data with large blocks. We measure the exact crossover with
Huffman (milestone M4). *Analogy: Huffman prices everything in whole coins; rANS can make change to the cent.*

**RRR (Raman–Raman–Rao).** A way to store a bit-vector near its entropy **while keeping `rank` fast**. It cuts the
bits into small blocks and stores each as `(class, offset)` = (how many 1s, which specific arrangement). Skewed
bit-planes (like the BWT's) then cost far below 1 bit/bit. The clever part is decoding one block "in registers"
with a tiny combinatorial calculation. (Milestone M4.) *Analogy: instead of writing out a 15-bit pattern, note "3
ones, arrangement #452" — much shorter when patterns are lopsided.*

**Block / block-wise coding.** Compression is often sequential (you must decode from the start). Cutting the stream
into independent fixed-size **blocks** restores two things: **parallelism** (one GPU thread per block) and **random
access** (jump to a block, decode within it). The **block size** is a dial: bigger = tighter compression but less
parallelism; smaller = faster random access but more overhead.

**Superblock / two-level directory.** To answer `rank` quickly you pre-store running totals as "signposts." A
**two-level** scheme stores coarse signposts (big `int`s, rarely) plus fine deltas (tiny `int`s, often), halving
the signpost memory. (Milestone M3.) *Analogy: mile-markers every 10 miles plus small "+0.2 mi" ticks, instead of
a full-size marker at every tenth of a mile.*

---

## LLM / GPU systems

**GPU / VRAM.** The graphics processor and its dedicated fast memory. VRAM is the scarce resource — models,
caches, and activations must all fit. This project trades compute (abundant) for VRAM (scarce).

**Kernel.** A small program that runs on the GPU across thousands of threads at once. A "fused kernel" is one that
decodes-and-computes in a single pass (see **Fusion**).

**Device-native / device pointer.** An API that takes and returns *pointers to GPU memory* and runs on the caller's
GPU stream — **no copying data back to the CPU** and no hidden synchronization. This is essential: copying to the
CPU would erase the memory savings. (Principle P5.)

**KV cache (keys & values).** As an LLM reads a conversation, it stores a Key and a Value vector for every token so
it doesn't recompute them. This cache **grows with the conversation** and becomes the dominant VRAM cost in
long-context serving. Compressing it (milestone M6/M9) directly buys longer context. *Analogy: the model's
scratch-pad notes on everything said so far — the longer the chat, the fatter the notebook.*

**Attention.** The operation where a token "looks at" earlier tokens (via their Keys) and pulls a weighted mix of
their Values. It's the main consumer of the KV cache. **Windowed/sparse attention** only looks at a subset of
earlier tokens — a "light/sparse consumer," which is exactly when fusion pays off.

**Quantization.** Rounding the model's numbers to fewer bits (e.g. 16-bit → 4-bit) to save memory. It is *lossy*
(introduces rounding error) — that's the model author's accuracy trade, made *before* chromoFold. chromoFold then
stores those quantized values **losslessly** and can entropy-code them smaller still. **KIVI** is a specific
KV-quantization scheme (per-channel for Keys, per-token for Values).

**Embedding / embedding table.** A big lookup table mapping each token id to a vector. "Embedding gather" =
fetching the rows for a batch of tokens.

**GEMM / matmul.** A matrix multiply — the workhorse of neural networks. "Fused decode-in-matmul" multiplies by a
*compressed* weight matrix, decoding each weight inside the multiply so the full dense matrix never materializes
(milestone M6).

**Delta / reference-delta cluster.** A way to store many *near-identical* sequences (e.g. requests sharing a long
prompt, or a library of fine-tuning adapters): keep **one base** copy + each sequence's small list of
**differences**. Reconstructing a token = "take the base, unless this sequence overrides it here." (Milestone M8.)
*Analogy: a group email thread stored as the original message + each person's short edits, not 50 full copies.*

---

## Project-specific shorthand

**P1–P10.** The ten principles in the [constitution](../specs/00-constitution.md). The ones you'll see most:
**P2** navigable while compressed · **P3** decode only inside the consumer · **P7** report negatives honestly.

**M0–M11.** The milestones in the [roadmap](../specs/03-roadmap.md). Roughly: M0 harness → M1 read → M3 rank → M4
compressed searchable index + coders → M5 build it natively → M6 fused consumers → M7 search → M8 dedup → M9 the
real-model proof.

**The prototype / oracle / Warp.** The Python research prototype (`warp_compress`) whose outputs are the "golden"
references this engine must match exactly. [Warp](https://github.com/NVIDIA/warp) is NVIDIA's Python-to-GPU
framework it's written in.

**`.cfwv`, `.cfrr`, `.cfrw`, `.cffm`, `.cffw`, `.cfkv`, `.cfrs`, `.cfsg`, `.cfdc`.** The little binary "golden"
reference-file formats, one per capability (packed wavelet, RRR bit-vector, RRR wavelet, FM-index, fused weights,
KV cache, rANS stream, sparse-gather, delta cluster). They are generated on demand, not committed.
