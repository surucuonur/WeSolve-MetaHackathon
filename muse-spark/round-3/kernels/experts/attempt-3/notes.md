# attempt-3 (round 3): grouped/tiled prefill replica (builder-2-c)

Hypothesis `hyp-2-experts-prefill`, slot `experts`, baseline `none`,
starting point `muse-spark/round-2/kernels/experts/attempt-7`
(G1+G2 pass, slower at 1 row and at every prefill size).
Direction (c): grouped/tiled replica of `gather_qmm_rhs` issue.

## The one main change

Four kernels replace the single fused kernel. The decode path is
starting-point code; the `rows >= 60` prefill branch is replaced by a
grouped, fused, MMA-tiled path in the shape of MLX's `gather_qmm_rhs`.

- **Decode kernel** (`block-0.metal` + `block-1.metal`): attempt-7's
  body with one added line (`if (prefill) return;` above the dispatch)
  and the output renamed `y` to `y_decode` (3 stores). Every decode
  computation line below the return is byte-identical to attempt-7,
  including the dead prefill `else` branches (never executed: the
  return exits first at rows >= 60, the `!prefill` guard skips them
  below). Helpers (`qvm_gate_up_pair`, `qvm_down_pair`, `qmm_accum`)
  are byte-identical by copy. Grid/threadgroup unchanged
  (`rows*2048`, 512). At decode sizes it computes exactly attempt-7;
  at prefill sizes its groups exit before any load.
- **Count kernel** (`block-4.metal`, new): one group of 64 threads
  counting-sorts the `B = rows*4` pairs by expert (threads 0-59 own
  one expert, two scans over `inds`) and writes `sorted_pairs[B]`,
  `expert_offsets[61]` and `tile_offsets[61]` (row-tile prefix, 16
  pairs per tile). Runs at every size; trivial cost (<= 8192 pairs).
  Computed once per call (MLX recomputes offsets per product).
- **Prefill kernel** (`block-2.metal` + `block-3.metal`, new): one
  group per row tile (`rows*256` threads, 64 per group, `rows*4`
  groups). Groups past `tile_offsets[60]` exit after one read (no
  empty-tile schedule scans). Each useful group owns one (expert,
  16-pair tile): it loads the tile's pair list, streams gate/up
  weight tiles once per tile (shared across <= 16 rows), accumulates
  with 8-wide frag MMA in contiguous BK=32 order (replicating
  `BlockMMA` per-output order: sequential K-tiles, 4 8-wide steps),
  runs the full-bfloat16 swiglu on-chip into `tg_h`, then streams
  down weights the same way and stores positionally. Dequant is
  MLX's bits=4 form verbatim (`s*(low)+b`, `(s/16)*(high mask)+b`,
  bfloat-rounded). Threadgroup: `tg_h` 12,288 B + `Xs` 1,280 B +
  two `Ws` 2,560 B each + pairs 64 B = 18,752 B (of 32,768 B).
  Rows < 60 return immediately.
- **Select kernel** (`block-5.metal`, new): one thread per `y`
  element copies `y_decode` or `y_prefill` on `rows >= 60`. Exact
  bfloat copy, no rounding.

No `max_rows`: kernels run at every row count (decode kernel below
60, prefill kernel at/above, selected by the same threshold MLX uses).

## Which inefficiency it removes, and how

- `ineff-baseline-experts-1` (scalar prefill 15-28x slower): the
  scalar `qmm_accum` per-pair branch never runs (decode kernel exits,
  prefill kernel replaces it). Cross-row weight sharing removes the
  8.5/13.7/15.3x distinct-traffic gap (one dequant per tile serves
  <= 16 rows); hardware MMA removes the scalar-issue gap (32 8x8
  MACs per K-tile per group vs 32 scalar MACs); 64-thread groups
  remove the 512-thread-per-pair overprovisioning. Takes the
  tile-waste prize (no dispatched empty tiles; partial tiles still
  pad, as in MLX) and the offsets prize (one count pass, not three)
  for free, plus x row-tile re-reads (x staged once per K-tile, not
  per N-tile) and fused gate/up/hidden traffic (never leaves the
  chip per tile).
- `ineff-baseline-experts-2` (1-row decode gap): deliberately kept.
  The decode path is attempt-7's, so the 1-row loss persists; hyp-4
  fixes it on this attempt's winner. A prefill fix that slows decode
  is not an improvement (ADR 0011): the added decode cost is one
  exact copy plus two near-empty launches (below).

## Limits (by hand)

Decode: 512 threads/group, 3,840 B (unchanged). Prefill: 64
threads/group, 18,752 B. Count: 64 threads, ~500 B. Select: 256
threads, none. All <= 1,024 threads and <= 32 KB. Grids scale with
rows at every size; the `gid -> (expert, tile)` map covers the worst
case (all pairs one expert needs `ceil(B/16)` tiles; `B` groups with
dense tile offsets always suffice). Each `y` element is written
exactly once (each pair in exactly one tile; within a tile each
thread's 8 frag elements are disjoint; the select copies each once).

## Predicted us per layer at the six sizes (chained)

Basis (measured, piece,
`results/runs/2026-11-04-muse-round-2/grade-phases-experts-attempt-7.json`):
starting point vs compiled 92.1 vs 78.6 (1 row), 162.8 vs 235.5 (8),
515.3 vs 797.8 (32), 15,220.0 vs 960.8 (127), 119,154.4 vs 4,306.0
(1,016), 240,681.1 vs 11,998.9 (2,048). Baseline `none`.

Decode (rows < 60 run decode + count + select; prefill groups exit):
select copies 6/49/196 KB at 0.03/0.25/1.0 us byte floor + ~1 us
launch; empty prefill groups (4/32/128 x 64 threads, one compare)
~0.5/1/2 us; count (B <= 128) < 0.5 us. Total overhead ~2/3/4 us.

- 1 row: 92.1 + 3 = **95** (vs 78.6 compiled, `slower`, keeps the
  starting-point loss for hyp-4).
- 8 rows: 162.8 + 3 = **166** (vs 235.5, `faster`, keeps the win).
- 32 rows: 515.3 + 4 = **519** (vs 797.8, `faster`, keeps the win).

Prefill (rows >= 60 run count + prefill + select; decode groups exit):
grouped MMA at the compiled rate (~961/4,306/11,999) minus banked
waste (empty-tile schedule scans on 1,488/624/1,488 empty groups per
layer, offsets <= 13 us/layer, x re-reads where they miss, fused
gate/up/hidden device traffic ~12/50/100 us if DRAM) plus overhead
(select 3.9/31/63 us byte floors + launch, decode-empty 508/4,064/
8,192 x 512 threads ~5/15/30 us, count ~1/5/10 us). Net ~5% faster:

- 127 rows: **900** (vs 960.8, `faster`).
- 1,016 rows: **4,100** (vs 4,306.0, `faster`).
- 2,048 rows: **11,500** (vs 11,998.9, `faster`).

G1 prediction: pass (decode is attempt-7's legal launch; prefill is
64 threads and 18.8 KB; count/select trivial). G2 prediction: pass
with `differ == 0` (decode byte-identical; prefill replicates dequant
values, BK=32 frag order and the four bfloat16 points; select exact).
`beats_compiled` prediction: false (prefill `faster` everywhere,
decode `slower` at 1 row) -- the prefill phase win is the claim;
hyp-4 takes the 1-row remainder.

## What this attempt leaves unmeasured

Kernel speed at 33-126 rows (piece timing jumps 32 to 127); sizes
above 2,048; prefill beyond the first seeded chunk, other
lengths/distributions, mixed batches; the 32 KB limit (hand-checked
above); unwritten threadgroup reads and thread-timing dependence
(none by construction: all groups take uniform early exits, all tg
writes precede barriers); NaN-vs-NaN equality; whole-request
composition (ready check). The 2,048-row compiled basis carries
run-to-run variance (round-3 `territory-quantized-kernels.md` S5
caveat: 11,998.9 vs 7,347.5 across runs).

## Verdict

G1 **fail** (no G2, no timing). `grade.json`: `g1: false`,
`error: [metal::Device] Unable to build metal library from source`,
six errors all from one cause: `block-3.metal` declares
`uint half = thread_idx & 1u;` (`dequant_tile_4bit`), but `half` is a
reserved type name in Metal (half-precision float), so the compiler
rejects it as `cannot combine with previous 'type-name' declaration
specifier` and every use of the variable fails to parse. Nothing else
was reached: decode exactness, prefill bits and all six timings are
unmeasured. Predicted 95/166/519 decode and 900/4,100/11,500 prefill
vs compiled 78.6/235.5/797.8 and 960.8/4,306.0/11,998.9 stand as
predictions beside no measured values.

## Why it failed (miss analysis)

The failure is a Metal reserved word, not the design: `half` (like
`float`, `uint`, `bfloat`) cannot name a variable. The fix is a
rename (`half` -> `part`) in `dequant_tile_4bit` (1 declaration + 5
uses, `block-3.metal`). No other identifier in the attempt collides
(`e1`, `yy`, `ss`, `pp` in block-2 avoid the `y`/`s`/`p` outputs;
`sg`, `fm`, `fn`, `w_lo`/`w_hi` are free). The design itself --
count + decode + grouped-MMA prefill + select, 18,752 B threadgroup,
dense tile-offset map -- never compiled, so its G2/timing predictions
are untested, not refuted.

## Next change

Copy this attempt to a new number, rename `half` to `part` in
`block-3.metal`, and grade. If G1 then passes and G2 fails, the first
suspect is the MMA frag mapping (per-output 8-wide order vs MLX's
interleaved WM/WN assignment), diagnosed by the first differing
element's (row, expert slot, column) in the G2 report.
