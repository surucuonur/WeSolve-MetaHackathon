# attempt-1 (round 2): branched fused experts, 512 threads/pair (builder-1-b)

Direction (b) on starting point `muse-spark/kernels/experts/attempt-4`
(baseline `none` — no earlier attempt passes the current G2). One fused
kernel, one threadgroup per (row, expert-slot) pair, branched in-Metal
on `rows >= 60` (`x_shape[0]`, the `gather_qmm_rhs` dispatch threshold,
not the grader's bracketing sizes):

- **Decode (`rows < 60`):** attempt-4's qvm arithmetic unchanged —
  `qvm_block` copied verbatim (strided lanes `k ≡ lane (mod 32)`,
  float32 dequant `x·(scale·q+bias)`, `simd_sum`), so every block's
  accumulation order is identical. The one change is width: 512
  threads/pair (16 simdgroups) instead of 256, so gate/up run their 12
  blocks in 1 wave (was 2) and down its 24 blocks in 2 waves (was 3):
  4 simd waves instead of 7, more blocks in flight per call. This is
  the ineff-baseline-experts-2 fix (1-row chained latency): at 1 row
  the call still has only 4 threadgroups, but each now keeps up to 16
  blocks in flight instead of 8.
- **Prefill (`rows >= 60`):** per-pair path with `gather_qmm_rhs`
  arithmetic — each weight dequantized to bfloat16 first
  (`bfloat(scale·q+bias)`, `qmm_accum` in block-1) and K accumulated
  contiguously in BK=32 tile order by a single thread per output (one
  output per lane, no `simd_sum`), mimicking BlockMMA's sequential
  tile order. Same silu (full-bfloat16, attempt-4's) and the same four
  bfloat16 rounding points (gate, up, hidden, down).

Removes: ineff-baseline-experts-1 (the G2 gate — the branch supplies
the prefill arithmetic attempt-4 lacks) and attacks
ineff-baseline-experts-2 (1-row latency) without touching any block's
`simd_sum` order. ineff-baseline-experts-3 (per-pair re-reads) is
deliberately kept — round-1 evidence says grouped reads lose at this
duty cycle; per-pair reads at/above 60 rows are correctness-only cost
(prefill is untimed).

Limits: 512 threads/group ≤ 1,024; threadgroup memory unchanged at
3,840 B (tg_x 768 + tg_gate/up/h 384×3 bfloat16) ≤ 32 KB. Valid at
every `rows` (grid `rows*2048` threads = `rows*4` groups of 512; the
branch is on the same `rows` the grid is built from).

## Predicted µs per layer at 1 / 8 / 32 rows (chained, grader timing)

Starting point chained (console-chain.txt, via ineff-baseline-experts-2):
132.9 / 175.9 / 587.6; reference chained 80.3 / 239.6 / 797.6. Bytes per
pair 552,960 B → 2.21 / 17.69 / 70.78 MB per layer at 1 / 8 / 32 rows.

- **1 row: 85.** Latency-bound: attempt-4's 132.9 ≈ ~20 µs fixed
  (launch/link, x load, silu, barriers) + ~113 µs wave-serial; waves
  7 → 4 scale the serial part: 20 + 113×4/7 ≈ 85. Roughly the
  reference's 80.3 — closing most of the +52.6 gap.
- **8 rows: 165.** Streaming: 17.69 MB at ~107 GB/s (attempt-4 chained
  ran 100.6 GB/s; 16 in-flight blocks/group should add a little
  memory-level parallelism) → ~165.
- **32 rows: 565.** 70.78 MB at ~125 GB/s (attempt-4: 120.5 GB/s) →
  ~566. Risk noted: 512-thread groups halve resident groups per core;
  if occupancy binds instead, this lands nearer 590.

G2 prediction: decode bit-exact (identical per-block math). Prefill
bit-exact only if BlockMMA's 8-wide fragment dots round like serial
float adds in BK order — territory-quantized-kernels §E warns they may
not; the dequant rounding (the larger of the two divergences) is
matched exactly either way.

## Verdict

G1 pass; **G2 pass at all six sizes, differ 0 / max_ulp 0.0 everywhere**
(decode 1/8/32 and prefill 127/1,016/2,048) — the branch works: decode
keeps attempt-4's bits, and the scalar prefill path (bfloat16 dequant +
serial BK=32 tile order) matched `gather_qmm_rhs` bit for bit on every
tested layer. Chained timing µs/layer (this run; candidate vs reference
vs compiled): **1 row 104.3 / 87.6 / 77.6 (loses both)**; 8 rows
193.3 / 231.2 / 235.7 (wins); 32 rows 535.7 / 820.6 / 791.6 (wins).
`beats_compiled` false — only the 1-row size misses. No baseline this
round (`none`), so no `wins_vs_baseline`/`beats_baseline`.

Against the starting point's chained numbers (132.9 / 175.9 / 587.6,
different harness — indicative only): −22% at 1 row, +10% at 8 rows,
−9% at 32 rows.

## Miss analysis (prediction 85 / 165 / 565, measured 104.3 / 193.3 / 535.7)

1 row missed by +23% (over the 20% bar); 8 rows +17%; 32 rows −5%
(fast side). The wave-count model was wrong about what dominates at
1 row. Cutting waves 7 → 4 did help (132.9 → 104.3 vs the starting
point), but the residual is not wave-serial: with 4 pairs there are
only 4 threadgroups, so at most 4 of the 16 GPU cores can hold work no
matter how wide each group gets — the reference spreads 1 row over
24+24+48 small groups. The fixed remainder (launch/link on the chain,
tg_x load, 512-thread barriers, lane-0's serial 32-output stores per
block) also does not shrink with waves. Conclusion: threads-per-pair
widening is nearly exhausted as the 1-row lever (256 → 512 bought
~29 µs chained); the binding limit is threadgroups per call, which a
single fused per-pair kernel cannot raise — down needs the pair's full
hidden vector, so blocks cannot be split across groups without a
device-memory intermediate. The 8-row +17% miss has the same shape in
milder form: 512-thread groups halve resident groups per core vs
256-thread groups, costing more at 32 groups than the extra in-flight
blocks repay; at 128 groups (32 rows) the trade turns positive again.
