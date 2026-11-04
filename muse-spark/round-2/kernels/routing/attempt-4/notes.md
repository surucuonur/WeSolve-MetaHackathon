# Round-2 attempt 4 — builder-2-d, hardening of hyp-2-routing-exact

Builds on `muse-spark/round-2/kernels/routing/attempt-3` (scalar
full-G2; chained 46.9/300.0/297.2 µs/layer vs compiled 58.0/84.5/90.9 —
latency-bound from uncoalesced device loads + `xsh` bank conflicts).
Baseline: none, so no `beats_baseline`; the bar is the compiled reference.

## The one main change

Stage each K-chunk's `w_router` block into threadgroup memory with
coalesced vector loads and sum from threadgroup with a padded `x`
layout, nothing else:

- Staging: chunks are staged 4 at a time (12 groups cover all 48
  BK=16 chunks). Each staged chunk is 60×16 floats, loaded as 240
  `float4` loads (consecutive threads cover one expert's 4 vectors,
  then the next expert) and stored transposed in `wsh[slot][k][expert]`
  with expert stride 64.
- Padded `x`: chunk `c` of `x` sits at `xpad[c*17 + k]` (one pad float
  per chunk), replacing attempt-3's flat `xsh[768]`.
- Scalar chain: expert-parallel lanes read `wsh` at consecutive expert
  banks (conflict-free) and `xpad` uniformly per `k` (broadcast).
  Partials live transposed in `parts[partition][expert]` (stride 64)
  and are resumed product-by-product in increasing `d` across staged
  groups, so each (expert, partition) partial is still one serial
  scalar sum in increasing `d`; partials are then summed sequentially
  in partition order. 24 barriers on the scalar path (2 per group).
- Threadgroup memory: (816 + 60 + 2048 + 4096) × 4 B = 28,080 B of
  32 KB. Same grid (`rows*256`), threadgroup (256) and one-kernel layout.

Kept verbatim from attempt-3: the regime branch on `rows` (GEMV only
at `rows == 1`; split-K while `rows <= 768`; regular above — the GEMV
branch is unchanged apart from reading the padded `x` layout), the
split-K partition formula (32 partitions at 8/32 rows, 8 of 96 at 127,
2 at 257–768, single partition above 768, last partition absorbing the
remainder chunks), the GEMV lane order and shuffle tree 16,8,4,2,1,
MLX's verbatim sigmoid (`1/(1+precise::exp(abs(s)))`, `(s<0)?y:1-y`),
the strictly-greater insertion top-4 (earliest index wins ties), and
`((w0+w1)+w2)+w3 + 1e-20`, divide, ×2.5 on unbiased scores.

## Inefficiency removed

`ineff-baseline-routing-1` stays removed (same orders as attempt-3).
The target is attempt-3's diagnosed feeding cost (its notes: the
scalar chains read `w_router` uncoalesced — warp lanes 64 B apart,
16 cache lines per step — and `xsh[p*16+k]` with 16-way bank
conflicts on every product). Staging turns the device reads into
coalesced `float4` loads and the dependent chain's reads into
conflict-free threadgroup reads. The per-row `w_router` re-read
(`ineff-baseline-routing-4`) and the serial thread-0 scan are
untouched.

## Predicted µs per layer at 1 / 8 / 32 rows

Anchor: attempt-3 measured 46.9/300.0/297.2, compiled 58.0/84.5/90.9
(same grader run). Traffic per row is unchanged at 184 KB of
`w_router` (0.9 µs at 200 GB/s), so floors are 0.9/7.4/29.5 µs; the
rest is chained-call latency and per-row work. Attempt-3's flat
~300 µs was the dependent device-load chain; the staged chain reads
threadgroup memory instead, at the cost of 24 barriers/row.

- 1 row: GEMV branch unchanged but for the padded index (a shift and
  mask per access) → ≈ attempt-3's 46.9 → **48**.
- 8 rows: fixed chained latency ≈ 40 µs (attempt-3's 1-row time less
  its small GEMV work) + traffic 7.4 + staged scalar work. The
  critical path is the last partition's 272 sequential threadgroup
  FMAs, overlapped across 8 threadgroups; allowance ≈ 20 µs →
  40 + 7.4 + 20 ≈ **67**.
- 32 rows: 40 + 29.5 + 25 (more threadgroups fill the GPU, critical
  path unchanged) ≈ **95**.

Prediction: **48 / 67 / 95 µs** — beats compiled (58.0/84.5/90.9) at
1 and 8 rows, near it at 32. G2 prediction: pass at all six sizes —
no accumulation order changes, only where operands are read from.

## Verdict (2026-11-04T03:56:12Z, one grader run)

G1 pass, **G2 pass at all six sizes** (rows_differ 0,
rows_other_experts 0, max_rel_err 0.0 at 1/8/32 and 127/1016/2048).
Chained µs/layer (ref / compiled / candidate): 1 row 71.1 / 60.7 /
**47.5**; 8 rows 102.9 / 84.9 / **69.1**; 32 rows 104.9 / 89.5 /
**72.9**. **`beats_compiled` true** — the first routing attempt to
beat the compiled reference at all three row counts. Baseline none,
so no `beats_baseline`. Vs attempt-3 (46.9/300.0/297.2): the staged
scalar path is 4.3×/4.1× faster at 8/32 rows; the 1-row GEMV branch
is unchanged (47.5 vs 46.9, drift).

## Why the 32-row timing missed the prediction by >20%

Predicted 48/67/95, measured 47.5/69.1/72.9: 1 and 8 rows hold
(−1%/+3%), 32 rows is 23% under. The prediction added a fixed ~40 µs
chained latency, the full 29.5 µs traffic floor and a per-row work
allowance as if they were serial per layer call. They are not: at 32
rows, 32 threadgroups across 16 GPU cores overlap one threadgroup's
coalesced staging loads and barriers with other threadgroups' scalar
compute, so the marginal cost of rows 9–32 is only (72.9−69.1)/24 ≈
0.16 µs/row, against the predicted ≈1.2 µs/row. The kernel is
essentially flat 69→73 µs from 8 to 32 rows — once enough
threadgroups are resident to hide the staging latency, extra rows
are nearly free, and the 28 KB threadgroup footprint (which caps
residency) does not bind at these row counts. Traffic-floor
arithmetic overstates the cost whenever staging and compute overlap
across threadgroups.

Next change (one): none needed for the slot's stop rule (G1+G2+
beats compiled at all three row counts). If pushed further, cut the
24 barriers by staging 8 chunks per group (halves barrier count but
needs ~2× the `wsh` footprint or a narrower expert stride).
