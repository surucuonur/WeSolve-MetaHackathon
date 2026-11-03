# Hypothesis 1: fused grouped experts kernel

- **Owner:** lead
- **Status:** hardening
- **Version:** 6
- **Updated:** 2026-11-03T22:45:00Z
- **Next:** best is attempt-2 (only beats_compiled=true, bit-exact, boundary pass); builder-1-d hardening it with one change (1-row occupancy); boundary-1 checking attempt-1

## Inefficiency

Slot `experts`. One combined fix for five confirmed entries (savings do not add; joint ceiling 443 MB at streaming rate ≈ 11.5 ms at 32 rows, per the verifier):

- `ineff-quantized-kernels-1` (rank 1, mean 19.9% of step): per-pair qvm grids at ~1/3 peak; ideal 0.5 / 2.5 / 8.6 ms at 1/8/32 rows.
- `ineff-findings-2` (rank 3, mean 10.4%): measured 25% grouped-kernel win; 0.40 / 1.30 / 3.67 ms.
- `ineff-switch-layers-2` (rank 4, mean 4.3%): unfused products+activation; ~0.4 ms at all row counts.
- `ineff-runs-1` (rank 6, mean 4.2%): 465 MB excess re-reads; floor 0.0 / 0.6 / 2.3 ms.
- `ineff-switch-layers-1` (rank 7, mean 0.8%): sort glue 0.7 ms at 32 rows, removable only inside this fix.

Design: one Metal kernel that batches same-expert pairs with coalesced reads, runs gate/up/down plus the activation with intermediates on-chip, and needs no sort/unsort. Must keep the G2 boundary (2 bf16 ULP, fp32 accumulation, bf16 rounding points).

## Boundary

(written by boundary-1; every line carries its source)

### Inputs and outputs

Sources: the brief's `experts` slot section (`claude/muse-agent-brief.md:70-92`) and `bench/grader/slots.py:19-40,271`. Constants `D,E,K,H = 768,60,4,384` (slots.py:19; brief:26). All arrays row-contiguous (brief:24). `rows` is read from the first slot input's `shape[0]` (slots.py:271); any 1–32 in decode (brief:24).

| Name | Shape at 1 / 8 / 32 rows | Dtype | Layout / notes |
|---|---|---|---|
| `x` | [1,768] / [8,768] / [32,768] | bfloat16 | brief:74 |
| `inds` | [1,4] / [8,4] / [32,4] | uint32 | brief:75; cast to u32 on load, slots.py:219-220 |
| `gate_w`, `up_w` | [60,768,48], rows-independent | uint32 | [expert,in,out] transposed layout (brief:76; mobilemoe.py:42,202-214); 8 4-bit values per word along out; output column `o` sits in word `o/8` at bits `4*(o%8)` (brief:82) |
| `gate_scales`, `gate_biases`, `up_scales`, `up_biases` | [60,768,12] each, rows-independent | bfloat16 | one scale and bias per 32 outputs (brief:77); `w = scale*q + bias`, affine, group_size 32, bits 4 (brief:82; slots.py:67-77) |
| `down_w` | [60,384,96], rows-independent | uint32 | same packing as gate/up (brief:78) |
| `down_scales`, `down_biases` | [60,384,24] each, rows-independent | bfloat16 | one scale and bias per 32 outputs (brief:79) |
| `y` (output) | [1,4,768] / [8,4,768] / [32,4,768] | bfloat16 | brief:80; `y[r,j]` is expert `inds[r,j]`'s output, positional (see neighbour B) |

Weight bytes per layer (arithmetic shown): gate/up/down `_w` are each 60·768·48 = 60·384·96 = 2,211,840 words = 8,847,360 B; gate/up/down scales and biases are each 60·768·12 = 60·384·24 = 552,960 values = 1,105,920 B. Total ≈ 3·(8,847,360 + 2·1,105,920) ≈ 33.2 MB per layer, shared by all rows.
The spec's `slot_outputs` must provide `y` (slots.py:266-268). Grid counts threads, not threadgroups (slots.py:283).

### Before and after

Source: `MoE.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`) and the split MoE the grader installs (`bench/grader/slots.py:97-119`, bit-identical to the port per `check_identity`, slots.py:122-134). Order per layer: post_attention_layernorm → `x`; router → `inds`, `weights`; experts(`x`, `inds`) → `y`; shared_expert(`x`) → `shared` (independent of this slot, runs alongside); combine(`y`, `weights`, `shared`) → `out` → residual / next layer (mobilemoe.py:230-232,257-258; slots.py:105-118). The grader flattens model `[B,L,D]` to `[rows,D]` and restores shapes afterwards (brief:24; slots.py:106-117).

Neighbour A — what produces the slot's inputs (layernorm + routing), per tracking.md checklist:
- **Callers:** `MoE.__call__` in all 20 layers of every decode step (mobilemoe.py:217-230; brief:24).
- **Callees:** router Linear in float32, sigmoid, `argpartition(-biased, kth=3)`, `take_along_axis`, normalize `+1e-20` then `×2.5` (slots.py:44-48; mobilemoe.py:219-228).
- **Data:** `x` is bf16 RMSNorm output, flattened [rows,768]; `inds` is u32 [rows,4], 4 distinct experts per row in arbitrary order, values in [0,60) (slots.py:44-48; routing G2 rejects inds ≥ 60, check_candidate.py:65).
- **Neighbours:** this slot must accept `inds` in ANY order, including either side of a routing near-tie: within-row order is free (brief:47) and routing G2 matches expert sets, skipping rows whose 4th/5th biased scores are within 1e-6 (check_candidate.py:71-80).
- **Cost:** routing's own ~12 kernels/layer (brief:47) — not this slot's cost.
- **Invariants:** `inds` valid [0,60); `x` finite bf16; row count any 1–32 (brief:24), not just 1/8/32.

Neighbour B — what consumes `y` (combine + residual), per tracking.md checklist:
- **Callers:** combine is called as `combine(y.reshape(-1,K,D), weights, shared.reshape(-1,D))` (slots.py:116).
- **Callees:** `(y * weights[...,None]).sum(axis=-2).astype(bf16) + shared` (slots.py:51-52; mobilemoe.py:231-232).
- **Data:** `y` [rows,4,768] bf16 "in `inds` order" (brief:53); `weights` [rows,4] f32; `shared` [rows,768] bf16; `out` [rows,768] bf16 → reshaped to lead dims, added to the residual, fed to the next layer (slots.py:116-117).
- **Neighbours:** combine pairs `y[...,j,:]` with `weights[...,j]` POSITIONALLY (slots.py:52), so `y[r,j]` MUST be the output of expert `inds[r,j]` — no reordering of the K axis is visible downstream. Slot-edge traffic floor: `y` (rows·6144 B per layer) is written here and re-read by combine; ≈0.04 ms/step at 32 rows, unremovable within the interface (`territory-grader.md`, Boundary floor).
- **Cost:** combine's own 5 kernels/layer (brief:64) — not this slot's cost.
- **Invariants:** `y` must have the exact shape and bf16 dtype or G2 fails (check_candidate.py:117-118).

### What must stay identical

- **Reference math:** for each row `r` and `j`, with `e = inds[r,j]`: `y[r,j] = (silu(x[r] @ Wgate[e]) * (x[r] @ Wup[e])) @ Wdown[e]` (brief:84-88); port code `ref_experts` (slots.py:55-82) with `swiglu(gate,x) = silu(gate)*x` (activations.py:9-11), `gather_qmm(..., transpose=False, group_size=32, bits=4, mode="affine")` (slots.py:67-77).
- **Rounding points:** each gate/up/down product accumulates in float32 and rounds its output to bfloat16; `silu(gate)*up` is computed on those bfloat16 values and rounded to bfloat16 before the down product (brief:90). Keep fp32 accumulation (nothing narrower) and all four bf16 rounding points (carry no wider precision across them) — this hypothesis's stated G2 boundary.
- **Tie rule:** there is NONE inside this slot — instead the slot inherits routing's freedom: it must handle any 4 valid expert ids per row in any order, never assuming sorted `inds` or the reference `argpartition` order (brief:47; check_candidate.py:71-80).
- **Allowed reorderings:** any summation order inside fp32 accumulation; any grouping of (row, expert) pairs and any thread/block decomposition; sort or no sort at ANY row count including ≥16 (the reference sorts iff `inds.size >= 64`, slots.py:62, switch_layers.py:181, but G2 compares outputs only, so the choice is free — `territory-grader.md` note on ineff-runs-3); any dequantization order. **Forbidden:** moving any `y` element off its `[r,j,d]` position, changing shapes/dtypes, narrowing accumulation below fp32, skipping a bf16 rounding, or fusing across the slot edges (`x`/`inds` in, `y` out stay in DRAM between slots).

### What the grader checks, and what it can't see

Checks (`bench/grader/check_candidate.py`):
- **G1:** compiles and runs on real inputs at 1, 8 and 32 rows (`g1=True` after the loops, 206; any exception → error + g1 false, 227-228, Metal errors trimmed by `short_error`, 153-163). Note: shape/dtype mismatches are per-layer stats, not exceptions — they fail G2 with g1 true (188-191).
- **G2 experts** (`check_experts`, 112-130): exact shape [rows,4,768] and bf16 dtype (117-118); EVERY element within 2 bf16 ULP of the reference, with the ULP floored at |r| = 0.001 so reorderings near zero don't fail (`ulp_bf16`, 44-47; 119-123). All 20 layers' captured inputs at each of 1/8/32 rows must pass (179-205). Inputs are seeded (`mx.random.seed(rows)`, slots.py:158) — deterministic every run.
- **Piece timing** runs only if G2 passes (209-226): median of 30 repeats of all-20-layer evals, per layer; `beats_compiled` needs candidate < compiled reference at ALL three row counts. "It is a guide; G3 decides" (brief:193).
What it can't see (checked by hand later — boundary.md *check an attempt*):
- Row counts other than 1, 8 and 32: any 1–32 occur in decode (brief:24); grids and indexing must be valid for all of them.
- Inputs beyond the captured set: one seeded run of 128-token random prompts, decode step 8 (slots.py:158-168). Real prompts, steps and activation distributions differ — no assumption fitted to the captured values survives.
- Reads of threadgroup memory never written, results that depend on thread timing, out-of-bounds reads that luckily pass, threadgroup memory over 32 KB (`build` enforces only the thread COUNT, slots.py:274-276, not the 32 KB).
- End-to-end and G3, run after merge (`time_candidates.py`): the same greedy tokens over 24 generated tokens at 1 and 32 requests in plain MLX and vLLM (brief:185), and a lower decode-step period than both the original and the compiled reference with the gap exceeding the repeat spread (brief:186).

### The GPU limits

- At most 1,024 threads per threadgroup (brief:98; count enforced in slots.py:275) and 32 KB of threadgroup memory (brief:98; NOT enforced by `build` — verify by hand from the Metal source).
- Grid counts threads (slots.py:283); sizes are ints or products of ints and rows/D/E/K/H only (slots.py:230-240); dtypes float32/bfloat16/float16/uint32/int32 (slots.py:22-28).
- Only `mx.fast.metal_kernel`, MLX 0.32.4; the grader runs no Python of the candidate's — the spec is data, only the Metal compiles (brief:96-97). A grader run should take under 3 minutes (brief:98).
- Hardware: Apple M1 Pro GPU, 16 cores, SIMD width 32, bfloat16 as `bfloat` in Metal (brief:99).

## Builders

### builder-1-b — direction (b): fusion first, keeping per-pair reads

One threadgroup per (row, expert-slot) pair; gate/up share one K-loop over an
on-chip x row; gate/up/hidden in threadgroup memory (3840 B); weights read
once per pair, no batching, no sort. Numerics replicate the reference
bitwise: qvm strided-lane accumulation + `simd_sum` per 32-output block, and
silu computed ENTIRELY in bf16 (round after exp/add/recip/sub/mul/mul —
found by CPU search over all 64 rounding placements, only full-bf16 matches).

- attempt-2 (`muse-spark/kernels/experts/attempt-2`, v3): predicted 30 / 185 /
  470 µs per layer at 1/8/32 rows; measured 107.3 / 188.4 / 647.6 (original
  99.8 / 218.7 / 768.6; compiled 135.0 / 216.9 / 761.7). G1 pass; G2 pass
  with max_ulp 0.0 at all rows (bit-exact); beats_compiled true at all rows.
- Taught: (1) down outputs near cancellation amplify hidden 1-ULP flips, so
  gate/up/hidden must match bitwise — any summation order will NOT do;
  (2) MLX's compiled swiglu rounds every op to bf16 (all other builders must
  replicate this); (3) rows-1 is occupancy-bound (512 threads, 20 GB/s
  effective), not streaming — fusion alone regresses 7.5% vs eager there;
  (4) no cross-pair cache reuse at 32 rows (full 70.8 MB/layer @ ~109 GB/s).

### builder-1-c — direction (c): TWO-kernel split with grouped same-expert reads

Kernel 1 fuses gate+up+silu-mul, kernel 2 the down product; one threadgroup
per expert (60 groups) finds its pairs by scanning `inds` and streams each
weight tile once per 2-pair chunk (~34 MB/layer at 32 rows vs 71 MB
per-pair). Final form (v7): 192/384 threads, 2 outputs per thread (an even
pair shares one weight word + one scale/bias), branchless padded chunks,
K unrolled x2 with even/odd accumulators. Tmem ~16.1 KB per kernel by hand.
Numerics: MLX's stable bfloat Sigmoid bit-exact
(`y=1/(1+precise::exp(abs(x)))`, `(x<0)?y:1-y`, per-op bf16 rounds — read
from libmlx.dylib, verified on CPU against all 2.5M captured outputs).

- attempt-3 (`muse-spark/kernels/experts/attempt-3`, v7): predicted 25 / 85 /
  210 µs per layer at 1/8/32 rows; measured 68.1 / 238.4 / 654.1 (original
  70.0 / 214.7 / 769.8; compiled 64.9 / 213.1 / 764.0). G1 pass; G2 pass
  with max_ulp 1.0 at all rows; beats_compiled false (wins 32 rows ~15%,
  misses 1 row by 5% / 8 rows by 12%).
- Taught: (1) dynamic-index accumulator arrays spill to stack (~10x slow);
  unroll with scalars; (2) per-iteration guards + 8x loop bloat eat all
  tile-sharing gains — chunk-2 padded branchless wins because pairs/expert
  is always ~1-2.4; (3) tiled load→barrier→compute halves DRAM duty cycle
  (~55 GB/s achieved vs 150 ideal); (4) at 8 rows grouping's DRAM edge
  barely outweighs padding waste — hardening should hybridize with
  attempt-2's per-pair win (188 µs at 8 rows).

### builder-1-a — direction (a): full fused grouped kernel (ONE kernel)

One threadgroup per expert (60 groups) scans `inds` for its pairs and runs
gate/up/down + silu-mul fused with intermediates on-chip, no sort/unsort.
v1-v9 replicated the reference bit-exactly (qvm strided lanes + `simd_sum`,
per-op bf16 silu derived from MLX source); v10+ reused attempt-3's
G2-proven even/odd-split + tmem-tile forms. Final (v14): 192 threads,
chunks of 2 pairs, gate/up = attempt-3 k1 shape, down = k2 per-thread
shape over 2 output blocks. Tmem ~17.1 KB by hand.

- attempt-1 (`muse-spark/kernels/experts/attempt-1`, v14 final of v1-v15):
  predicted 35 / 100 / 220 µs per layer at 1/8/32 rows; measured 79.3 /
  260.3 / 743.7 (original 64.1 / 214.4 / 767.1; compiled 62.0 / 214.1 /
  761.0). G1 pass; G2 pass with max_ulp 1.0, bad 0; beats_compiled false
  (beats compiled only at 32 rows). Full version log + analysis in
  `attempt-1/notes.md`.
- Taught: (1) bit-exact qvm replication passes G2 at max_ulp 0.0 but runs
  ~1/2.3 reference throughput — the strided-lane structure is unfixable
  (threads/unroll/scalars/coalescing all verified neutral by grade);
  (2) G2 does NOT need bitwise products: even/odd split + plain dequant +
  bfloat silu passes (max 1.0), and all three builders' v1s failed
  identically (max 11/32/68 ULP) on the five bf16 rounding points of the
  compiled swiglu alone; (3) wider sharing backfires: C=4 (+21-50%) and
  384 threads cost more in tmem reads/pressure than they share —
  2-pair chunks at 192 threads are the sweet spot; (4) vs attempt-2
  (107.3/188.4/647.6): fusion-without-grouping wins at 8/32 rows because
  grouping's byte savings don't convert in the piece harness (cache
  absorbs re-reads) while scan/chunk/barrier/phantom overhead is real —
  hardening should start from attempt-2 and add grouping only on G3
  cache-pressure evidence.

## Boundary checks

- **attempt-2 — pass** (boundary-1): one fused per-pair kernel, G2 bit-exact. Checked
  `muse-spark/kernels/experts/attempt-2/block-0.metal` (B0), `block-1.metal` (B1),
  `spec.json` against the five unseen-boundary items; no defect found.
  - Other row counts: grid `rows*512` threads ÷ 128 = `rows*4` groups, one per
    (row, slot) pair for any rows 1–32 (spec.json). `pair` in [0,rows*4)
    (B0:7); `r = pair>>2` in [0,rows) (B0:11); `inds[pair]` of rows*4 (B0:12),
    `x[r*768+i]` of rows*768 (B0:20), `y[pair*768+b*32+j]` of rows*4*768
    (B0:70) all in range; no hardcoded row count, no sort threshold.
  - Unwritten threadgroup reads: none. `tg_x` fully written by the strided loop
    (B0:19-21) before the barrier (B0:22) and any `qvm_block` read (B1:21);
    `tg_gate`/`tg_up` get all 12 blocks exactly once (`b = 4v+simd`, v in 0..2,
    B0:29-47) before the barrier (B0:48) and silu reads (B0:54,60); `tg_h` fully
    written (B0:53-61) before the barrier (B0:62) and down reads.
  - Threadgroup memory: 768+384·3 = 1920 bf16 = 3840 B (B0:14-17), ~12% of the
    32 KB limit; 128 threads/group ≤ 1024 (spec.json). Far from both limits.
  - Thread timing: all cross-thread sharing guarded by barriers (B0:22,48,62);
    each `tg_gate`/`tg_up` block and each `y` element written exactly once
    (down `b = 4v+simd`, v in 0..5, covers 0..23 once, B0:65-73); pairs in
    separate groups write disjoint `y` slices; no atomics. `simd_sum` (B1:39)
    runs under uniform control flow (K = 768/384 both divisible by 32).
  - Captured-input assumptions: none. `e = inds[pair]` used directly (B0:12) —
    any order, duplicates fine; `e` in [0,60) is the shared boundary invariant,
    and index maxima fit exactly (gate/up W 36863 < 36864, S/B 9215 < 9216;
    down W 36863, S/B 9215; B1:22-30). No value early-exits. Residual, not a
    defect: the bf16 silu lowering (B0:53-61) matches by 64-placement search
    plus bit-exact G2 over all layers, so an unseen gate value flipping hidden
    by 1 ULP (amplifiable by down near-cancellation) can't be ruled out in
    principle — but no input class is assumed away. Cosmetic only: the B0:6
    header comment still says "fp32" though v3 computes silu in bf16.
- **attempt-3 — pass** (boundary-1): two-kernel grouped split, G2 max 1.0 ULP.
  Checked `muse-spark/kernels/experts/attempt-3/block-0.metal` (K1, gate/up/act),
  `block-1.metal` (K2, down), `spec.json` against the five unseen-boundary
  items; no defect found.
  - Other row counts: grid is rows-independent (60 groups, one per expert;
    spec.json). `P = rows*4 <= 128` (K1:16-17, K2:8-9); `match_`/`plist` hold
    128 (K1:19-20, K2:11-12), and `o < P` writers always exist (192/384
    threads). `plist` build reads only `t < P` (K1:36, K2:25); chunk reads
    `plist[c0(+1)]` guarded by `n` (K1:55-56, K2:44-45); any `Pe` 0..P works,
    empty experts skip the loop, odd tails pad with pair 0 and skip the store
    (K1:136, K2:95). All DRAM indices fit exactly at every rows 1–32: gate/up
    W < 2,211,840 and S/B < 552,960 (K1:67,76), down W < 2,211,840 and S/B <
    552,960 (K2:54,61), `x` row `t>>2 < rows` (K1:87), `hidden` `t*384+o0+1`
    (K1:127-146), `y` `t*768+o0+1` (K2:93-98); threadgroup tile maxima also
    exact (K1 gW 1535<1536, gS 383<384, xT 63<64; K2 dW 3071<3072, dS
    767<768). No hardcoded row count, no sort threshold.
  - Unwritten threadgroup reads: none. `match_` entries read only for `t < P`,
    all written (K1:30-31, K2:19-20); `plist`/`nP` written before the barrier
    (K1:34-44, K2:23-33) and read after; every tile fully written each
    `(chunk, i0)` iteration (K1:63-88 cover 1536/384/64 exactly once;
    K2:50-71 cover 3072/768/64) before the compute barrier (K1:89, K2:72),
    with the end-of-tile barrier (K1:120, K2:91) guarding the next overwrite.
  - Threadgroup memory: K1 = 512+512+6144+6144+4·768+128+4 = 16,516 B;
    K2 = 512+512+12288+2·1536+128+4 = 16,516 B (matches notes.md ~16.1 KB);
    both ~52% of the 32 KB limit with clear margin. 192/384 threads per
    group ≤ 1024 (spec.json). Far from both limits.
  - Thread timing: every barrier is in uniform control flow (`Pe`, `n`, `c0`,
    `i0` uniform; `o`-guarded writes sit outside barriers); each (pair,
    output) of `hidden` and `y` written exactly once (each pair `t` belongs
    to exactly one expert's list since `inds[t]` is single-valued; each `o`
    owns a disjoint output pair); the two kernels sequence through DRAM
    `hidden`; no atomics, no `simd_` ops.
  - Captured-input assumptions: none. `inds` used directly in any order
    (K1:31, K2:20); duplicates within a row still land in distinct positional
    outputs; any pairs-per-expert 0..P; no value early-exits or thresholds.
    Residual, not a defect: gate/up/down use an even/odd split-K fp32
    summation order that is not bit-exact (G2 max 1.0 ULP), but any fp32
    summation order is boundary-allowed, and all four bf16 rounding points
    plus MLX's stable bf16 sigmoid lowering are kept — 1 ULP of margin to
    the 2-ULP contract over all captured layers/rows.

## Decision

(open)

## From other agents
