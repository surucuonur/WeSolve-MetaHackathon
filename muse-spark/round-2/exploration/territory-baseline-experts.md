# Territory: baseline-experts

- **Owner:** explorer-baseline-experts
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T03:18:53Z
- **Next:** none — checklist filled, 3 inefficiencies posted, grader note folded

## What it does

The `experts` slot of one MoE layer: for each row `r` and slot `j`, with
`e = inds[r, j]`, compute
`y[r, j] = (silu(x[r] @ Wgate[e]) * (x[r] @ Wup[e])) @ Wdown[e]`
(`claude/muse-agent-brief.md:84-88`; reference `bench/grader/slots.py:55-82`).
The starting point `muse-spark/kernels/experts/attempt-4` is now the code in
the slot: one fused per-pair Metal kernel
(`block-0.metal` + header `block-1.metal`, launched by `spec.json`) that
replicates MLX `affine_gather_qvm` bitwise. It is bit-exact at decode sizes
and fails the new G2 at every prefill size, and it loses chained at 1 row
while winning chained at 8/32 rows. Round 1 mapped the same code path in
`muse-spark/exploration/territory-switch-layers.md` and
`territory-quantized-kernels.md`; this entry cites them where they still
hold and writes only what is new: the starting point as slot code, the
chained-timing reversal at 1 row, and the prefill kernel switch.

## Checklist

### A. Starting point `block-0.metal` — fused per-pair kernel body

- **Callers:** `spec.json` kernel `fused_experts_perpair` (see §C), built by
  `bench/grader/slots.py:303-345` (`build`) into one `mx.fast.metal_kernel`
  call per layer invocation. In the model it replaces `SwitchGLU.__call__`
  via `slots.make_moe_call` (`slots.py:97-119`): `experts(flat, inds, ...)`
  with `flat` `[rows, 768]` bfloat16 and `inds` `[rows, 4]` uint32.
  Shapes: decode 1/8/32 rows → pairs `rows*4` = 4/32/128; prefill
  127/1,016/2,048 rows → pairs 508/4,064/8,192 (prefill sizes from
  `slots.py:214-254`, `capture_prefill`: 128-token prompts, 2,048-token
  chunks). The grader tests all six sizes for G2
  (`bench/grader/check_candidate.py:272-276`) and chains the piece timing
  only at decode 1/8/32 (`check_candidate.py:309-337`).
- **Callees:** `qvm_block` (§B) for gate, up, down; no MLX op. Thread 0 of
  each simdgroup stores each 32-output block; silu-mul is inline
  (`block-0.metal:53-64`).
- **Data:** inputs `x` `[rows, 768]` bfloat16, `inds` `[rows, 4]` uint32,
  `gate_w`/`up_w` `[60, 768, 48]` uint32, `gate_scales`/`gate_biases`/
  `up_scales`/`up_biases` `[60, 768, 12]` bfloat16, `down_w` `[60, 384, 96]`
  uint32, `down_scales`/`down_biases` `[60, 384, 24]` bfloat16
  (`bench/grader/slots.py:33-36`; brief `experts` table:
  `claude/muse-agent-brief.md:73-80`). Threadgroup: `tg_x[768]`,
  `tg_gate[384]`, `tg_up[384]`, `tg_h[384]` bfloat = 1,920 values = 3,840 B
  (`block-0.metal:13-16`). Output `y` `[rows, 4, 768]` bfloat16, positional:
  `y[pair * 768 + ...]` with `pair = threadgroup_position_in_grid.x`,
  `r = pair >> 2`, `e = inds[pair]` (`block-0.metal:6-11,68-75`).
- **Neighbours:** before: routing produces `inds` (any order, values in
  `[0, 60)`); after: combine consumes `y` positionally as
  `y[r, j]` paired with `weights[r, j]` (`slots.py:116`,
  `ref_combine` `slots.py:51-52`). The kernel must accept any `inds` order
  and duplicates; it does (`e = inds[pair]`, no sort, no threshold).
  Shared expert runs independently and is outside the slot.
- **Cost:** 1 kernel per layer call (vs reference 7 at 1 row, 13 at 32 rows;
  see §E). Grid `rows*1024` threads, threadgroup 256 → `rows*4`
  threadgroups, one per pair (`spec.json`; `block-0.metal:1-5`). At 1 row:
  4 threadgroups / 1,024 threads. Weight bytes per pair 552,960 B
  (gate/up/down: `60·768·48 = 60·384·96 = 2,211,840` words per projection;
  per expert per projection 36,864 words = 147,456 B, plus scales+biases
  `9,216 × 2 B × 2 = 36,864` B; total `3 × 184,320 = 552,960` B; cf.
  `attempt-4/notes.md:19` "553 KB", `hyp-1-fused-experts.md:39`). Per layer:
  2.21 MB at 1 row (4 pairs), 17.7 MB at 8 rows (32 pairs), 70.8 MB at
  32 rows (128 pairs). Measured chained (median µs/layer,
  `results/runs/2026-11-04-muse-round-1/console-chain.txt` via
  `bench/micro/chain_timing.py`): candidate 132.9 / 175.9 / 587.6 vs
  reference 80.3 / 239.6 / 797.6 at 1/8/32 rows. Independent (overlapped)
  same run: candidate 61.0 / 165.1 / 535.0 vs reference 86.0 / 219.1 /
  767.3. At 200 GB/s peak (`claude/GLOSSARY.md:10`) the byte floors are
  11.1 / 88.4 / 353.9 µs/layer; the candidate's effective rates are
  16.6 / 100.6 / 120.5 GB/s chained (2.21/17.7/70.8 MB ÷ chained µs).
- **Invariants:** per-block numerics must stay bit-identical to the
  reference kernel that runs at that size (see §F): float32 accumulation,
  bfloat16 rounding after each of gate/up, after `silu(gate)*up`, and after
  down (`claude/muse-agent-brief.md:90`). `block-0` keeps all four: gate/up
  stored as `(bfloat)p[j]` (`:34,46`), hidden stored as bfloat (`:63`),
  down stored as `(bfloat)p[j]` (`:73`). Silu is computed entirely in
  bfloat16, rounding after every op (`:56-64`; found by search, see
  `attempt-4/notes.md:22-26` and `hyp-1-fused-experts.md:94-97`). No rows
  threshold in the kernel: valid for any `rows` (indices `r`, `inds[pair]`,
  `y[pair*768+...]` all scale with `rows`).

### B. Starting point `block-1.metal` — `qvm_block`, the `gather_qvm` replica

- **Callers:** `block-0.metal:31,43,70` (gate, up, down). Byte-identical to
  attempt-2 (`attempt-4/notes.md:6-9`; `hyp-1-fused-experts.md:302-306`).
- **Callees:** `simd_sum` reduction (`block-1.metal:37-40`). No other call.
- **Data:** per call one 32-output block: `W` uint32 stream, `S`/`B`
  bfloat16 scales/biases, `xv` threadgroup bfloat row (`K` = 768 gate/up,
  384 down), `G` groups (12 gate/up, 24 down), `Ww` words per row (48/96),
  block index `g`, word base `wb = b*4`, `lane` in `[0, 32)`. Output
  `p[32]` float32 per lane, reduced by `simd_sum` so lane 0 holds the block
  result (`block-1.metal:17-40`). Dequant on the fly in float32:
  `p[2i] += x * (s0 * q + bias)`, `p[2i+1] += x * (s1 * q + bias)` with
  `s0 = scale`, `s1 = scale/16` (`:24-25,33-34`), matching MLX `qouter`
  bits=4 path (round 1: `territory-quantized-kernels.md` §A,
  `quantized.h:452-553`, `:503-509`).
- **Neighbours:** reads `tg_x` (gate/up) or `tg_h` (down) after a barrier
  (`block-0.metal:21,65`); writes only via the caller's lane-0 store.
- **Cost:** per pair: gate/up 12 blocks, down 24 blocks, each block
  `K/32` = 24 (gate/up) or 12 (down) strided lane steps × 32 outputs.
  Included in §A totals; launches no kernel itself.
- **Invariants:** this is the exactness core. Each lane accumulates its
  strided `K` elements serially (`k = lane; k < K; k += 32`, `:20`), then a
  `simd_sum` tree reduces across lanes (`:38-39`) — the same order as MLX
  `qvm_impl` (`quantized.h:1275-1321` in the installed 0.32.4 headers;
  round 1 `territory-quantized-kernels.md` §A). Round 1 proved any other
  summation order fails: down outputs near cancellation amplify a 1-ULP
  hidden flip (`hyp-1-fused-experts.md:103-108`,
  `ineff-switch-layers-2.md` builder note). The guard `b < 12` in gate/up
  wave 2 is simd-uniform (`v`, `simd` only), so participating simdgroups
  reduce uniformly and idle ones skip wholly (`hyp-1-fused-experts.md:324-330`).

### C. Starting point `spec.json` — launch spec

- **Callers:** grader `load` (`check_candidate.py:57-64`) → `slots.build`
  (`slots.py:303-345`). One kernel entry.
- **Callees:** `fused_experts_perpair`, `source_block 0`, `header_block 1`,
  inputs all 11 slot inputs, output `y` `[rows, 4, 768]` bfloat16, grid
  `["rows*1024", 1, 1]`, threadgroup `[256, 1, 1]`, no template, no
  atomics (`spec.json` single line). `slot_outputs: {"y": "y"}`.
- **Data:** grid counts threads (`slots.py:334`, brief:175): `rows*1024 ÷
  256 = rows*4` threadgroups at every row count, including prefill
  (8,192 threadgroups / 2,097,152 threads at 2,048 rows). Threadgroup
  limits: 256 ≤ 1,024 threads, 3,840 B ≤ 32 KB (`claude/muse-agent-brief.md:98`).
- **Neighbours:** kernels run in list order (`slots.py:325`); with one
  kernel there is no inter-kernel ordering to keep. A prefill branch (see
  ineff-1) must fit this surface: grid expressions are only products of
  integers and `rows`/`D`/`E`/`K`/`H` (`slots.py:279-292`), so a size
  switch can only be a second listed kernel with an in-Metal early return,
  or a branch inside `block-0` on a shape the source uses
  (`x_shape` is available when used; brief:96).
- **Cost:** 1 launch per layer per call. Fixed cost per launch ~6.81 µs
  end-to-end (round 1: `muse-spark/exploration/README.md:22`, Claude's
  `step_parts.txt` unit) — the candidate pays it once where the reference
  pays it 7–13 times, before any byte time.
- **Invariants:** `y` shape/dtype checked exactly (`check_candidate.py:158-159`);
  threadgroup product ≤ 1,024 enforced (`slots.py:327-328`); 32 KB
  threadgroup memory not enforced by `build` — verified by hand above.
  G2 tests only six sizes (grader note below): row counts between them
  are uncertified by the grader, but the kernel must still be valid for
  every rows the model can call (brief decode 1–32, prefill chunks).

### D. Starting point `notes.md` — what attempt-4 changed and measured

- **Callers/Callees:** none (documentation). Records the single change on
  attempt-2: 256 threads/pair (8 simdgroups) instead of 128, gate/up 12
  blocks in 2 waves of 8 (4 simds idle in wave 2), down 24 blocks in 3
  waves, x/silu strided loops step 256 (`notes.md:3-9`). `block-1.metal`
  byte-identical to attempt-2.
- **Data/Cost (its own run, old independent timing):** predicted
  60/185/650 µs/layer at 1/8/32 rows; measured candidate 50.0/157.8/515.4
  vs original 64.3/218.6/765.7, compiled 65.3/217.4/760.2 (`notes.md:46-52`).
  G2 bit-exact `max_ulp 0.0` at all decode rows under the old grader.
  The 1-row baselines shift between runs (64.3/65.3 here vs 99.8/135.0 in
  attempt-2's run; `notes.md:54-57`) — within-run comparison is the valid
  one. Claude's re-grade on the same old grader gave candidate
  92.1/165.4/529.9 vs compiled 139.5/221.4/762.4
  (`results/runs/2026-11-04-muse-round-1/grade-experts.json`), and the new
  chained timing reverses the 1-row verdict (§A, §E).
- **Invariants:** states the kept boundary explicitly: same per-block
  numerics, same float32 accumulation and bfloat16 rounding points,
  3,840 B threadgroup memory, valid for all rows (`notes.md:22-26`).

### E. Reference `ref_experts` + `SwitchGLU` — what the slot must match

- **Callers:** grader `REFERENCE["experts"]` (`slots.py:85`), called per
  layer on captured decode inputs and on `capture_prefill` inputs
  (`check_candidate.py:153-169,272-305`); in the model, `MoE.__call__`
  calls `self.experts(x, inds)` (`slots.py:110-114` installs the slot the
  same way).
- **Callees:** `ref_experts` (`slots.py:55-82`): `expand_dims`, optional
  `_gather_sort`/`_scatter_unsort` (`slots.py:62-65,80-81`), three
  `mx.gather_qmm(..., transpose=False, group_size=32, bits=4,
  mode="affine", sorted_indices=do_sort)` (`slots.py:67-79`), and compiled
  `swiglu` between them (`slots.py:79`; `third_party/mlx-lm/mlx_lm/models/switch_layers.py:9`
  import, `activations.py` fused kernel). The port's twin is
  `SwitchGLU.__call__` (`switch_layers.py:176-199`): same sort threshold
  `indices.size >= 64` (`switch_layers.py:181`; `slots.py:62`), same three
  products via `QuantizedSwitchLinear` replaced by
  `TransposedQuantizedSwitchLinear` with `transpose=False` in the served
  model (round 1 `territory-switch-layers.md` §A, §D). Kernel inventory:
  7/layer at 1–8 rows (3×(Arange+GatherQMM) + silu-mul), 13/layer at
  32 rows (+ 6 sort glue) (round 1 `territory-switch-layers.md` §A;
  `results/runs/2026-11-01-breakdown/tables.md:136-139`).
- **Data:** same shapes/dtypes as §A. Sorted path (rows ≥ 16): pairs
  flattened, `order`/`inv_order` [rows*4], x gathered to `[rows*4, 1, 768]`;
  products on sorted pairs, then unsorted back to positional `y`
  (`switch_layers.py:12-24`; `slots.py:64-65,81-82`). Output squeezed to
  `[rows, 4, 768]` (`slots.py:82`).
- **Neighbours:** identical slot edges as §A (routing before, combine
  after). The sort is internal to the reference: G2 compares only `y`
  (`check_candidate.py:153-169`), so a candidate may sort or not — but the
  sort flag changes which MLX kernel the reference dispatches (§F), and
  that changes the bits.
- **Cost:** knock-out 0.92 ms/step at 1 row, 14.67 ms at 32 rows of a
  28.19 ms step (round 1 `territory-switch-layers.md` §A;
  `claude/findings/step-breakdown.md:170,211`). Chained reference
  80.3/239.6/797.6 µs/layer (§A table). Reads 51 MB at 1 row, 908 MB at
  32 rows vs 443 MB distinct-expert must-read (GPU counters; round 1
  `ineff-runs-1.md`, `results/runs/2026-11-02-gpu-counters/tables.md:26,32`).
- **Invariants:** two thresholds, often conflated. (1) Sort:
  `inds.size >= 64` ⇔ rows ≥ 16 (`switch_layers.py:181`). (2) Kernel
  dispatch: `gather_qmm_rhs` needs `M==1 && B>=16 && right_sorted_ &&
  B/E>=4` with `B = rows*4`, `E = 60` ⇔ rows ≥ 60 (round 1
  `territory-switch-layers.md` §G, `territory-quantized-kernels.md` §A/§C;
  MLX v0.32.4 `quantized.cpp:2005,2012,2073-2093`). Decode 1/8/32 rows
  (B/E ≤ 2.13) always run `gather_qvm` even when sorted; prefill
  127/1,016/2,048 rows (B/E = 8.5/67.7/136.5) run `gather_qmm_rhs`. Each
  product accumulates float32 and rounds to bfloat16; `silu(gate)*up` on
  bfloat16 values, rounded to bfloat16 before down
  (`claude/muse-agent-brief.md:90`). New G2: every element equal at all
  six sizes (`claude/adr/0010-exact-at-every-size.md:22-27`).

### F. MLX dispatch — `affine_gather_qvm` (decode) vs `affine_gather_qmm_rhs` (prefill)

- **Callers:** `GatherQMM::eval_gpu` (MLX v0.32.4 `quantized.cpp:1980-2094`;
  installed version 0.32.4). Decode call shape `M=1, N=384/768, K=768/384,
  B=rows*4` (round 1 `territory-quantized-kernels.md` §A).
- **Callees (decode):** `affine_gather_qvm` (`quantized.h:2355-2416`
  installed headers) → `qvm_impl` (`quantized.h:1222-1330`): grid
  `(M, (N+63)/64, B)` threadgroups of 64 threads (2 simdgroups); gate/up
  `(1, 6, B)`, down `(1, 12, B)`. At 1 row that is 24/24/48 threadgroups
  per product across the three sequential kernels. Each lane owns one
  strided `K` element per iteration and decodes 32 outputs via `qouter`;
  float32 accumulation, `simd_sum` at the end, bfloat16 store
  (`quantized.h:1317-1329`).
- **Callees (prefill):** `affine_gather_qmm_rhs`
  (`quantized.h:2580-2718` installed headers): `BM=16, BN=32, BK=32`
  tiles (round 1 `territory-quantized-kernels.md` §C),
  `schedule_row_tile<BM>` (`utils.h:501-542` installed headers) maps each
  tile to a run of same-expert rows via per-expert `offsets`;
  `QuantizedBlockLoader` (`quantized.h:674-791`) dequantizes weight tiles
  into threadgroup memory, `BlockMMA` (`steel/gemm/mma.h:451`,
  `AccumType=float` default) multiplies. Round 1 mapped this kernel as
  "never run in decode" and did not read its arithmetic
  (`territory-quantized-kernels.md` §C); the arithmetic below is new in
  this round, read from the installed 0.32.4 headers.
- **Data:** identical weight/scale/bias layouts both paths. Difference is
  precision placement: `qvm` keeps `scale*q + bias` in float32 registers
  through the product (`qouter`, `quantized.h:452-553`); `qmm_rhs`'s loader
  calls `dequantize<T, pack_factor, bits>` (`quantized.h:557-663`, bits=4
  at `:614-620`) writing `static_cast<U>(sc * q + b)` with `U = T` =
  bfloat16 into threadgroup `Ws` (`quantized.h:2623-2624,2667-2674`) —
  the dequantized weight is rounded to bfloat16 before the multiply.
  Accumulation order also differs: tiled `gemm_loop_aligned` over
  `K/BK` tiles with simdgroup-matrix fragments
  (`quantized.h:2676-2684`, `mma.h:512-538`) vs `qvm`'s strided-lane
  serial sum + `simd_sum` tree.
- **Neighbours:** both paths sit between the same `swiglu` (compiled,
  full-bfloat16 per-op rounding; §B invariant) and the same unsort; the
  kernel switch changes gate/up/down bits, hence hidden bits, hence `y`.
- **Cost:** reference chained times in §E are the `qvm` path (decode
  sizes). Prefill path is not timed by G3 (decode only;
  `claude/adr/0010-exact-at-every-size.md:42`) — its cost here is
  correctness only (see ineff-1).
- **Invariants:** a candidate exact at every size must reproduce **both**
  arithmetics, or one arithmetic equal to both. Attempt-4 reproduces only
  `qvm` (§B), which is why decode passes and prefill fails (ineff-1).

## Relations

- **Fed by:** routing territory (`inds`), model territory (`x` from
  post-attention norm). Round 1: `muse-spark/exploration/territory-model.md`.
- **Feeds:** combine territory (`y` positional, paired with routing
  `weights`); shared expert independent.
- **Reference internals:** `quantized-kernels` territory owns the MLX
  kernels (`affine_gather_qvm` mapped round 1 §A; `affine_gather_qmm_rhs`
  arithmetic new here, §F, for that territory's round-2 entry to fold);
  `switch-layers` territory owns sort/unsort glue (round 1
  `ineff-switch-layers-1`, still holds: sort cannot change kernel choice
  below 60 rows).
- **Timing:** `grader` territory owns the chained-timing definition
  (`check_candidate.py:178-213`: `link` kernel chains each layer's
  smallest input to the previous layer's last output).

## Inefficiencies

- `ineff-baseline-experts-1` (open): prefill G2 failure — `qvm` replica vs
  `gather_qmm_rhs` rounding from 60 rows; correctness gate, 0 µs saving.
- `ineff-baseline-experts-2` (open): chained 1-row latency — 4 threadgroups
  + wave/barrier serialization; up to 52.6 µs/layer (≈1.05 ms/step) at
  1 row against the reference.
- `ineff-baseline-experts-3` (open): per-pair weight re-reads persist —
  465 MB excess at 32 rows; floor 2.3 ms/step at peak, unbanked by the
  starting point and not banked by round 1's grouped attempts either.

## Files read

- `muse-spark/kernels/experts/attempt-4/block-0.metal` (76 lines): kernel
  body, waves, barriers, silu — §A, §B call sites.
- `muse-spark/kernels/experts/attempt-4/block-1.metal` (41 lines):
  `qvm_block` strided accumulation + `simd_sum` — §B.
- `muse-spark/kernels/experts/attempt-4/spec.json`: 1 kernel, grid,
  threadgroup — §C.
- `muse-spark/kernels/experts/attempt-4/notes.md`: the 256-thread change,
  its predictions and old-grader verdict — §D.
- `bench/grader/slots.py:19-36,55-119,214-276,303-345`: slot shapes,
  `ref_experts`, sort threshold, prefill capture, `build` — §A, §C, §E.
- `bench/grader/check_candidate.py:153-213,244-345`: G2 checks, `link`
  chained timing, six-size loop — §A, §E, Relations.
- `third_party/mlx-lm/mlx_lm/models/switch_layers.py:12-24,75-90,176-199`:
  `_gather_sort`/`_scatter_unsort`, `SwitchGLU` twin — §E.
- `claude/muse-agent-brief.md:70-99,177-201`: slot contract, G2/G3 rules.
- `claude/adr/0010-exact-at-every-size.md`: bit-exact G2 at six sizes,
  chained timing, prefill not timed.
- `claude/findings/muse-round-1.md`: re-grade, chained measurement §4,
  exact-grade §6, starting points.
- `results/runs/2026-11-04-muse-round-1/console-chain.txt`: chained vs
  independent µs/layer (all chained numbers).
- `results/runs/2026-11-04-muse-round-1/grade-experts.json` (old G2 pass,
  independent timing) and `grade-exact-experts.json` (new G2 prefill
  differ counts 4,868,667 / 39,078,912 / 78,741,557, max 43/65/110.5 ULP).
- Installed MLX 0.32.4 headers:
  `.venv/.../mlx/backend/metal/kernels/quantized.h:557-663` (dequantize),
  `:674-791` (QuantizedBlockLoader), `:1222-1330` (qvm_impl),
  `:2355-2416` (affine_gather_qvm), `:2580-2718` (affine_gather_qmm_rhs);
  `utils.h:501-542` (schedule_row_tile); `steel/gemm/mma.h:440-545`
  (BlockMMA float accumulator) — §F.
- Round 1 (cited, not rewritten): `territory-switch-layers.md` §A-§G,
  `territory-quantized-kernels.md` §A-§C, `ineff-switch-layers-1/2`,
  `ineff-quantized-kernels-1`, `ineff-runs-1`,
  `implementation/hyp-1-fused-experts.md` (boundary, four builders,
  attempt-4 decision).

## From other agents

explorer-grader, 2026-11-04T03:18:30Z — the grader side of your two failures, from `bench/grader/` (full map in my `territory-grader.md` §4–6). (1) **Prefill exactness:** G2 now requires `differ == 0` at prefill 127/1,016/2,048 rows as well as decode 1/8/32 ([check_candidate.py:90-99, 172-175, 272-276](../../../bench/grader/check_candidate.py)); at those sizes `ref_experts` always takes its sorted path (`inds.size >= 64`, [slots.py:62](../../../bench/grader/slots.py)) and MLX's `gather_qmm` underneath runs `gather_qmm_rhs`, not the `gather_qvm` your kernel copies ([ADR 0010:11](../../../claude/adr/0010-exact-at-every-size.md)). The 60-row switch is only *bracketed* by tested sizes 32 and 127 — the grader never tests 59/60/61 — so a rows-branch threshold must come from MLX's dispatch source (quantized-kernels territory), not from grader probing; likewise every row count strictly between the six tested ones is uncertified. (2) **Timing:** the bar is the *chained* median at decode 1/8/32 ([check_candidate.py:178-213](../../../bench/grader/check_candidate.py)); your 1-row chained loss (132.9 vs 80.3 µs/layer) is the operative number, and prefill speed is not timed at all ([check_candidate.py:311](../../../bench/grader/check_candidate.py)), so a prefill branch may be slow for free but not wrong. (3) No `--baseline` bar exists this round (no attempt passes the new G2); `beats_compiled` (median candidate < compiled at all three decode sizes, [check_candidate.py:335](../../../bench/grader/check_candidate.py)) is the only timing gate.

**Folded** (owner, 2026-11-04T03:18:53Z): all three points hold and are now in the body. (1) The 60-row threshold in §E/§F and ineff-1 is from MLX dispatch source (round 1 `quantized.cpp` citations), not grader probing; §C now notes untested intermediate row counts are uncertified but the kernel must still be valid for all rows. (2) ineff-1 already states prefill speed is untimed — a slow-but-exact prefill branch is free; ineff-2 uses the chained medians as the operative bar. (3) All predicted savings are stated against the compiled reference under the chained-median rule; no `--baseline` claim is made anywhere in this territory.

explorer-g3, 2026-11-04T03:18:59Z — the end-to-end side of your chained 1-row entry (`ineff-baseline-experts-2`), from `results/runs/2026-11-04-muse-round-1/` (full map in my `territory-g3.md` pieces 3/5/6). (1) The chained gap (132.9 vs 80.3 µs/layer, `console-chain.txt`) is +1.052 ms/step, and G3 measures the attempt alone at +1.162 ms in vLLM (`console-vllm.txt`) and +1.228 ms in plain MLX (`console-plain.txt`) against base — the chain reproduces the step slowdown to 0.11–0.18 ms, so no second in-step cause is needed at 1 request. (2) My `ineff-g3-2` duplicates your `-2` from the measurement side; I have marked it superseded by yours. (3) When your kernel is combined with the routing+combine attempts, the step loses a further +2.08/+3.28 ms (vLLM, 8/32 requests) beyond the three alone changes (`ineff-g3-1`) — that interaction is not explained by your slot alone and is the open question I own.
