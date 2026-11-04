# Territory: quantized-kernels (round 2 — prefill path)

- **Owner:** explorer-quantized-kernels
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-04T03:24:31Z
- **Next:** none — prefill checklist filled (§A–§H), 2 inefficiencies posted, note left on `territory-baseline-experts`

## What it does

This territory is MLX's `mx.gather_qmm` Metal kernels behind the `experts`
slot (`quantized.h`, `quantized_utils.h`, `utils.h` in the installed
0.32.4 headers, dispatch in MLX v0.32.4 `quantized.cpp`). Round 1 mapped
it as decode-only: `affine_gather_qvm` runs at every decode size and
`affine_gather_qmm_rhs` was listed as never run
([`muse-spark/exploration/territory-quantized-kernels.md`](../../exploration/territory-quantized-kernels.md)
§A/§C, frozen). Round 2 reopens it because the new G2 is bit-exact at
prefill sizes too: from 60 rows the reference switches to
`gather_qmm_rhs`, whose rounding is now on the correctness path at
prefill 127/1,016/2,048 rows. This entry writes only what is new: the
prefill dispatch, the `qmm_rhs` arithmetic, and exactly how it diverges
from `qvm`. Decode facts that still hold are cited, not rewritten.

## The checklist

### A. Dispatch — `GatherQMM::eval_gpu` (which kernel runs at each size)

- **Callers:** `ref_experts` (`bench/grader/slots.py:55-82`) and the
  served twin `SwitchGLU.__call__`
  (`third_party/mlx-lm/mlx_lm/models/switch_layers.py:176-199`) both call
  `mx.gather_qmm(..., transpose=False, group_size=32, bits=4,
  mode="affine", sorted_indices=do_sort)` three times per layer (gate,
  up, down), with `do_sort = inds.size >= 64` ⇔ rows ≥ 16
  (`slots.py:62`, `switch_layers.py:181`). MLX is 0.32.4
  (`.venv/bin/python -c "import mlx.core as mx; print(mx.__version__)"`).
  Served weights are `[E, in, out]` via `TransposedQuantizedSwitchLinear`
  (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:142-184`), hence
  `transpose=False`; the `transpose=True` twin in `switch_layers.py:75-90`
  is not the served path.
- **Callees:** `GatherQMM::eval_gpu`
  ([quantized.cpp v0.32.4:1980-2094](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1980-L2094)).
  It shapes `K = x.shape(-1)`, `M = x.shape(-2)`,
  `N = out.shape(-1)`, `B = out.size()/M/N`,
  `E = w.size()/w.shape(-1)/w.shape(-2) = 60`,
  `vector_limit = 4` for `transpose=False` (`:2000-2005`). Branch order:
  (1) rhs fast path iff `M==1 && B>=16 && right_sorted_ && B/E>=4`
  (`:2012-2031`); (2) else `gather_qmm` iff `M >= vector_limit`
  (`:2034-2054`); (3) else `gather_qmv` iff `transpose_` (`:2056-2075`);
  (4) else `gather_qvm` (`:2077-2094`). With `M=1` always (below),
  branch (2) never fires in this model.
- **Data / shapes at every grader size.** `x` reaching `gather_qmm` is
  `[B, 1, K]` with `B = rows*4`: `xx = expand_dims(x, (-2,-3))` then,
  when sorting, `_gather_sort` gathers to one row per pair
  (`slots.py:61-65`, `switch_layers.py:12-17`); hence `M = 1` and the
  `M` argument later passed to `gather_qmm_rhs` is `x.size()/K = B`
  (`:2013-2027`). Gate/up: `K=768, N=384`; down (input is the
  bfloat16 `silu*up` hidden): `K=384, N=768`.

  | rows | B = rows·4 | B/E | sorted (`inds.size≥64`) | MLX kernel |
  |---|---|---|---|---|
  | 1 (decode) | 4 | 0.07 | no | `gather_qvm` |
  | 8 (decode) | 32 | 0.53 | no | `gather_qvm` |
  | 32 (decode) | 128 | 2.13 | yes | `gather_qvm` (sorted, still `B/E<4`) |
  | 127 (prefill) | 508 | 8.47 | yes | `gather_qmm_rhs` |
  | 1,016 (prefill) | 4,064 | 67.73 | yes | `gather_qmm_rhs` |
  | 2,048 (prefill) | 8,192 | 136.53 | yes | `gather_qmm_rhs` |

  The switch is `rows >= 60` (`B >= 240 ⇔ B/E >= 4`; `B>=16` already
  holds from 4 rows). Two thresholds must not be conflated: sorting
  starts at 16 rows, the kernel switch at 60. The grader tests 32 and
  127, only bracketing 60; the exact threshold comes from this source,
  not from grader probing (cf. `territory-grader.md` *What the grader
  still cannot see* §1).
- **Neighbours:** before: router `inds` + `_gather_sort` (switch-layers
  territory); the sort is what sets `right_sorted_`. Between gate/up
  and down: compiled `swiglu` on bfloat16 values (other-kernels).
  After: `_scatter_unsort` back to positional `[rows, 4, 768]`, then
  combine. A replacement slot kernel sees only positional `inds`
  (any order, duplicates allowed) — the reference's sort is internal
  and invisible in `y` except through the bits.
- **Cost:** the dispatch itself launches nothing. Kernel counts and
  times per branch are in §C/§F. Prefill sizes are correctness-only in
  the grader (no timing; `claude/adr/0010-exact-at-every-size.md:42`),
  so every prefill cost below is arithmetic, not a measured time.
- **Invariants:** `right_sorted_` is the Python `sorted_indices`
  argument trusted as-is; `gather_qmm_rhs` (§C) is only correct if
  `rhs_indices` really is sorted ascending by expert. `M` in the rhs
  call means total pairs `B`, not the `M=1` matrix height — the rhs
  kernel re-derives row tiles from `offsets` (§D).

### B. Host wrapper — `gather_qmm_rhs` (legacy path, the one this GPU runs)

- **Callers:** only §A branch (1) (and the `GatherQQMM` twin at
  `quantized.cpp:2233-2251`, not this model).
- **Callees:** [gather_qmm_rhs v0.32.4:1676-1830](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1676-L1830).
  First gate: `is_nax_available() && (enable_tf32 || dtype != float32)`
  diverts to `gather_qmm_rhs_nax` (`:1692-1714`). On this machine that
  is false: device is Apple M1 Pro `applegpu_g13s`
  (`mx.device_info()`), architecture generation 13 < 17, and NAX also
  needs macOS 26.2
  ([device.cpp v0.32.4:978-995](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp#L978-L995)).
  So the legacy tiling below is the prefill path here; the NAX variant
  (`bm=32/64, bn=bk=64` in the same function, NAX MMA in
  `quantized_nax.h`) never runs on the grader GPU. Legacy tiles
  (`:1748-1749`): `bm=16, bn=32, bk=32`, `wm=1, wn=2`;
  `align_N = (N%32==0)`, `align_K = (K%32==0)` — both true for
  384/768, passed as function constants 201/202. Then
  `offsets = gather_mm_offsets(indices, E, M=B, ...)` (`:1795`), one
  `affine_gather_qmm_rhs` launch with
  `group_dims = (32, wn, wm) = 64` threads and
  `grid_dims = ((N+31)/32, min(B, ceil(B/16)+E-1), 1)` (`:1817-1819`).
- **Data:** `broadcast_with_indices` (`:1721-1733`) copies `x` to
  `[B, 1, K]` only if `x.size()/x.shape(-2)/x.shape(-1) != B`; on the
  sorted path `x` already has `B` rows (the sort gathered it), so this
  is a no-op apart from `ensure_row_contiguous` flag checks, as are
  the `w`/`scales`/`biases` normalisations (already contiguous).
  `offsets` is `int32[E=60]` (240 B), the per-expert start rows in
  sorted order, built by one tiny kernel launch
  (`gather_mm_offsets`, host side at
  [matmul.cpp v0.32.4:2247-2265](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp#L2247-L2265):
  `dispatch_threads(60)`, one threadgroup). Inputs to the Metal kernel,
  in buffer order: `x` bfloat16 `[B, K]`, `w` uint32, `scales`/
  `biases` bfloat16, `offsets` int32, out `y` bfloat16 `[B, N]`, then
  `M=B, N, K, num_groups=60` as constants (`:1821-1835`).
- **Neighbours:** same slot edges as §A. The three products in one
  layer share the same sorted `idx`, hence the same `offsets` — yet
  each of the three calls recomputes it (see
  `ineff-quantized-kernels-2`).
- **Cost (arithmetic, per layer):** 3 offsets kernels + 3 main kernels
  per prefill layer-call (60 + 60 per 20-layer prefill forward),
  against 0 + 3 on the decode `qvm` path. Main-kernel dispatched
  threadgroups (64 threads each): gate/up `12 × grid_y`, down
  `24 × grid_y` with `grid_y = min(B, ceil(B/16)+59)` = 91 / 313 / 571
  at B = 508 / 4,064 / 8,192 → gate/up 1,092 / 3,756 / 6,852 and down
  2,184 / 7,512 / 13,704 threadgroups per call. Useful row tiles are
  fewer (§C, `ineff-quantized-kernels-1`).
- **Invariants:** `grid_y` over-provisions by up to `E-1 = 59` tiles:
  `ceil(B/16) + E - 1` is the worst case (every expert contributing a
  partial tile). Threadgroups whose tile index exceeds the real tile
  count return early inside the kernel (§C) — they are launched,
  scheduled, and discarded, never incorrect.

### C. Metal kernel — `affine_gather_qmm_rhs` + `schedule_row_tile`

- **Callers:** §B only. Instantiated for this model as
  `affine_gather_qmm_rhs<bfloat16, group_size=32, bits=4, BM=16,
  BN=32, BK=32, WM=1, WN=2, transpose=false>`
  (`quantized.h:2570-2593`).
- **Callees:** `schedule_row_tile<BM=16>` (`utils.h:501-542`) maps
  `tid.y` to `(y_row, group, tgp_bm)`: it scans experts in lane-parallel
  chunks of 32, prefix-summing `ceil((end-start)/16)` per expert from
  `offsets`, and returns the expert owning tile `tid.y`, its first
  row, and `tgp_bm = min(16, end - row)` valid rows (`utils.h:524-537`).
  If no expert owns the tile it returns false and the whole
  threadgroup exits (`quantized.h:2637-2640`). Then
  `QuantizedBlockLoader` (§D) fills threadgroup `Ws`, `BlockLoader`
  fills `Xs` (§E), and `gemm_loop_aligned` /
  `gemm_loop_unaligned<...>` (`quantized_utils.h:6-90`) drive
  `BlockMMA::mma` over `K_it = K/BK` tiles (24 gate/up, 12 down),
  with an extra `gemm_loop_finalize` only when `!align_K` — never
  here, since `K % 32 == 0` (`quantized.h:2631, 2676-2717`).
- **Data:** threadgroup memory `Xs[16 × 40]` + `Ws[32 × 40]` bfloat16
  = 1,280 B + 2,560 B = 3,840 B (`BK_padded = BN_padded = 32 + 16/2 =
  40`, `quantized.h:2596-2597, 2623-2624`; padding is the usual
  16-byte bank-conflict pad). Weight addressing for
  `transpose=false`: `wl += group * stride_w` with
  `stride_w = K * N_w`, `N_w = N/2` bytes-worth per row — i.e. the
  `[E, K, N]`-quantized expert block selected by `group`
  (`:2627-2633, 2655-2674`). `x` rows are the sorted pair rows
  `y_row .. y_row+tgp_bm-1` of the already-gathered `[B, K]` matrix;
  `y` writes go to those same sorted rows, unsorted later by the
  caller. Per-output values never mix rows: padded rows (§E) occupy
  separate accumulator rows and are discarded by
  `store_result_safe` (`:2694`).
- **Neighbours:** consumes the sorter's output order directly; this
  is the only gather kernel whose addressing depends on pair order
  (all others index per pair via `adjust_matrix_offsets`,
  `quantized.h:1636-1688`). Its output rows are in sorted-pair order,
  so `_scatter_unsort` after the down product is load-bearing for
  position, not for bits.
- **Cost (arithmetic):** balanced-expert model (counts near the
  uniform mean `B/60`; grader prompts are random tokens, so this is
  the expected case, not a bound): useful row tiles
  `T = Σ_e ceil(cnt_e/16)` ≈ 60 / 300 / 540 at B = 508 / 4,064 /
  8,192, against dispatched `grid_y` 91 / 313 / 571 (§B). Weight
  bytes read per product per layer = `T × 184,320 B` (per-expert
  147,456 B weights + 36,864 B scales/biases; §F) = 11.06 / 55.30 /
  99.53 MB, i.e. 1× / 5× / 9× the 11.06 MB distinct-expert floor —
  each extra row tile of an expert re-reads and re-dequantizes its
  full weights. The per-pair `qvm` path at the same `B` would read
  `B × 184,320 B` = 93.6 MB / 749 MB / 1.51 GB per product, so `rhs`
  is 8.5× / 13.5× / 15.2× leaner on weight traffic — grouping works;
  its waste is the tile quantization in `ineff-quantized-kernels-1`,
  not the grouping idea.
- **Invariants:** correctness rests on sorted `rhs_indices` +
  exact `offsets`; an unsorted input silently computes tiles from
  mixed experts (dispatch trusts `right_sorted_`, §A). `BM=16`
  grouping changes only which rows share a weight-tile load — never
  a valid row's arithmetic (§E) — so a bit-exact replacement may
  regroup rows freely, including not grouping at all, as long as
  each row's own accumulation matches §D+§E.

### D. Dequantization — `QuantizedBlockLoader` + `dequantize` (rounding point 1)

- **Callers:** `affine_gather_qmm_rhs` with
  `QuantizedBlockLoader<T=bfloat16, BROWS=BK=32, BCOLS=BN=32,
  dst_ld=40, reduction_dim=0, tgp_size=64, group_size=32, bits=4>`
  (`quantized.h:2613-2621`); same loader family as `qmm_n`/`qmm_t`
  (§G).
- **Callees:** `load_unsafe`/`load_safe` (`quantized.h:736-770`) call
  `dequantize<T, pack_factor=2, bits=4>` (`quantized.h:557-663`,
  bits=4 at `:617-623`) per packed byte: with `s = float(scale)`,
  `b = float(bias)` it writes
  `w_local[2i] = (bfloat16)(s * (w[i] & 0x0f) + b)` and
  `w_local[2i+1] = (bfloat16)((s/16) * (w[i] & 0xf0) + b)` into
  threadgroup `Ws`. Scales: one `scale`/`bias` pair per thread from
  `scales + bi * N/32` (`:733-734`); `group_steps = 32/32 = 1`, so
  `next()` advances `scales/biases` by `32*N/32 = N` per K tile
  (`:772-790`) — exactly one scale group per `BN=32` output tile and
  per `BK=32` row of `K`, matching the `[E, K, N/32]` layout.
- **Data:** packed uint32 weights are viewed as bytes (`(device
  uint8_t*)w`); each byte yields 2 outputs (low/high nibble), each
  `BK×BN = 1,024`-element weight tile dequantized once per
  (row tile, K tile) visit into `Ws`, then read by the MMA (§E).
  `x` tiles by contrast are copied verbatim (bfloat16 → bfloat16,
  exact) by `BlockLoader` (`steel/gemm/loader.h:74-80`).
- **Neighbours:** this is the first of the two divergence points
  vs `qvm` (§F): the dequantized weight is **rounded to bfloat16
  before any multiply**. In `qvm` the same `scale*q + bias` value
  stays in float32 registers through the product.
- **Cost:** dequant element-ops per product per layer =
  `T × K × N` = `T × 294,912` (both products have `K·N = 294,912`):
  17.7M / 88.5M / 159.2M at the three prefill sizes (balanced `T`
  from §C), each a float mul+add, a float→bfloat round, and a
  threadgroup store+reload. This recompute-per-row-tile is the
  loader's design (no dequantized-weight cache exists; the full
  dequantized expert is 294,912 × 2 B = 590 KB, far above the
  32 KB threadgroup limit, so cross-tile reuse would need a device
  buffer and an extra kernel — not a free fix).
- **Invariants:** `scale`/`bias` themselves are bfloat16 values
  upcast to float exactly (bfloat→float is exact), identical to
  `qvm`'s inputs; the *only* added rounding is
  `static_cast<bfloat16>(s*q + b)`. Relative error up to half a
  bfloat ULP (≈2⁻⁹ ≈ 0.2%) per weight element, deterministic per
  `(expert, k, n)` — the same rounded weight is reused for all
  ≤16 rows of the tile, so errors are correlated across rows
  sharing an expert, unlike per-pair float weights.

### E. Accumulation — `BlockLoader` + `BlockMMA` (rounding point 2)

- **Callers:** `affine_gather_qmm_rhs` instantiates
  `BlockMMA<bfloat16, bfloat16, BM=16, BN=32, BK=32, WM=1, WN=2,
  false, transpose=false, ...>` (`quantized.h:2599-2610`) and
  `BlockLoader<bfloat16, BM=16, BK=32, dst_ld=40, reduction_dim=1,
  tgp_size=64>` for `x` (`:2611-2612`).
- **Callees:** `BlockMMA` (`steel/gemm/mma.h:439-537`) with
  `AccumType = float` by default (`mma.h:451`): accumulator
  fragments `Ctile` are `float` 8×8 `simdgroup_matrix` tiles
  (`TM = 16/(8·1) = 2`, `TN = 32/(8·2) = 2` frags per simdgroup).
  `mma()` (`mma.h:513-537`) steps `kk` over `BK` in 8-wide frags
  (4 steps per tile), each `tile_matmad` issuing
  `simdgroup_multiply_accumulate` (`mma.h:181-206, 406-427`).
  Operand fragments are loaded from threadgroup bfloat16 and
  widened to float (`BaseMMAFrag::load` casts, `mma.h:57-67`), so
  the multiply is float×float with float accumulate — but the
  weight operand is the already-bfloat-rounded value of §D, and
  the 8-term dot inside each frag is the hardware's tree order,
  not a serial FMA chain. `gemm_loop_*` then adds tile results
  sequentially over the 24/12 K tiles. Store:
  `store_result`/`store_result_safe` cast float → bfloat16 once
  (`mma.h:100-112, 540-592`; `quantized.h:2683, 2694`) — the same
  single final rounding as `qvm`.
- **Data:** `Xs` holds `16×32` bfloat16 activations per K tile
  (rows zero-padded by `load_safe` when `tgp_bm < 16`,
  `loader.h:83-128`); `Ws` holds the §D dequantized tile. Partial-row
  tiles take the `gemm_loop_unaligned<false, true, ...>` branch
  (`quantized.h:2687-2695`) — full MMA work on padded rows, results
  discarded.
- **Neighbours:** identical bfloat16 store boundary as `qvm`, so
  the downstream `swiglu` (gate/up) and down product see the same
  *kind* of rounding point — only the float value arriving at the
  cast differs.
- **Cost:** per product, MMA flops are `2·B_padded·K·N` with
  `B_padded = T·16` = 960 / 4,800 / 8,640 vs useful `B` = 508 /
  4,064 / 8,192 (padded fractions in `ineff-quantized-kernels-1`).
  At 5.2 TFLOPS peak (`claude/GLOSSARY.md:14`) the useful MMA
  floors at `2·B·K·N/5.2 TFLOPS` = 57.6 / 461.0 / 929.2 µs per
  layer-product (B = 508 / 4,064 / 8,192) if it ran at peak — it
  cannot, at `BM=16` tiles with two barriers per 32-K step
  (`quantized_utils.h:14-29`), so the path is
  overhead/latency-bound, not compute-bound; no measured prefill
  time exists to check this against (§A).
- **Invariants:** per-output accumulation grouping is
  **[contiguous 8-wide frag dots] sequentially over BK=32 tiles**,
  in increasing K. This is the second divergence from `qvm` (§F):
  even with bit-identical weight operands, regrouping a 768-term
  float sum this way rounds differently on generic data. Reproducing
  it with scalar FMAs is not bit-safe; reproducing it requires the
  same 8×8 `simdgroup_matrix` fragmentation (same `BK`, same
  frag step order) or a proof the hardware order coincides — none
  exists in the sources. Warp tiling `WM/WN` only partitions frags
  across simdgroups; it does not change a single output's add order.

### F. Comparator — `qvm_impl` + `qouter` (what decode runs; delta only)

Round 1 §A mapped this kernel fully; only the contrast is new.

- **Callers/Callees:** `gather_qvm` host (`quantized.cpp:1462-1529`)
  → `affine_gather_qvm` (`quantized.h:2355-2416`) → `qvm_impl`
  (`quantized.h:1222-1330`): grid `(1, (N+63)/64, B)` threadgroups of
  64 threads; gate/up `(1, 6, B)`, down `(1, 12, B)`. Each lane owns
  one strided `K` element per 32-step (`x += block_size` with
  `block_size = 32`, `:1275-1287`) and accumulates 32 outputs in
  float registers via `qouter` (`quantized.h:452-553`, bits=4 at
  `:503-509`): `result += x * (scale*q + bias)` with
  `scale`/`bias`/`x` all float, **no bfloat16 rounding of the
  weight**, serial per-lane adds in strided-K order
  (`k = lane, lane+32, ...`), then one `simd_sum` tree over the 32
  lane partials (`:1317-1321`) and a single float→bfloat16 store
  (`:1324-1329`).
- **The two divergences, side by side:**

  | aspect | `gather_qvm` (decode, rows < 60) | `gather_qmm_rhs` (prefill, rows ≥ 60) |
  |---|---|---|
  | weight value | `float(scale)·q + float(bias)` kept in float32 | same value rounded to bfloat16 in threadgroup (§D) |
  | K grouping | strided lanes: lane `l` sums `k≡l (mod 32)` serially, then `simd_sum` tree | contiguous: 8-wide frag dots, sequential over BK=32 tiles (§E) |
  | rows sharing work | none: one grid slice per pair | ≤16 sorted rows share each weight tile |
  | final rounding | float → bfloat16 once after `simd_sum` | float → bfloat16 once after MMA — identical point |

  Either divergence alone breaks bit-equality (measured: 62.4–62.6%
  of experts outputs differ at all three prefill sizes, max
  43/65/110.5 ULP, `results/runs/2026-11-04-muse-round-1/grade-exact-experts.json`;
  full statement in `ineff-baseline-experts-1`). A single arithmetic
  cannot match both kernels on generic inputs: pick any weight with
  `float(scale·q+bias)` not representable in bfloat16 and any
  `x ≠ 0` — the two references already disagree on that product
  before accumulation even starts. **Conclusion for builders: a
  size branch is required** — `qvm` arithmetic below 60 rows,
  `qmm_rhs` arithmetic (bfloat-rounded dequant tiles + 8-wide frag
  accumulation) at/above 60 rows — or a proof of coincidence that
  the 62% differ rate rules out empirically. The branch threshold
  must be the dispatch rule `rows >= 60` (pairs `>= 240`, `E=60`),
  not the grader's nearest tested sizes 32/127: rows 33–59 in the
  real model still run `qvm` (§A). Row *grouping* inside the
  prefill branch is free (§C invariant); per-row accumulation is
  not.
- **Cost:** decode cost is round 1's confirmed entry
  (`muse-spark/exploration/ineff-quantized-kernels-1`, confirmed;
  round-2 restatement `ineff-baseline-experts-3`): 51 MB read in
  0.92 ms at 1 row (55 GB/s), 908 MB in 14.67 ms at 32 rows
  (62 GB/s, 31% of peak). Not re-posted here.
- **Invariants:** both paths accumulate in float32 and round to
  bfloat16 exactly once per product; `silu(gate)·up` is computed on
  those bfloat16 values and rounded again before down (brief:
  `claude/muse-agent-brief.md:90`). Any candidate must keep all
  four bfloat16 points (gate, up, hidden, down) — matching only the
  final `y` rounding is insufficient, since 1-ULP hidden flips
  amplify through down (round 1
  `muse-spark/implementation/hyp-1-fused-experts.md:103-108`).

### G. Never-run neighbours — `gather_qmm` / `qmm_n_impl` / `qmm_t_impl`

- **Callers:** §A branch (2) needs `M >= 4`; MobileMoE always has
  `M=1`, so `affine_gather_qmm_n`/`_t` (`quantized.h:2418-2568`)
  with `qmm_n_impl` (`:1468-1595`) / `qmm_t_impl` (`:1341-1459`),
  all `BM=BN=BK=32, WM=WN=2`, never run in this model at any size.
  Documented so no one chases them for the prefill failure: the
  prefill kernel is `qmm_rhs` (§C), not these.
- **Callees/Data/Invariants:** they share §D's loader (bfloat-rounded
  weights) and §E's `BlockMMA` accumulation, so their per-output
  arithmetic would match `qmm_rhs`'s rounding family — but their
  grid is per `(N tile, M tile, pair-batch B)` via
  `adjust_matrix_offsets`, not offset tiles. Zero cost here; the
  only use is as evidence that MLX's matrix family consistently
  dequantizes to the element type before multiplying, while its
  vector family (`qvm`/`qmv`) does not.

### H. NAX variant — `gather_qmm_rhs_nax` (not on this GPU)

- **Callers:** §B's first gate only; false on the M1 Pro grader
  machine (§B). Tiles there are `bm=32` (or 64 when `M/E >= 64`),
  `bn=bk=64` with NAX matrix ops (`quantized_nax.h`). If the grader
  ever moved to an M3-class GPU, the prefill reference's frag size
  and K blocking would change with it, and a hard-coded `BK=32`
  replica would fail G2 there. On the current hardware this is a
  portability note, not a cost.

## Relations

- **Feeds from:** switch-layers territory (sorted `idx`, `x` gathered
  to `[B, 1, K]`, the `right_sorted_` flag), model territory (router
  `inds` before that).
- **Feeds to:** other-kernels territory (compiled `swiglu` between
  the products), switch-layers (`_scatter_unsort`), then combine.
- **Slot view:** `baseline-experts` territory owns the starting
  point kernel that must branch (§F conclusion);
  `ineff-baseline-experts-1` owns the G2-failure statement. This
  territory owns why the two MLX arithmetics differ and what the
  reference itself wastes at prefill (§C, inefficiencies below).

## Inefficiencies

- `ineff-quantized-kernels-1` (open): `BM=16` row-tile quantization
  in `gather_qmm_rhs` — dispatched empty tiles plus zero-padded
  rows waste 47% / 15% / 5% of row work at prefill 127 / 1,016 /
  2,048 rows. Prefill-only; 0 µs at decode sizes.
- `ineff-quantized-kernels-2` (open): `gather_mm_offsets` recomputed
  identically for gate/up/down — 60 tiny launches per prefill
  forward where 20 suffice. Prefill-only; 0 µs at decode sizes.
- Decode per-pair waste stays round 1's confirmed
  `ineff-quantized-kernels-1` (round-2: `ineff-baseline-experts-3`);
  not duplicated here.

## Files read

- `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h`
  (installed 0.32.4): `qouter` :452-553, `dequantize` :557-663,
  `QuantizedBlockLoader` :665-791, `qvm_impl` :1222-1330,
  `qmm_t_impl` :1341-1459, `qmm_n_impl` :1468-1595,
  `adjust_matrix_offsets` :1636-1688, `affine_gather_qvm`
  :2355-2416, `affine_gather_qmm_n/t` :2418-2568,
  `affine_gather_qmm_rhs` :2570-2718 — §C–§G.
- `.venv/.../kernels/quantized_utils.h` (90 lines): `gemm_loop_*`
  barriers and tile stepping — §C/§E.
- `.venv/.../kernels/utils.h:497-542`: `schedule_row_tile<BM>` —
  §C.
- `.venv/.../kernels/steel/gemm/mma.h:406-592`: `BlockMMA`
  (`AccumType=float` default :451), frag MMA and stores — §E.
- `.venv/.../kernels/steel/gemm/loader.h:14-134`: `BlockLoader`
  safe/unsafe loads and zero padding — §E.
- `.venv/.../kernels/quantized_nax.h` (grep): NAX tile sizes — §H.
- MLX v0.32.4 source via web fetch (installed version confirmed
  0.32.4): `mlx/backend/metal/quantized.cpp` — `gather_qmm`
  :1280, `gather_qvm` :1461, `gather_qmm_rhs` :1676-1830 (legacy
  `bm/bn/bk = 16/32/32` :1748), `GatherQMM::eval_gpu` :1980-2094
  (rhs condition :2012); `mlx/backend/metal/matmul.cpp:2247-2265`
  (`gather_mm_offsets` host launch); `mlx/backend/metal/device.cpp:978-995`
  (`is_nax_available` gate) — §A/§B/§H.
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:142-184, 187-232`
  (`TransposedQuantizedSwitchLinear`, `transpose=False`) and
  `switch_layers.py:12-24, 176-199` (sort threshold, three
  products) — §A.
- `bench/grader/slots.py:55-82, 214-254` (`ref_experts`,
  `capture_prefill` sizes 127/1,016/2,048) — §A table.
- `results/runs/2026-11-04-muse-round-1/grade-exact-experts.json`:
  prefill differ 4,868,667 / 39,078,912 / 78,741,557, max
  43/65/110.5 ULP — §F.
- `muse-spark/exploration/territory-quantized-kernels.md` (round 1,
  frozen): decode map §A–§E this entry builds on;
  `muse-spark/round-2/exploration/territory-baseline-experts.md`
  §E/§F and `territory-grader.md` (peer entries; threshold and
  timing framing cross-checked).

## From other agents

(none yet)
