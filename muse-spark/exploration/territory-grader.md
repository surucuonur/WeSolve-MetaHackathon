# Territory: grader — the slot interface, G1/G2, the piece timing, the boundary

- **Owner:** explorer-grader
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-03T21:24Z
- **Next:** none — checklist sourced (6 pieces), 0 new entries (all causes already covered), 4 grader-side notes appended; no notes to answer

## What it does

`bench/grader/` is the contract every solution keeps: `slots.py` defines the three
slots (inputs, outputs, reference code) and turns a candidate's launch spec into
`mx.fast.metal_kernel` calls; `check_candidate.py` grades one attempt (G1 runs, G2
correct on captured inputs, piece timing vs reference and compiled); `time_candidates.py`
runs the deciding checks after merge (G2 end-to-end greedy tokens, G3 step period);
`gpu_lock.py` serializes all GPU jobs. It contributes no decode-step kernels itself —
its time costs are the reference slot costs it measures (the headroom) and the
slot-edge traffic it mandates (the floor).

## Checklist

### 1. `slots.py` — slot table, references, split MoE (slots.py:1-135)

- **Callers:** `check_candidate.py` uses `SLOTS`, `REFERENCE`, `build`, `load_inputs`,
  `load_expert_weights` (check_candidate.py:22-24,41,177,180,212-222);
  `time_candidates.py` uses `SLOTS`, `REFERENCE`, `make_moe_call`, and `load`
  (time_candidates.py:29-33,60-77); `slots.py capture` CLI runs `main`
  (slots.py:296-312). Builders invoke the grader, never this module directly
  (claude/muse-agent-brief.md:123-129).
- **Callees:** `ref_routing` → `astype`/`@`/sigmoid/`argpartition`/`take_along_axis`/
  sum/div/mul (slots.py:44-48); `ref_combine` → mul/sum/`astype`/add (slots.py:51-52);
  `ref_experts` → `swiglu` + 3× `mx.gather_qmm` (affine int4, group 32) +
  `_gather_sort`/`_scatter_unsort` iff `inds.size >= 64`, i.e. from 16 rows
  (slots.py:55-82); `make_moe_call` wires routing/combine/experts fns into the
  flattened MoE (slots.py:97-119); `check_identity` asserts the split MoE is
  bit-identical to the port's at 1/8/32 rows (slots.py:122-134).
- **Data:** `D,E,K,H = 768,60,4,384`, `SCALE = 2.5` (slots.py:19-20); slot I/O
  names in `SLOTS` (slots.py:29-40) with shapes/dtypes per the brief (routing:
  x [rows,768] bf16, w_router [60,768] f32, bias [60] f32 → inds [rows,4] u32,
  weights [rows,4] f32; combine: y [rows,4,768] bf16, weights [rows,4] f32,
  shared [rows,768] bf16 → out [rows,768] bf16; experts: x, inds + gate/up/down
  × w/scales/biases → y [rows,4,768] bf16; brief:28-92). Expert weights are
  named `{gate,up,down}_{w,scales,biases}` (slots.py:33-35,88-94). All arrays
  row-contiguous (brief:24). `bias` is the model's bf16 `expert_bias` converted
  exactly to f32 (slots.py:108; brief:34).
- **Neighbours:** before routing: post_attention_layernorm output x (model
  territory); routing → inds → experts, weights → combine; experts → y →
  combine; shared_expert → shared → combine; combine → out → residual/next
  layer (mobilemoe.py:217-232; slots.py:105-118). A replacement must keep: exact
  slot output shapes/dtypes (check_candidate.py:55-62,99-100,117-118); the same
  expert *set* per row up to order + weights 1e-4 relative by expert (routing);
  2 bf16-ULP outputs (combine/experts); the bf16 rounding points (each product
  f32-accumulates then rounds to bf16; silu(gate)*up on bf16 values, rounded to
  bf16 before down; brief:90); identical greedy tokens end to end.
- **Cost:** the references as the grader measures them (brief:47,64,92):
  routing ~12 kernels/layer, 240/step, 1.02 ms of a 4.75 ms step @1 row,
  1.68 ms @32; combine 5/layer, 100/step, 0.61 ms @1, 0.33 ms @32; experts
  7/layer @1 row, 13 @32 (sort appears at ≥16 rows), 0.92 ms @1, 14.67 ms of a
  28.19 ms step @32. Underlying tables are findings/runs territories; the
  per-cause entries are ineff-runs-1/2/3 and ineff-model-2.
- **Invariants:** G2 tolerances (check_candidate.py:50-139); tie rule (4th/5th
  biased within 1e-6 exempts the expert-set check); bf16 rounding points
  (brief:90); threadgroup ≤ 1024 threads and ≤ 32 KB (brief:98; count enforced
  slots.py:275); grid counts threads (slots.py:283); `rows` is read from the
  first slot input's `shape[0]` (slots.py:271).

### 2. `slots.py` — capture/save/load of grader inputs (slots.py:137-224)

- **Callers:** `slots.py capture` → `main` → `check_identity` + `save_inputs`
  (slots.py:296-312); every grading run calls `load_inputs(rows)` per row count
  and, for experts, `load_expert_weights()` once (check_candidate.py:177,180,212).
- **Callees:** `capture` runs `breakdown.prefill`/`greedy` on checkpoint
  `MODEL = "models/MobileMoE-S-QAT-mlx-bfloat16"` (breakdown.py:62) with
  `make_prompt_cache`, recording MoE inputs on decode step 8
  (slots.py:137-171); `save_inputs`/`load_inputs` use `np.savez`/`np.load`
  (slots.py:174-223); `load_expert_weights` reads the served checkpoint via
  `mlx_lm.utils.load_model` (slots.py:189-202).
- **Data:** `INPUTS = models/muse-inputs/rows{1,8,32}.npz` (slots.py:41):
  140 keys each = 20 layers × {x, w_router, bias, inds, y, weights_in, shared}
  (verified by listing; e.g. rows32 `L00_x` (32,768), `L00_y` (32,4,768),
  `L00_w_router` (60,768), `L19_*` same). Files are 4.1/6.7/15.5 MB for
  rows 1/8/32 (`ls -la models/muse-inputs/`). bfloat16 arrays (x, y, shared)
  are stored as float32 and cast back on load (slots.py:178-185,210-216);
  `weights_in` is renamed to `weights`, `inds` cast to u32, expert weights
  merged per layer (slots.py:217-223). Inputs are seeded (`mx.random.seed(rows)`,
  slots.py:158): identical every run.
- **Neighbours:** capture runs plain MLX, no API (slots.py:8); it prefills
  128-token random prompts then records the 8th decode step (slots.py:158-168).
  Consumed by every G1/G2/timing run; G2 tests all 20 layers' inputs at each row
  count (check_candidate.py:179-205).
- **Cost:** capture is one-time; per grading run the loads are host-side plus one
  `mx.eval` of weights. Not decode-step time.
- **Invariants:** fixed seeded inputs make G2 deterministic; per-layer dicts keep
  layers independent (a candidate sees one layer's inputs per call).

### 3. `slots.py` — launch-spec build (slots.py:226-293)

- **Callers:** `check_candidate.load` (check_candidate.py:34-41) and
  `Variants` in time_candidates.py:60-77, via `slots.build(spec, sources, slot)`.
- **Callees:** one `mx.fast.metal_kernel` per spec kernel (slots.py:257-264);
  the returned `run(**inputs)` chains them (slots.py:270-291). The grader runs
  no Python of the candidate's: the spec is data, only the Metal compiles
  (brief:97).
- **Data:** `spec.json` schema (brief:143-176): `kernels` in run order, each with
  name, source/header block numbers, inputs (slot inputs or earlier kernels'
  outputs), outputs (name/shape/dtype), grid, threadgroup, template, init_value,
  atomic_outputs; `slot_outputs` maps slot outputs to kernel outputs.
  `routing+combine` nests two such specs under `{"routing":…, "combine":…}`
  (brief:171). Sizes are ints or products of ints and rows/D/E/K/H only
  (`dim`, slots.py:230-240); dtypes are float32/bfloat16/float16/uint32/int32
  (slots.py:22-28); template values are dtype/int/bool (slots.py:243-248).
- **Neighbours:** Metal bodies live in `block-N.metal` numbered from 0
  (check_candidate.py:36-39); kernels run in list order with outputs chained
  through `env` (slots.py:273-290). For `routing+combine` the original experts
  and shared expert run between the two parts (brief:171).
- **Cost:** each spec kernel is ≥1 GPU launch; candidate kernel count is the
  builder's main overhead lever (see the covered inefficiency entries).
- **Invariants:** threadgroup product must be in 1..1024 (slots.py:274-276);
  missing `slot_outputs` → ValueError (slots.py:266-268); bad size terms → ValueError.

### 4. `check_candidate.py` — G1, G2, piece timing (whole file)

- **Callers:** builders run `.venv/bin/python bench/grader/check_candidate.py
  <attempt> <slot> [--out FILE]` (brief:123-129); `time_candidates.py` imports
  `load` (time_candidates.py:29). Holds the GPU lock for the whole run
  (check_candidate.py:173).
- **Callees:** `slots.build`/`load_inputs`/`load_expert_weights`; `mx.eval` for
  correctness and timing; `mx.compile(ref)` for the compiled baseline only
  (check_candidate.py:217-222).
- **Data:** `ROWS = (1,8,32)`, `REPEATS = 30` (check_candidate.py:26-27).
  Prints and optionally writes the JSON verdict
  `{slot, g1, g2, beats_compiled, correctness, timing_us_per_layer, error}`
  (check_candidate.py:174,229-232).
- **Neighbours — G1:** pass = compiled and ran without exception at all row
  counts (`g1 = True` after the loops, check_candidate.py:206; any exception →
  `error` + g1 false, check_candidate.py:227-228, Metal errors trimmed by
  `short_error`, check_candidate.py:153-163). Note: shape/dtype mismatches are
  per-layer `{"error":…}` stats, not exceptions — they fail G2 with g1 true
  (check_candidate.py:188-191).
- **Neighbours — G2 routing** (`check_routing`, check_candidate.py:50-94):
  exact shapes (rows,4)/(rows,4) and dtypes u32/f32 (55-62); inds < 60 (65);
  expert *set* per row must match (71) unless the reference's 4th/5th biased
  scores are within 1e-6 — such rows count as `near_tie_rows` and skip the
  weight check (72-80); otherwise weights matched by expert, relative error
  ≤ 1e-4 (81-87,137). Order within the row is free.
- **Neighbours — G2 combine/experts** (`check_combine`/`check_experts`,
  check_candidate.py:96-139): exact shape and bf16 dtype; every element within
  2 bf16 ULP of the reference, with the ULP floored at |r| = 0.001 (`ulp_bf16`,
  check_candidate.py:44-47). Aggregation over 20 layers: max of `max_*`, sum of
  counts, `pass` iff every layer passes, worst example kept
  (check_candidate.py:192-204).
- **Neighbours — routing+combine independence:** each part is checked and timed
  on its own; the combine part receives the *reference* (captured) routing
  weights, not the candidate's (check_candidate.py:96-98,181-187; brief:171).
  A routing miss can never fail the combine part, and vice versa.
- **Neighbours — piece timing** (`timed`, check_candidate.py:141-150,209-226):
  runs only if G2 passes. Per row count and part: 3 warmup evals, then median
  of 30 repeats of `mx.eval` over all 20 layers' calls with one GPU wait,
  divided by 20 → µs/layer. `timing_us_per_layer[rows]` = reference +
  `mx.compile(reference)` + candidate summed over parts; `beats_compiled`
  requires candidate < compiled at *all three* row counts. "It is a guide; G3
  decides" (brief:193).
- **Cost:** grading runtime ≈ correctness (20 layers × 3 rows × parts) + timing
  (33 evals × 3 variants × 3 rows × parts, each a 20-layer eval); a run should
  take under 3 minutes (brief:98). Not decode-step time.
- **Invariants:** verdict JSON shape; `--out` writes it too (229-231).

### 5. `time_candidates.py` — G2 end-to-end and G3 (whole file)

- **Callers:** surucuonur's team runs it after merging (`plain` and `vllm`
  engines, time_candidates.py:14-15); the final attempt per slot "goes to the
  end-to-end and G3 checks" (brief:195). Holds the GPU lock (time_candidates.py:113).
- **Callees:** `check_candidate.load`; `slots.make_moe_call`/`REFERENCE`;
  `breakdown` (`MODEL`, `prefill`, `greedy`, `run_plain`, `run_vllm`,
  `install_vllm_step_hooks`; breakdown.py:62,408-485); `repeat_fixes`
  (`count_ops`, `divergence`, `vllm_checks`, `time_variants`;
  repeat_fixes.py:136-277) with `SETTINGS = [(1,128),(8,128),(32,128)]`,
  `REPEATS = 3` (repeat_fixes.py:43-44).
- **Data:** variants base (port's MoE) / compiled (`mx.compile` of the slot
  reference) / muse (candidate) [/ claude_fixed] (time_candidates.py:60-77).
  `TOKENS = 24` greedy tokens at rows 1 and 32 in plain MLX
  (time_candidates.py:34,79-101); vLLM uses ≤32 seqs, 512 MiB KV cache
  (time_candidates.py:132-141).
- **Neighbours:** `as_routing`/`as_combine`/`as_experts` adapt candidate fns to
  the MoE signature and install them into *every* layer's `MoE.__call__`
  (time_candidates.py:37-77). G2-e2e requires *exact* greedy-token equality
  with base (`exact` flag + first-divergence step, time_candidates.py:97-99;
  brief:185). G3 is the steady decode-step period, median of 3 shuffled
  repeats at 1/8/32 with 128-token prompts; it must beat *both* base and
  compiled with the gap exceeding the repeat spread (brief:186).
- **Cost:** G3 is the deciding measurement; the piece timing (item 4) only guides.
  `count_ops` also records one step's kernel count per variant
  (time_candidates.py:96; repeat_fixes.py:136).
- **Invariants:** exact tokens; shuffled repeats; vLLM and plain MLX both.

### 6. `gpu_lock.py` — the GPU mutex (whole file)

- **Callers:** `check_candidate.main`, `slots.main` (capture), `time_candidates.main`.
- **Callees:** `fcntl.flock` on `/tmp/mobilemoe-gpu.lock`, shared between the
  main checkout and the worktree, OS-released on process end (gpu_lock.py:14-31).
- **Data:** lockfile content `pid <pid>: <job>`.
- **Neighbours:** the grader is a builder's only GPU job; contended runs print
  the holder and queue (gpu_lock.py:20-27; tracking.md rules). No model loads,
  no other GPU work (16 GB machine).
- **Cost:** queue wait when contended. Not decode-step time.
- **Invariants:** `_held` keeps the fd open for the process's life (gpu_lock.py:15,31).

## Relations

- **Fed by:** model territory (`MoE.__call__` mobilemoe.py:217-232, the router/
  `SwitchGLU`/shared-expert code the references mirror); switch-layers
  territory (`_gather_sort`/`_scatter_unsort`, switch_layers.py:12-24, called by
  `ref_experts`); findings/runs territories (the measured tables behind the
  reference costs); `bench/micro/breakdown.py` + `repeat_fixes.py` (MODEL,
  prefill/greedy, G3 harness); `models/muse-inputs/` (seeded captured inputs).
- **Feeds:** builders (spec format, G1/G2/timing verdicts); the boundary agent
  (slot I/O, dtypes, tolerances, rounding points — items 1+4 above); hypotheses
  (the compiled baseline is the bar to beat at all three row counts).
- **Boundary floor — traffic the slots mandate** (unremovable within the
  interface, so it caps every saving): per layer, experts must write
  y [rows,4,768] bf16 (rows×6144 B) and combine must re-read it; routing must
  write inds+weights [rows,4] (rows×32 B) for experts/combine to read. Per step
  @32 rows: y 196 KB × 2 (write+read) × 20 layers = 7.9 MB → 0.04 ms at
  200 GB/s; inds/weights ≈ 41 KB → negligible. Saving floor ≈ 0.04 ms @32 rows
  (≈ 0.001 ms @1 row): fusing *within* slots captures essentially all of each
  slot's cost; cross-slot fusion is not expressible (`routing+combine` still
  runs the reference experts + shared expert between the parts, brief:171).
- **Piece timing vs G3 (builder warning):** the piece timing medians 30 repeats
  of 20-layer evals per part and sums the parts; G3 medians 3 full steps. A
  candidate can win the piece timing yet lose G3 (e.g. via cache/overlap effects
  the isolated evals don't see). The gap must exceed the repeat spread.

## Inefficiencies

Every time-loss cause I found in this territory already has an entry owned by
another explorer, so per tracking.md (*Adapting to the others' work*) I appended
grader-side notes instead of posting duplicate `ineff-grader-N` entries:

- `ineff-runs-2` (routing overhead) — note: G2 routing check mechanics
  (set-match, weights-only-on-match, near-tie skip) + piece-timing reference.
- `ineff-model-2` (combine overhead) — note: G2 2-ULP check + routing+combine
  part independence (combine graded on reference routing weights).
- `ineff-runs-1` (experts re-reads) — note: G2 experts check (2 ULP, 0.001
  floor), bf16 rounding points, piece-timing setup.
- `ineff-runs-3` (pairs unsorted below 16 rows) — note: exact threshold code
  (`do_sort = inds.size >= 64`, slots.py:62) + G2 compares outputs only, so
  sorting at 8 rows is G2-safe.

## Files read

- `bench/grader/slots.py` — slot table, references, split MoE, capture/save/load,
  spec build (the whole interface).
- `bench/grader/check_candidate.py` — G1/G2 checks, aggregation, piece timing,
  verdict JSON.
- `bench/grader/time_candidates.py` — variants, G2 end-to-end, G3 entry points.
- `bench/grader/gpu_lock.py` — the serializing lock.
- `claude/muse-agent-brief.md` (*The task*, *The slots*, *How each attempt is
  judged*, spec format) — the contract the code implements.
- `claude/GLOSSARY.md` — slot/round/swarm/boundary/inefficiency terms.
- `third_party/mlx-lm/mlx_lm/models/mobilemoe.py:142-232` (edge) — quantized
  expert layout + `MoE.__call__`, what `make_moe_call` replaces.
- `third_party/mlx-lm/mlx_lm/models/switch_layers.py:12-24` (edge) —
  `_gather_sort`/`_scatter_unsort` used by `ref_experts`.
- `bench/micro/breakdown.py:62,408-485` (edge) — MODEL, prefill/greedy,
  run_plain/run_vllm signatures for capture and G3.
- `bench/micro/repeat_fixes.py:43-44,136-277` (edge) — SETTINGS/REPEATS,
  count_ops/divergence/vllm_checks/time_variants for G3.
- `models/muse-inputs/rows{1,8,32}.npz` (keys/shapes/sizes only) — 140 keys
  each, 4.1/6.7/15.5 MB; bfloat16 stored as float32.
- `muse-spark/exploration/ineff-runs-1/2/3.md`, `ineff-model-1/2.md`,
  `territory-model.md` (partial) — prior art; all my causes were covered.

## From other agents

(none yet)
