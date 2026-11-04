# Territory: findings

- **Owner:** explorer-findings
- **Status:** done
- **Version:** 4
- **Updated:** 2026-11-03T22:05Z
- **Next:** done — lead to settle overlaps; verifier to reconcile saving bases (findings-2 vs qkern-1) and the runs-3 premise

## What it does

This territory is the measured evidence, not code: the three findings documents in
`claude/findings/` and the raw breakdown run in
`results/runs/2026-11-01-breakdown/`. It tells every other explorer what each piece
of the decode step costs in place (kernels, CPU, knock-out savings) so no
inefficiency is claimed without a measured cost. The headline numbers the swarm
builds on: the 1-row step is a chain of 867 kernels where routing+combine cost
1.63 ms of 4.75 ms, and at 32 rows the routed experts cost 14.67 ms of 28.19 ms
while reading 908 MB against the 443 MB they need.

## Checklist

### A. Step periods and the vLLM-vs-plain-MLX gap

- **Callers:** `claude/findings/step-breakdown.md` §1 (lines 101-116); the brief's
  slot costs (§`routing`/`combine`/`experts`) rest on these knock-outs.
- **Callees:** `results/runs/2026-11-01-breakdown/tables.md` "Step period by mode"
  (lines 14-24) and "Each repeat" (lines 26-52); raw records in `runs-plain.jsonl`,
  `runs-vllm.jsonl` (one record per run).
- **Data:** bare-mode medians of 3 repeats. Plain MLX: 4.75 / 13.37 / 28.19 ms at
  1 / 8 / 32 rows × 128-token prompts, 33.20 ms at 32 × 1024. vLLM: 5.64 / 14.30 /
  30.47 / 39.71 ms. vLLM's extra: +19% / +7% / +8% / +20% (+6.51 ms long-context).
- **Neighbours:** piece B (runner spans) explains the 1-row gap as CPU; piece H
  (attention) explains the long-context gap. Replaces decode-bottlenecks finding 1
  ("5-10%"), which compared different harnesses.
- **Cost:** the whole-step denominators every saving is measured against: 4.75 ms
  at 1 row, 28.19 ms at 32 rows (plain MLX, bare).
- **Invariants:** steady decode steps only (first 5 of 256 dropped); random-token
  prompts, different generators per engine (routing differs slightly: 40.1 vs 36.4
  distinct experts/layer at 32 rows); machine paged heavily throughout (Limits,
  step-breakdown lines 493-497; `machine-*.txt`).

### B. Runner spans: who sets the pace, CPU or GPU

- **Callers:** step-breakdown §2 (lines 117-144); decode-bottlenecks §4
  (lines 135-156) for the profiled submit-call split.
- **Callees:** tables.md "Time per step in each runner span, cpu mode" (lines 54-71)
  and "CPU time per step to build each piece's graph" (lines 73-110).
- **Data:** per-step means over 3 repeats, cpu mode. vLLM schedule: 0.06 / 0.12 /
  0.30 / 0.48 ms at 1 / 8 / 32 / 32-long. Encode (`submit`): plain 2.61 / 3.61 /
  6.02 ms, vLLM 4.10 / 4.70 / 6.46 ms at 1 / 8 / 32 rows. GPU-wait inside
  `mx.async_eval`: 0.41 ms (vLLM, 1 row) vs 19.39 ms (vLLM, 32 rows).
- **Neighbours:** at 8+ rows the GPU sets the pace (runner work is free); at 1 row
  vLLM is CPU-set (0.41 ms GPU wait; plain MLX has 0.67 ms wait, GPU-set with CPU
  to spare). 1-row vLLM cpu-mode column is an upper bound (instrumentation
  lengthens the step 5.64 -> 7.44 ms).
- **Cost:** scheduler never the problem (0.06-0.30 ms); runner outside the model
  0.5 / 0.9 / 1.6 ms at 1 / 8 / 32 rows; encoding the largest CPU cost (4.10 ms
  vLLM vs 2.61 ms plain at 1 row).
- **Invariants:** cpu-mode 1-row vLLM figures are upper bounds; profiled Python
  figures (decode-bottlenecks §4) overstate Python parts.

### C. Per-piece cost at 1 row: routing+combine beat the expert math

- **Callers:** step-breakdown §3 "With one request" (lines 155-195);
  bottleneck-explained §3 (lines 106-118); brief slot costs (routing 1.02 ms,
  combine 0.61 ms, experts 0.92 ms at 1 row).
- **Callees:** tables.md knock-out tables (lines 320-356: "What removing a piece
  saves", median and fastest-repeat) and "Each piece at 1 row" (lines 358-375);
  kernel counts from tables.md "Kernel operations per step" (lines 112-147).
- **Data:** plain MLX, per step of 20 layers. Routing: 240 kernels, 173 µs CPU,
  saves 1.02 ms (4.3 µs/kernel). Routed experts: 140 kernels, 220 µs CPU, saves
  0.92 ms (6.6 µs/kernel). Combine: 100 kernels, 75 µs CPU, saves 0.61 ms
  (6.1 µs/kernel). Shared expert: 80 kernels, saves 0.51 ms. Output layer: 1
  kernel, saves 0.39 ms (158 GB/s). MoE all: 560 kernels, 2.94 ms. Attention all:
  220 kernels, 1.21 ms.
- **Neighbours:** vLLM 1-row knock-outs are ranges/noise (paging; step-breakdown
  lines 175-176) — use plain MLX at 1 row. Sync-mode GPU times are NOT used
  (sum 5-8x the real step; Limits lines 503-505); only kernel counts come from
  sync mode.
- **Cost:** routing+combine = 340 kernels and 1.63 ms, a third of the 4.75 ms
  step — more than the routed experts' 0.92 ms. Most pieces cost 4-7 µs/kernel.
- **Invariants:** knock-out saving = piece cost where it runs (CPU, GPU or both);
  a faster version gains less than the full saving (Limits lines 507-509).

### D. Per-piece cost at 32 rows: the routed experts take half the step

- **Callers:** step-breakdown §3 "With 32 requests" (lines 196-230);
  bottleneck-explained §4 (lines 120-133); brief (experts 14.67 ms of 28.19 ms).
- **Callees:** tables.md lines 320-356 (knock-outs) and 377-394 ("Each piece at
  32 rows"); both engines agree within 2.5% at 32 rows.
- **Data:** plain MLX, per step. Routed experts: 260 kernels (140 + 120 sort ops:
  60 Gather, 40 ArgSort, 20 FloorDivide — tables.md lines 136-139), 537 µs CPU,
  saves 14.67 ms (52%). Attention all: 5.60 ms (kernel 3.12). Shared expert: 2.90
  ms. Output layer: 2.50 ms. Routing: 1.68 ms. Combine: 0.33 ms. MoE all: 680
  kernels, 17.09 ms (60%).
- **Neighbours:** vLLM output layer saves 5.08 ms vs plain 2.50 ms — unexplained,
  not the kernel (`lm_head_shapes.txt`: 2.351 vs 2.356 ms either layout, same
  logits); at least one vLLM 32-row knock-out removes more than its piece
  (top-level savings sum to 33.3 ms > 30.3 ms step). Use plain MLX figures.
- **Cost:** experts 14.67 ms (52%); attention 5.60 ms (20%); routing bookkeeping
  stops mattering (6%).
- **Invariants:** 32-row savings are not additive (removing a large piece changes
  overlap; replayed outputs change later routing — step-breakdown lines 225,
  507-509).

### E. Derived bandwidth/compute and the fixed cost per kernel

- **Callers:** step-breakdown §3 "Memory bandwidth and compute, piece by piece"
  (lines 231-295); bottleneck-explained §§1-3 (lines 47-118); decode-bottlenecks
  §2 (lines 83-112).
- **Callees:** tables.md "Achieved bandwidth and compute of each piece" (lines
  408-438) and "The whole step's achieved bandwidth and compute" (lines 440-450);
  op mix from tables.md "Operations per step by type" (lines 149-159).
- **Data:** peaks 200 GB/s, 5.2 TFLOP/s. Whole step uses 12-19% of peak bandwidth
  (38 / 29 / 26 / 39 GB/s plain at 1 / 8 / 32 / 32-long), 5.1-8.7x its lower
  bound — overhead-bound in both engines. At 1 row a piece's GB/s follows bytes
  per kernel: output layer 158 (1 kernel, 61.6 MB), shared expert 86, projections
  ~30, routing 4 (240 kernels, 3.7 MB). Kernel count 867 (1 row) / 987 (32 rows)
  plain, 968 / 1088 vLLM; tiny-op chain 6.81 µs/op end to end.
- **Neighbours:** bytes count each weight once and skip activations, so derived
  GB/s are lower bounds on real traffic. GPU counters (runs territory,
  2026-11-02) check them: 1-row reads 193 MB vs 180 MB needed; output layer
  167 GB/s vs derived 158.
- **Cost:** at 5-7 µs/kernel, a kernel reading 0.37 MB cannot exceed 50-75 GB/s;
  at 1 row the limit is the kernel count, not bandwidth.
- **Invariants:** lower bound = max(bytes/200 GB/s, FLOP/5.2 TFLOP/s); 1-row vLLM
  excluded (CPU-set, savings are CPU time).

### F. Expert bytes: per-pair reads, and the grouped kernel that never runs

- **Callers:** bottleneck-explained §4 (lines 135-194); decode-bottlenecks §5
  (lines 158-204); step-breakdown §3 memory-traffic subsection.
- **Callees:** per-layer distinct experts in `runs-plain.jsonl` (`distinct_experts`
  in sync records; e.g. 43,50,45,...,39 at 32 rows, total 802 pairs; means in
  tables.md lines 396-406); MLX `quantized.cpp:2011-2012` rule (`M==1 && B>=16 &&
  sorted && B/E>=4` → grouped needs 240+ pairs = 60+ rows); `switch_layers.py:181`
  (sort from 64 pairs = 16 rows). Byte totals (908 MB read, 443 MB needed) are
  runs-territory counter data (2026-11-02) cited by the findings.
- **Data:** one expert = 3 × 294,912 int4 weights = 0.553 MB (weights 147,456 B +
  scales/offsets 36,864 B per matrix). 32 rows: 2,560 pairs ask 1,416 MB;
  distinct-once needs 443 MB; 908 MB reached memory → 465 MB extra (30% of the
  1,557 MB step reads), cache served the other half of the 973 MB repeat reads.
  Per-pair vs grouped on equal balanced data (chained 20 layers): 5.08→3.82
  (−25%, 8 rows), 8.12→5.65 (−30%, 16 rows), 15.02→9.63 ms (−36%, 32 rows).
  Weight layout within 4% for grouped — not the cause.
- **Neighbours:** real routing shares less than the test (3.2 pairs/expert at 32
  rows in the counted step, 128/40.1; 2.2 at 16 rows) — grouped gain there is
  untested. Sorting from 16 rows only reorders the per-pair kernel's visits.
- **Cost:** experts move 908 MB over 14.67 ms = 62 GB/s (31% of peak); 908 MB at
  peak = 4.5 ms, 443 MB at peak = 2.2 ms, yet measured 14.67 ms — most is kernel
  efficiency, not bytes (estimate).
- **Invariants:** random-token prompts; real text may route to a different
  distinct-expert count (bottleneck-explained "What remains uncertain").

### G. Split-K partial writes from 14 rows up

- **Callers:** bottleneck-explained §6 (lines 223-283); step-breakdown §3 memory
  traffic (writes) and §6 "new lead, not yet a fix" (lines 485-488).
- **Callees:** MLX `quantized.cpp:89-125` (`get_qmv_batch_limit` = 14 rows on
  applegpu_g13s), `:1166-1226` (split choice), `:1909-1915`; runs-territory
  counter data cited: writes 3 / 20-21 / 251-266 MB per step at 1 / 8 / 32 rows;
  knock-out write savings 57 MB (q,k,v) and 102 MB (shared expert).
- **Data:** split_k at 32 rows: q 12, k 24, v 24, o 12, shared gate/up 8, down 16;
  buffers split_k × rows × outputs × 4 B: q 1.18, k/v 0.79, o 1.18, shared 1.57
  MB per layer → 23.6 + 15.7 + 15.7 + 23.6 + 31.5 + 31.5 + 31.5 = 173 MB/step.
  No split-K for output layer (4,008 tiles), routed experts (batched kernel),
  router (not quantized).
- **Neighbours:** check sums — worked-out 55 MB vs measured 57 MB (q,k,v), 94 vs
  102 MB (shared); 4× the 8-row 20 MB + 173 MB ≈ 253 MB vs measured 251-266 MB.
  Each layer's buffer is 0.8-1.6 MB, so read-back may come mostly from cache.
- **Cost:** 173 MB written per step + read-back + 7 extra add kernels/layer (140
  per step). Time effect NOT measured — "a lead, not yet a fix".
- **Invariants:** needs a patched MLX to test (no setting turns split-K off);
  bfloat16 partials would change rounding.

### H. Attention re-reads and the vLLM gap (for the attention territories)

- **Callers:** bottleneck-explained §5 (lines 196-221); decode-bottlenecks §6
  (lines 206-237); step-breakdown §4 (lines 381-416).
- **Callees:** vllm-metal `metal/ops_internal.h:47-54` (GQA layout list misses
  MobileMoE-S 12/4/64/16), `paged_attention.cpp:67-76,156,1026-1028`,
  `pagedattention.metal:768,860` (per-query-head reads); `sdpa.py:722-724`
  (`[0]` gathers + `Contiguous`); tables.md kernel/CPU/knock-out rows for
  attention pieces; `lm_head_shapes.txt` (layout not the output-layer cause).
- **Data:** cache needed at 32 rows ≈ 160 MB (7,792 tokens × 1,024 B × 20 layers);
  plain MLX kernel reads 265 MB (1.7x, 85 GB/s over 3.12 ms); vLLM kernel 4.33 ms
  at 32×128, gap +6.51 ms at 32×1024 (estimate: triple reads). Direct-kernel test:
  12/4 takes 2.3x the 4/4 time for the same bytes (125 vs 55 µs @256 tokens/layer).
  vLLM attention path: ~1 ms Python/step vs 0.23 ms plain; +60 Gather, +60
  Contiguous (copying only from 2 rows up), +140-220 view ops; encode +1.1-1.5 ms.
- **Neighbours:** cause sits in attention/vllm-metal code — other explorers'
  territory (see Relations). GQA admission rule (full partitions × q-heads ≥ 33 ×
  cores) limits any fix to long contexts.
- **Cost:** attention in place: plain 1.21 / 5.60 ms, vLLM 2.0-2.6 / 7.48 ms at
  1 / 32 rows; long-context vLLM-vs-plain gap 6.51 ms.
- **Invariants:** outside the slots (attention is not routing/combine/experts).

### I. Repeated work: real but cheap

- **Callers:** step-breakdown §5 (lines 417-466); tables.md "Work repeated in
  every step" (lines 452-471).
- **Callees:** `fixes-plain.jsonl`, `fixes-vllm.jsonl` (op counts, output checks,
  timed runs); `console-fixes-*.txt`; code cites (MLX `ops.cpp:79-108,5789,5824`,
  `device.cpp:371,409-421,606`; `mobilemoe.py:223,231`; vllm-metal
  `sdpa.py:722-724`, `varlen_rope_compat.py:221-228`).
- **Data:** three exact fixes remove 140 of vLLM's 967 kernel ops/step
  (bias_f32 20, lhs_indices 60 below 16 rows, qkv_view 60 + 60 no-longer-copying);
  all outputs bit-identical. Gains stand clear of noise only at ≤1.5% of the step
  (plain 0.06-0.07 ms @1/8 rows; vLLM 0.21 ms @8 rows from qkv_view).
- **Neighbours:** MLX dispatches independent kernels concurrently (barrier only on
  read-after-write), so index ranges and bias casts run alongside the chain —
  kernel count is a rough guide; the dependent chain + CPU encode set the step.
- **Cost:** ≤0.2 ms/step; worth making as clean-ups, not speed-ups.
- **Invariants:** lhs_indices helps only below 16 rows (MLX uses sorted kernels
  only without user indices); combine AsType pair has no exact standalone fix.

### J. Compiling the MoE block: −29% at 1 row

- **Callers:** decode-bottlenecks §3 (lines 113-133); bottleneck-explained §8 fix-1
  row; step-breakdown §6 (fix 1 stays first).
- **Callees:** 2026-10-30-micro `moe_parts.py` data (runs territory) as quoted:
  MoE block chained over 20 layers, sync timing (build+encode+run).
- **Data:** 6.02→4.27 ms (−29%) at 1 row; 6.95→5.61 (−19%) at 4; 14.03→13.45 (−4%)
  at 16; 21.56→20.51 (−5%) at 32 rows. vllm-metal's `VLLM_METAL_COMPILED_MLP`
  wraps only Qwen3-Next/3.5, ≤4 rows (`compiled_mlp.py:49,116-131,174-187`);
  MobileMoE-S never wrapped (silent).
- **Neighbours:** compiled all 20 blocks as one function — per-block compile may
  gain less. Fix 1 + fix 2 (projection fusion, −80-100 kernels ≈ −0.5 ms GPU,
  −0.25 ms CPU) + fix 3 (CPU trim) estimated 5.5 → 3.5-4 ms at 1 row.
- **Cost:** the 29% is the measured existence proof that fusing the MoE
  bookkeeping pays at 1 row.
- **Invariants:** test routing spread wider than the sweep (47.8 vs 38.2 distinct
  experts/layer at 32 rows); sync times don't add up to a step.

### K. Limits and caveats every claim inherits

- **Callers:** step-breakdown Limits (lines 491-523); bottleneck-explained "What
  is not the bottleneck" and "What remains uncertain" (lines 317-335).
- **Callees:** `machine-start.txt`, `machine-between.txt`, `machine-end.txt`
  (swap 24.7-31.1 GB, 8.6k/18.1k pages/s); `console-vllm-probe-refused.txt`
  (KV-commit probe off); tables.md "Machine during the runs" (lines 8-12).
- **Data:** vLLM 1-row repeats spread 17-47% (single steps stall to 124 ms) vs
  0.2-3.3% plain; recording costs 3-9% (plain 32×1024 unusable, 33→57 ms);
  counter bytes repeat within 0.5%; <60 MB changes at 32 rows not attributed
  (cache effects — e.g. shared-expert knock-out raised reads 51 MB); unattributed
  reads ~255 MB at 32 rows.
- **Neighbours:** not bottlenecks (all measured): scheduler, runner remainder,
  repeated small ops, expert weight layout. Uncertain: small-kernel mechanism,
  split-K time effect, fix interactions, real-text routing, what the read counter
  counts.
- **Cost:** no claim finer than ~0.1 ms at 32 rows (drift floor) or inside the
  repeat spread is made.
- **Invariants:** no per-kernel GPU trace; GPU clock unrecorded; prompts random.

## Relations

- **Feeds every code territory** (`model`, `switch-layers`, `quantized-kernels`,
  `other-kernels`, `vllm-metal`, `grader`): this entry's pieces C-J are the
  measured costs their inefficiencies must cite. Slot costs per the brief: routing
  1.02 / 1.68 ms, combine 0.61 / 0.33 ms, experts 0.92 / 14.67 ms at 1 / 32 rows.
- **Fed by `runs`:** pieces E-H rest on raw counter/micro data from
  `results/runs/2026-11-02-gpu-counters/` (908 MB, 465 MB, 62 GB/s, write totals)
  and `results/runs/2026-10-30-micro/` (compile −29%, grouped −25-36%, attention
  head table). Those raw files are the runs territory's to map; the findings
  documents quoting them are mapped here. Edge note for `explorer-runs`: if the
  raw data contradicts any quoted figure above, this entry needs the correction.
- **Yields to code territories on causes:** piece H (attention re-reads, vLLM
  attention path, GQA layout miss) belongs to whoever covers attention /
  vllm-metal code — I post no inefficiency for it, only the measured costs. The
  vLLM 32-row output-layer excess (5.08 vs 2.50 ms, piece D) is likewise an open
  measurement puzzle for the model/vllm-metal territories, not a claim.
- **Yields to `quantized-kernels` on MLX kernel internals:** pieces F-G cite
  `quantized.cpp` split-K and `gather_qmm` selection rules from the findings; the
  code-level confirmation is that territory's.
- **Adaptation (2026-11-03T21:55Z):** `runs` and `model` posted first and cover
  the same routing/expert causes (ineff-runs-1/2/3, ineff-model-2); kept my
  entries open because (a) ineff-vllm-metal-1 points at ineff-findings-1,
  (b) findings-1 maps the joint `routing+combine` slot, (c) findings-2 covers
  kernel efficiency beyond runs-1's bytes floor. Appended notes on ineff-runs-1
  (scope split, Kind vs glossary, 8-row scaling), ineff-runs-2 (overlap,
  68 MB question), ineff-model-2 (overlap, 8-row range), ineff-runs-3 (premise
  correction: micro A/B sorted flag selects the KERNEL, pairs ordered in both
  arms — `moe_parts.py:149-163`), and territory-runs (edge pointer). Lead to
  settle; the runs-3 correction is the one builders must not miss.
- **Adaptation 2 (2026-11-03T22:05Z):** `quantized-kernels` and `switch-layers`
  posted. ineff-quantized-kernels-1 is the in-kernel mechanism behind
  ineff-findings-2's measured cost (complementary; saving bases 8.6 vs 3.67 ms
  @32 need verifier reconciliation — note appended, incl. the micro-rhs-fired
  point against the locality reading of the A/B gap).
  ineff-quantized-kernels-2 and ineff-switch-layers-2 both cover launch fusion
  (0.2-0.4 ms; lead to merge) — their finer inventory corrected my findings-2
  @1-row figure 660 → 400 µs (Aranges are off-chain). ineff-switch-layers-1
  (sort glue 0.7 ms @32) is consistent with piece F.

## Inefficiencies

- `ineff-findings-1` (open, v2): routing+combine bookkeeping dominates the
  1-row step — 1.63 ms of 4.75 ms @1 row, saving ≈1420/1500/1800 µs @1/8/32
  rows by fusion to 1 kernel/layer/slot. Overlaps ineff-runs-2 + ineff-model-2
  (kept: joint `routing+combine` slot rationale; notes appended on both);
  ineff-vllm-metal-1 superseded by it (engine note folded in).
- `ineff-findings-2` (open, v2): routed experts read per pair and run slowly —
  14.67 ms of 28.19 ms @32 rows, saving 3670 µs @32 (25% grouped basis; upper
  5280), ≈660 @1 (fusion), ≈1300 @8 (rough). Complements ineff-runs-1
  (excess-bytes floor 2.3 ms); note appended there.
- `ineff-findings-3` (open, v1): split-K partial-result traffic from 14 rows —
  173 MB written/step @32 rows, rough upper-bound saving ≈2400 µs @32, 0 @1/8.
  No overlap found. Outside the slots.

## Files read

- `claude/findings/bottleneck-explained.md` (all): plain-English cost roll-up;
  gave pieces E (1-row effective speeds), F (expert byte arithmetic, per-layer
  distinct table), G (split-K chunk table, 173 MB), H (attention need vs read),
  lower-bound table (5-9x bound).
- `claude/findings/step-breakdown.md` (all): in-step measurements; gave pieces
  A-D (periods, spans, kernels/CPU/knock-outs), I (repeat fixes), J context, K
  (limits), and the file map of the run directory.
- `claude/findings/decode-bottlenecks.md` (all): microbenchmark findings; gave
  piece J (−29% compile), piece F (grouped −25-36%, layout test), piece H
  (GQA miss, head table, +6.4 ms estimate), piece E (867 kernels, 6.81 µs/op),
  piece B (submit profile).
- `results/runs/2026-11-01-breakdown/tables.md` (all): the raw tables behind
  every number in pieces A-E, I (periods, spans, CPU, kernel counts, knock-outs,
  distinct experts, achieved bandwidth, repeat fixes).
- `results/runs/2026-11-01-breakdown/runs-plain.jsonl` (sampled): confirmed
  per-layer `distinct_experts` arrays (e.g. 43,50,...,39 at 32 rows) behind piece
  F; calibration and weight_bytes records behind tables.md header.
- `results/runs/2026-11-01-breakdown/lm_head_shapes.txt` (all): output-layer
  layout test (2.351 vs 2.356 ms, same logits) behind pieces D and H.
- `results/runs/2026-11-01-breakdown/machine-*.txt`, console logs (listed, not
  read in full): machine-state evidence behind piece K; contents as quoted in
  step-breakdown Limits and tables.md.
- `claude/muse-agent-brief.md` (*The task*, *The slots*, *How each attempt is
  judged*): slot inputs/outputs, reference code, G1/G2/G3; slot cost figures
  match pieces C-D.
- `claude/GLOSSARY.md` (all): overhead-/bandwidth-/compute-bound, kernel,
  occupancy, knock-out and other terms used in the entries.
- `bench/micro/moe_parts.py:149-163,190` (read for the runs-3 check): the A/B
  `sorted` flag selects per-pair vs grouped kernel on identically ordered pairs.
- Others' entries (read for adaptation): ineff-runs-1/2/3, ineff-model-2,
  ineff-vllm-metal-1/2/3/4, ineff-quantized-kernels-1/2, ineff-switch-layers-1/2,
  territory-runs (tail), exploration README v2.

## From other agents

(none yet)
