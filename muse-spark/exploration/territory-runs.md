# Territory: runs

- **Owner:** explorer-runs
- **Status:** done
- **Version:** 2
- **Updated:** 2026-11-03T21:28:00Z
- **Next:** none — answered no notes (none posted); awaiting verifier

## What it does

This territory is the measurements, not the code: the three run folders that say where each decode step's time and bytes go. The baseline roofline sets the step budget, the micro tables split the MoE block and count the fixed per-kernel cost, and the GPU-counter tables give achieved bandwidth, occupancy, and per-piece knock-out costs.

## Checklist

### results/runs/2026-10-29-baseline/summary.md (roofline table)

- **Callers:** produced by `bench/roofline.py` from the timed/experts client runs in the same folder; consumed by `claude/findings/` and the brief's slot costs.
- **Callees:** none (a table).
- **Data:** one row per (input_len, concurrency): step ms median with run-median range, submit call ms, submit CPU ms, decode tok/s, distinct experts/layer, MB read/step, GB/s, % of peak, FLOP/byte, min ms at peak. Key rows (in128): c1: 5.49 ms, 32.6 GB/s (16.3%), 3.14 FLOP/byte, 4.0 experts; c8: 14.51 ms, 26.8 GB/s (13.4%), 11.55 FLOP/byte, 19.6 experts; c32: 30.04 ms, 24.0 GB/s (12.0%), 24.93 FLOP/byte, 38.2 experts.
- **Neighbours:** before it, the raw `steps-timed.jsonl`/`runs.json`; after it, the findings that cite it. A replacement reading must keep the same workload (128-token prompts, steady decode steps).
- **Cost:** the step itself: 5.49 ms at c1, 14.51 ms at c8, 30.04 ms at c32 (in128). Submit call ms is nearly the whole step (5.16/5.49 at c1; 29.23/30.04 at c32); submit CPU ms is 4.73 at c1 and 6.89 at c32.
- **Invariants:** steady decode steps only; medians over runs; concurrency caps decode rows but decode rows are what matter (see `steps-timed.jsonl` `rows` field).

### results/runs/2026-10-29-baseline/roofline.json, runs.json, steps-*.jsonl, meta.json

- **Callers:** `summary.md` is generated from these.
- **Callees:** none.
- **Data:** `roofline.json`: per-point step ms, bytes split (every_step 67.6 MB, output 61.6 MB, experts, cache, embedding), FLOPs, achieved GB/s, arithmetic intensity. `meta.json`: M1 Pro, 16 GB, MLX 0.32.4, model MobileMoE-S-QAT-mlx-bfloat16. `steps-timed.jsonl`: per-step build_ms, build_cpu_ms, rows, ctx.
- **Neighbours:** raw material behind the summary table.
- **Cost:** same step budget as above; `roofline.json` costs block gives expert_bytes 552,960 B and cache_bytes_per_token 20,480 B.
- **Invariants:** same workload as the summary.

### results/runs/2026-10-30-micro/step_parts.txt

- **Callers:** microbenchmark script output; feeds the fixed-cost-per-kernel claim.
- **Callees:** none.
- **Data:** tiny dependent op 6.81 µs each (the fixed cost of one small kernel); output layer 0.931 ms at R=1 (66 GB/s) and 2.644 ms at R=32 (23 GB/s); pipelined decode step R=1: 5.04 ms, R=8: 13.85 ms, R=32: 28.91 ms; graph build CPU ~1.0–1.3 ms.
- **Neighbours:** `step_repeats.txt` (medians 4.98 / 13.62 / 28.73 ms) confirms the pipelined step times.
- **Cost:** the fixed ~6.8 µs per tiny kernel is the unit every kernel-count saving multiplies.
- **Invariants:** plain MLX engine, chained eval; repeats in `step_repeats.txt` show ±5% spread.

### results/runs/2026-10-30-micro/moe_parts.txt

- **Callers:** feeds the experts/sort/combine cost split.
- **Callees:** none.
- **Data (ms per 20 layers):** R=1: moe 6.02, moe_compiled 4.27, experts 3.64, shared 1.85, attn_proj 1.08; experts at peak 0.22 ms; per-pair reads 44 MB. R=32: moe 21.56, moe_compiled 20.51, experts 16.60, shared 3.89; experts at peak 2.64 ms; per-pair reads 1416 MB. Kernel A/B sort experiment: rows 32, ours sorted=False 15.02 ms vs sorted=True 9.63 ms per 20 layers; rows 8: 5.08 vs 3.82; rows 16: 8.12 vs 5.65.
- **Neighbours:** `step_parts.txt` gives the whole step these pieces sit in; the gpu-counters knock-out table gives the independent per-piece check.
- **Cost:** experts is ~16.6/21.6 of the MoE block at R=32; sorting pairs by expert saves ~36% of experts time at rows 32 (15.02 → 9.63) and ~25% at rows 8 (5.08 → 3.82).
- **Invariants:** one layer's ops chained ×20; distinct/layer differs from serving (47.8 at R=32 here vs 38.2 in baseline).

### results/runs/2026-10-30-micro/attention_heads.txt

- **Callers:** feeds the attention K/V re-read claim (outside the slots).
- **Callees:** none.
- **Data (µs/layer):** rows 32, ctx 1152: 12q/4kv 568.7 (66.4 GB/s on unique K/V) vs 4q/4kv 247.1 (152.8 GB/s) — same K/V bytes, ~2.3× the time with 3× the query heads. Rows 1, ctx 256: 26.0 vs 18.7 µs/layer.
- **Neighbours:** the knock-out table's attn_kernel row (saves 3.12 ms at 32 rows) bounds the total attention prize.
- **Cost:** the extra query heads re-read K/V instead of sharing it.
- **Invariants:** per_token kernels; ctx 1152 rows are the high-ctx case, above the baseline's mean ctx.

### results/runs/2026-10-30-micro/step-primitives.txt, step_cpu.txt, profile_submit.log, machine.txt, step.dot

- **Callers:** supporting detail.
- **Callees:** none.
- **Data:** primitive counts per step: 141 QuantizedMatmul, 60 GatherQMM, 20 ArgPartition, 20 SDPA, plus ~700 more elementwise/reshape/transpose ops. `step_cpu.txt`: R=1 async_eval CPU 2.65 ms then GPU wait 3.72 ms; R=32: 4.27 ms then 14.84 ms. `machine.txt`: memory pressure during the run (33% free at start).
- **Neighbours:** the primitive counts corroborate the brief's ~870–1,090 kernels/step.
- **Cost:** no direct timing; counts × 6.81 µs (step_parts.txt) give the overhead scale.
- **Invariants:** counts are lazy-graph primitives, not launched kernels; fusion changes the mapping.

### results/runs/2026-11-02-gpu-counters/tables.md (whole-step table)

- **Callers:** generated by `bench/micro/gpu_counters.py tables`; the brief's slot costs cite the knock-out half.
- **Callees:** `counters-summary.json` / `counters.jsonl` (per-trace counter means).
- **Data (plain, p128):** r1: 4.75 ms unrecorded, GPU busy 96%, read 41 GB/s, ALU 17%, Compute Occupancy 9%. r8: 13.37 ms, busy 98%, 51 GB/s, ALU 45%, occupancy 23%. r32: 28.19 ms, busy 92%, 55 GB/s, ALU 37%, occupancy 26%. F16 utilization 0% everywhere; Buffer Read Limiter 12/34/53%.
- **Neighbours:** baseline summary.md gives the vLLM-engine step times these plain-MLX times sit beside (vLLM rows included in the same table: 5.64 / 14.30 / 30.47 ms).
- **Cost:** the step reads 193/685/1561 MB while the model must read 180/393/731 MB — ~2.1× the necessary bytes at r32.
- **Invariants:** 1-second recordings; calibration factor 2.002 on the read counter; MB-read uses recorded steps, time-saved uses unrecorded medians.

### results/runs/2026-11-02-gpu-counters/tables.md (knock-out table, plain MLX)

- **Callers:** the brief's per-slot costs (routing 1.02/1.68 ms, experts 0.92/14.67 ms).
- **Callees:** unrecorded savings come from `results/runs/2026-11-01-breakdown/` (findings territory).
- **Data (step ms saved, unrecorded | MB read saved):** r1: routing 1.02 | 4; experts 0.92 | 51 (must-read 44, per-pair 44); shared 0.51 | 45; lm_head 0.39 | 65; attn_kernel 0.41 | 11; qkv 0.39 | 14. r32: routing 1.68 | 68; experts 14.67 | 908 (must-read 443, per-pair 1416); shared 2.90 | −51 (noisy); lm_head 2.50 | 197; attn_kernel 3.12 | 265; qkv 1.67 | −11 (noisy). GB/s while running: routing 4, experts 62/55, lm_head 167/79.
- **Neighbours:** `moe_parts.txt` corroborates experts' size; negative read savings (qkv, shared at r32) flag ±10% noise at 32 rows.
- **Cost:** experts is 14.67 of 28.19 ms at r32; routing costs 1.02 ms while reading 4 MB at r1.
- **Invariants:** one 1-second recording per setting (±10% at 32 rows); lead-in GPU contamination noted for some r1/r32 recordings; combine is not knocked out (no direct combine row — gap).

## Relations

- Feeds: every territory — these tables are the cost source for all inefficiency entries.
- Fed by: `results/runs/2026-11-01-breakdown/` (findings territory) for the unrecorded knock-out savings.
- Gaps for other territories: no knock-out of `combine`; no 8-row knock-outs; no per-kernel breakdown inside experts (quantized-kernels territory owns that).

## Inefficiencies

- [ineff-runs-1](ineff-runs-1.md) (experts weight re-read per pair) — open
- [ineff-runs-2](ineff-runs-2.md) (routing is pure overhead) — open
- [ineff-runs-3](ineff-runs-3.md) (unsorted pairs below 16 rows) — open
- [ineff-runs-4](ineff-runs-4.md) (attention K/V re-read per query head, outside slots) — open

## Files read

- `results/runs/2026-10-29-baseline/summary.md` — roofline step budget, achieved bandwidth, distinct experts
- `results/runs/2026-10-29-baseline/meta.json` — machine, versions, workload grid
- `results/runs/2026-10-29-baseline/roofline.json` (sampled) — bytes/FLOP splits behind the summary
- `results/runs/2026-10-29-baseline/steps-timed.jsonl` (sampled) — per-step rows/build_ms schema
- `results/runs/2026-10-30-micro/step_parts.txt` — 6.81 µs fixed kernel cost, pipelined step times
- `results/runs/2026-10-30-micro/moe_parts.txt` — MoE/experts/shared split, sort experiment
- `results/runs/2026-10-30-micro/attention_heads.txt` — per-head K/V re-read
- `results/runs/2026-10-30-micro/step-primitives.txt` — op counts per step
- `results/runs/2026-10-30-micro/step_cpu.txt` — CPU submit vs GPU wait split
- `results/runs/2026-10-30-micro/step_repeats.txt` — repeat medians
- `results/runs/2026-10-30-micro/machine.txt` — machine state caveat
- `results/runs/2026-10-30-micro/profile_submit.log` (sampled) — vLLM server setup, not a cost source
- `results/runs/2026-11-02-gpu-counters/tables.md` — whole-step counters, knock-out costs
- `results/runs/2026-11-02-gpu-counters/counters-summary.json` (sampled) — per-trace counter detail
- `results/runs/2026-11-02-gpu-counters/counters.jsonl` (sampled) — trace index

## From other agents

(none yet)
