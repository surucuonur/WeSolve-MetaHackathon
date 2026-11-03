# Inefficiency: each (token, expert) pair runs its own qvm grid at ~1/3 peak bandwidth

- **Owner:** explorer-quantized-kernels
- **Status:** open
- **Version:** 2
- **Updated:** 2026-11-03T22:05Z
- **Next:** verifier to confirm dispatch lines and the GB/s arithmetic

## Slot

`experts`

## What happens

Every routed-expert product in decode runs `affine_gather_qvm`
(`quantized.h:2355-2416` + `qvm_impl`, `:1222-1330`): `M=1` rules out the matrix
path and `B/E<=2.1 < 4` rules out the rhs batched path forever at `rows<=32`
(`quantized.cpp:2005-2093 @v0.32.4`; full proof in
`territory-quantized-kernels.md` section A). Consequences, all in this kernel:

1. Each pair re-reads its expert's full weights through its own grid (gate/up 6
   threadgroups, down 12, 64 threads each). At 32 rows, 128 grids per product
   stream 908 MB for 443 MB of distinct experts.
2. Lanes read strided: consecutive lanes sit `out_vec_size_w` apart (gate: 192 B
   stride for 16 B loads, `qvm_impl:1262`), scales `out_vec_size_g` apart (12
   elements, 2 B used per 24 B stride, `:1263-1264`) — poor DRAM coalescing.
3. Each pair's `x` row is re-read 12x per product (every tile's 2 simdgroups
   stream the full K row, `:1275-1287`): ~55 KB x per pair vs 3.75 KB needed,
   ~131 MB/step excess at 32 rows, inside the 908 MB.
4. At 1 row the grids are 24/24/48 threadgroups (~1.5-3K threads, ~5-10% of the
   16-core GPU's slots): occupancy-starved.

## Cost

Per decode step, plain MLX (GPU counters
`results/runs/2026-11-02-gpu-counters/tables.md:17-18`, knock-outs
`results/runs/2026-11-01-breakdown/tables.md:210-213,259-262`):

- 1 row: 60 GatherQMM launches x ~6 us (`step-breakdown.md:179-185`: routed
  experts 140 kernels / 0.92 ms) ~ 0.36 ms; 51 MB read in 0.92 ms = 55 GB/s.
- 8 rows: no serving knock-out; bytes modelled 383 MB pair-traffic
  (540 KB weights + 55 KB x per pair x 32 pairs x 20 layers); micro chained
  qvm products 5.08 ms per 20 layers (`moe_parts.txt:6`, pairs expert-ordered
  in both arms per the notes on `ineff-runs-3` — serving runs unsorted and may
  be slower) = ~75 GB/s. `ineff-findings-2` independently estimates ~5.1 ms.
- 32 rows: 908 MB read in 14.67 ms of a 28.19 ms step = 62 GB/s, 31% of peak.

## Kind

Overhead-bound at all row counts (`GLOSSARY.md`: far below both limits — 62 GB/s
= 31% of peak bandwidth, ~6% of peak compute at 32 rows), with a
bandwidth-waste component (per-pair grids, strided lanes, 12x `x` re-reads).
Arithmetic intensity ~3 FLOP/B << ridge point 26, so a coalesced kernel should
run near peak.

## Spans

`quantized-kernels` (kernel + dispatch) -> slot `experts`. Measurements shared
with `runs`; sort interaction with `switch-layers`.

## Predicted saving

A fused kernel batching same-expert pairs with coalesced reads (the design
`gather_qmm_rhs` was built for but never fires) runs the same traffic at a
normal streaming rate, ~150 GB/s (the output layer hits 158 GB/s where it sits,
`step-breakdown.md:193-195`). Same bytes at 150 GB/s, per decode step:

- 1 row: 51 MB: 0.92 - 0.34 = 0.58, minus ~0.1 ms residual (4 pairs stay small)
  = **0.5 ms**.
- 8 rows: 383 MB: 5.08 - 2.55 = **2.5 ms** (micro baseline, caveat below).
- 32 rows: 908 MB: 14.67 - 6.05 = **8.6 ms**.

Corroboration: the micro's rhs arm beats qvm 25-36% on equal ordered data
(15.02 -> 9.63 ms @32, `ineff-findings-2`), so grouped batching alone wins even
before full coalescing and fusion; this entry's ideal goes further (150 GB/s).
Overlap: complementary to `ineff-runs-1` (removes 465 MB of the 908 for 2.3 ms
at peak; combined ideal 443 MB @150 ~ 3.0 ms, total prize ~11.5 ms, consistent)
and compatible with `ineff-findings-2` (measured rhs basis, 3.67 ms @32 —
floor vs full-ideal, same relationship as runs-1/findings-2). `ineff-runs-3`'s
1.3 ms sorting prize premise fell (notes there; lead to settle) — the grouped
kernel it measured belongs here and in runs-1/findings-2. Launch overhead
shared with `ineff-switch-layers-2` (~0.4 ms, count once).
Caveat at 8 rows: micro used 8 experts and chained calls, not serving's ~20
distinct/layer and glue; serving prize may differ.

## Verdict

(verifier to write)

## From other agents

explorer-findings, 2026-11-03T22:05Z — complementary entries (yours: in-kernel
mechanism; mine ineff-findings-2: full-cost measurement), three items: (1) On
your runs-3 note ("rhs never fires"): true in SERVING (E=60), but in the MICRO
True arm rhs DID fire — the test constructs E=8/16/30 with B=32/64/128 so
B/E=4/4/4.27 ≥ 4 (`moe_parts.txt:6-17`), satisfying the rule; the script comment
(`moe_parts.py:149-150`) and decode-bottlenecks §5 ("pairs in expert order in
both arms") confirm the gap is per-pair vs grouped kernel, not locality. So the
"1.3 ms sorting prize" you share with runs-3 is the kernel prize, not locality;
see my runs-3 note. (2) Saving basis @32 rows: your 8.6 ms (908 MB at 150 GB/s
streaming) vs my 3.67 ms (measured grouped kernel, 25%) — the grouped kernel
also removes the 465 MB excess (→443 MB), so the combined ideal needs care to
avoid double counting; verifier to reconcile. (3) Your 383 MB @8-row traffic
model is a finer basis than my ~5.1 ms interpolation; suggest the verifier use
it (my entry keeps the interpolation as the cost-side estimate).
