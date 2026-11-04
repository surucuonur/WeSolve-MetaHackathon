# Inefficiency: baseline-combine-1 — volatile rounding barriers cost ~1.4 ms/step in the step, ~0 in the grader

- **Owner:** explorer-baseline-combine
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T17:50:16Z
- **Next:** done — confirmed by verifier (see Verdict); net-vs-compiled corrected to -0.2/-0.4/-0.1

## Slot

`combine`.

## What happens

The code in the slot
(`muse-spark/round-2/kernels/combine/attempt-1/block-0.metal:33-39`)
enforces the reference's per-op rounding with seven `volatile float`
thread variables (`p0..p3`, `s01`, `s012`, `acc`). By `volatile`
semantics each is a real memory access — 7 stores + 7 loads per output
element, 14 accesses × 4 B = 56 B of thread-private stack traffic per
thread — that the compiler cannot keep in registers, forward, elide, or
reorder. Everything else about the kernel is identical to round-1
attempt-1: the spec is byte-identical (empty `diff`), the index/base
lines and the final `bfloat(acc) + shared` line are verbatim; only the
sum's middle lines changed (one contracted expression → seven volatile
steps; `territory-baseline-combine.md` checklists A–B).

In the grader this kernel wins at all six sizes (decode 45.7 / 47.9 /
64.6 vs compiled 53.4 / 56.4 / 69.1 µs/layer). In the step it loses
+1.051 ms at 32 requests where its volatiles-free twin saved −0.390 ms
(`claude/findings/muse-round-2.md` §3 vs `claude/findings/muse-round-1.md`
§3). The twin swing — the only code delta is the barriers — is the
barriers' in-step cost (table under Cost). The prefill twin in
`ineff-g3-1` (+1501.7 ms on the 32×128 all-three prompt where the
prefill piece implies a saving) is the same barriers at prefill thread
counts (up to 88 MB/layer of spill bytes at 2,048 rows).

Ruled out as the main cause (detail in the territory entry): the G3
install path (lazy for candidate and compiled alike, checklist D — no
per-layer eval or sync exists to blame); hidden contiguity copies (exact
−80-kernel op balance, and any copy would hit the twin equally,
checklist E); grid/occupancy/wave shape (spec identical across the
twin, so shape cannot discriminate); tokens and noise (bit-exact at all
6 + 25 sizes and end to end; G3 ranges disjoint in both engines).

## Cost

Per layer, at decode 1 / 8 / 32 rows and prefill 127 / 1,016 / 2,048
rows (peak 200 GB/s, `claude/GLOSSARY.md`):

- Kernels × fixed cost: 1 kernel/layer either way — the barriers add no
  launch; the chained floor (~32–40 µs/layer decode) is shared with the
  twin. Not the cost.
- Bytes ÷ peak: slot traffic 9,232 B/row plus spill traffic 56
  B/thread = rows×43,008 B/layer, i.e. 52,240 B / 418 KB / 1.67 MB at
  decode = **0.26 / 2.1 / 8.4 µs**, and 6.6 / 53 / 107 MB at prefill =
  **33 / 265 / 533 µs**. Spill bytes alone: 43 KB / 344 KB / 1.375 MB
  decode, 5.5 / 43.7 / 88 MB prefill.
- Measured (the twin swing, round-2 G3 minus round-1 G3, per decode
  step over 20 layers; vLLM medians, plain in brackets):
  1 row **+0.142 (+0.433)** ms, 8 rows **+0.508 (+0.641)** ms, 32 rows
  **+1.441 (+1.574)** ms — i.e. **+7.1 / +25.4 / +72.1 µs/layer**
  (+21.7 / +32.1 / +78.7 plain). Cross-run drift is bounded at ±0.1 ms
  by the control both runs timed (round-1 triple within 0.06–0.09 ms;
  territory checklist B), so the 8- and 32-row swings are signal; the
  1-row swing is within 1-row cross-run noise and claimed as ≈0.
  At 32 rows the paid +72.1 µs/layer is ~9× the spill bytes' 8.4 µs
  bandwidth time — the cost is stalls on thread-private traffic
  (dependent store→load pairs the compiler cannot schedule across,
  fetched in cache lines far wider than the 4 B values), not bytes.

Why the grader prices it near zero is only partly understood: its
20-layer link chain runs the kernel against a quiet cache with nothing
but the tiny `link` kernel between layers, while the step buries each
call between megabytes of neighbouring traffic; the store→load reuse
that the quiet chain absorbs on-chip stalls in the busy step. The exact
on-chip accounting (L1 absorption, line amplification on strided
frames, register pressure) needs GPU counters and is **unverified** —
but the attribution does not depend on it: same spec, same grid, same
inputs, only delta the barriers, 1.44 ms swing with a ±0.1 ms control.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): at decode 32 the kernel pays
+72 µs/layer for 1.375 MB of thread-private traffic (6.9 µs at peak)
and ~0.03 µs of arithmetic — far below both limits, the time going to
stalled memory traffic that reads no weights and does no arithmetic.

## Spans

`baseline-combine` (the barrier lines), `g3` (`ineff-g3-1` owns the
measurement side; this entry owns the mechanism), `grader` (the chained
timing cannot see context-dependent stall costs — note in round-3
`territory-grader.md`).

## Predicted saving

Remove the volatiles while keeping the exact per-op rounding — e.g.
`__attribute__((noinline))` device-function barriers for the mul and
the adds, which the compiler cannot contract across and which pass
values in registers with no memory traffic (expressible in the spec
surface: helpers in `header_block`, calls in the body; `slots.build`
supports both, `slots.py:313-322`). Predicted effect, labelled
prediction:

- Decode step (vLLM basis): recover the twin swing, i.e. return to
  ≈ −0.51 / −0.71 / −0.39 ms/step: **≈0.5 ms/step at 8 requests and
  ≈1.4 ms/step at 32 requests**; ≈0 claimed at 1 row (within noise).
- Prefill: the same barriers at prefill thread counts drive the
  +1501.7 ms prompt twin (`ineff-g3-1`); removing them is predicted to
  recover it (order 1 s on 32×128 combine-including prompts) — no
  round-1 prefill twin exists to calibrate against, so this is the
  looser half of the prediction.
- Grader basis: the kernel already beats compiled everywhere; fewer
  stalls can only improve the chained time, so `beats_compiled` is
  predicted kept with a wider margin — the fix risks exactness (G2),
  never speed, and a saving in decode cannot cost prefill here since
  both phases run the identical lines.
- Net vs the compiled reference if the fix lands: ≈ −0.5 / −0.9 /
  −0.2 ms/step at 1 / 8 / 32 requests (twin's engine numbers), plus
  the prompt recovery above.

Discriminating checks: the ready check must time the fix combine-alone
at 32 requests (reproduces-or-refutes in-round); G3-timing round-1
attempt-1 in the same run would isolate the barriers in-run but the
fix's own G3 supersedes it.

## Verdict

verifier, 2026-11-04T17:50:16Z — **confirmed**. Twin control re-derived from both findings write-ups and both consoles; four corrections, one (net-vs-compiled) builders must take from here.

- **confirmed:** seven `volatile float` (p0..p3, s01, s012, acc, block-0:33-39) = 7 stores + 7 loads/output = 56 B/thread; 768×56 = rows×43,008 B/layer ✓; totals 52,240 B / 418 KB / 1.67 MB decode and 6.6 / 53 / 107 MB prefill ✓; floors 0.26/2.1/8.4 and 33/265/535 µs (written 533 — 2 µs rounding, immaterial); spill alone 43 KB/344 KB/1.375 MB and 5.5/43.7/88 MB ✓.
- **confirmed:** twin is volatiles-only delta — spec diff empty (verified), index/base lines (25-31) and final `bfloat(acc)+shared` (41) verbatim, only sum lines 20-23 vs 33-39 differ (verified by diff) ✓; grid/occupancy therefore cannot discriminate ✓.
- **confirmed:** twin swing — round-1 twin G3 −0.514/−0.705/−0.390 vLLM (muse-round-1.md §3 table) and −0.857/−0.753/−0.406 plain; round-2 +1.051/+1.168 at 32 etc. (consoles): swings +0.142/+0.508/+1.441 vLLM and +0.433/+0.641/+1.574 plain recomputed ✓; per-layer +7.1/+25.4/+72.1 and +21.7/+32.1/+78.7 (÷20) ✓; +72.1 ≈ 9× the 8.4 µs spill floor (8.6×) ✓.
- **confirmed:** vLLM control — round-1 triple +0.980/+0.163/−1.756 then (findings §3) vs +0.923/+0.127/−1.849 now (round-2 vLLM console): drifts 0.057/0.036/0.093, all < 0.1 ✓, 15× below the 1.44 ms swing ✓; 1-row ≈0 is a conservative no-claim ✓.
- **confirmed:** ruled out — install lazy both sides (time_candidates.py:65-66 lambda + make_moe_call; compiled via mx.compile :105,119; no eval in install) ✓; exact −80 balance (pairs jsonl 866→786 rows 1, 986→906 rows 32) with twin-equal-copy logic ✓; exactness 6 + 25 + end-to-end tokens (verified under ineff-g3-1) ✓; G3 ranges disjoint both engines (vLLM verified under g3-1; plain [29.499–29.677] vs [28.434–28.609] disjoint) ✓.
- **confirmed:** grader-wins numbers (45.7/47.9/64.6 vs 53.4/56.4/69.1, JSON) ✓; chained floor ~32–40 (round-3 README) ✓; kind 1.375 MB → 6.9 µs, ~0.03 µs arithmetic (order ✓); quiet-vs-busy-cache story honestly labelled partly-unverified with attribution independent of it ✓; prefill twin +1501.7 (verified under g3-1) ✓; no round-1 prefill twin exists (prefill untimed then, ADR 0010:44) ✓.
- **confirmed:** saving — return to ≈−0.51/−0.71/−0.39 (twin's medians) ✓; recover ≈0.5/≈1.4 ms at 8/32, ≈0 at 1 ✓; prefill order-1-s recovery honestly the looser half ✓; `beats_compiled`-kept and no-decode-costs-prefill (identical lines all sizes — no size branch in block-0) labelled prediction ✓; noinline-helper fix expressible via `header_block` (slots.py:313-322) ✓.
- **wrong (figures — use corrected values):** "Net vs the compiled reference if the fix lands: ≈ −0.5/−0.9/−0.2" — underivable from the twin's numbers. Same-run vLLM twin−compiled: (−0.514)−(−0.288) = −0.226, (−0.705)−(−0.260) = −0.445, (−0.390)−(−0.272) = −0.118, i.e. ≈ **−0.2/−0.4/−0.1 ms/step** — consistent with the findings' own verdict (clean G3 pass vs compiled only at 8; overlap at 1, within spread at 32). The swing-recovery prize is unaffected.
- **wrong (span):** control "within 0.06–0.09 ms at every size" — drifts are 0.057/0.036/0.093, i.e. 0.06/**0.04**/0.09. The ±0.1 bound holds; the middle value is smaller than stated.
- **wrong (scope):** the ±0.1 control bound is vLLM-only — in plain the same round-1-triple control drifts 0.127/0.107/0.335 (round-2 plain +1.054/+0.774/−1.895 vs round-1 plain +0.927/+0.881/−1.560). Plain 8/32-row swings (+0.641/+1.574) still clear it; only the blanket "±0.1" needs the engine qualifier.
