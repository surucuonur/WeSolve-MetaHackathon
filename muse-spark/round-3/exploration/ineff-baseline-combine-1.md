# Inefficiency: baseline-combine-1 — volatile rounding barriers cost ~1.4 ms/step in the step, ~0 in the grader

- **Owner:** explorer-baseline-combine
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T17:44:55Z
- **Next:** verifier to check; builders: exactness without volatiles (e.g. noinline barriers)

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

(none — open)
