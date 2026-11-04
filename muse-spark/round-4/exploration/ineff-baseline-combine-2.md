# Inefficiency: baseline-combine-2 — per-thread divide and redundant weight loads above the floor

- **Owner:** explorer-baseline-combine
- **Status:** confirmed
- **Version:** 2
- **Updated:** 2026-11-04T21:47:11Z
- **Next:** verifier to check the bound; builders should not chase it without a measurement above paired noise

## Slot

`combine`.

## What happens

Above the dispatch floor (`ineff-baseline-combine-1`), the
kernel body in the slot
(`muse-spark/round-3/kernels/combine/attempt-2/block-0.metal:26-37`)
still does two things per thread that carry no unique
information:

1. `row = idx / 768u` (`:27`) — a 32-bit integer divide by a
   non-power-of-two, once per output element (768 divides per
   row per layer call), plus the `row*3072` / `row*768`
   multiplies that follow from it (`:31-32`).
2. `weights[wbase .. wbase+3]` (`:34-37`) — the same 4 float32
   weights of a row loaded independently by all 768 threads of
   that row: 768 × 4 loads × 4 B = 12,288 B of logical load
   traffic per row for 16 B of unique data. The bytes hit L1
   as a broadcast (unique DRAM traffic stays 16 B/row), so the
   cost is load-issue slots and latency, not DRAM bytes.

Everything else in the body is minimal: 4 strided-coalesced
bf16 loads, 7 f32 ops in the frozen fold, 1 bf16 load, 1 store
(territory entry, section B). Round 2 bounded this same
micro-path for the predecessor body and posted no entry
(`muse-spark/round-2/exploration/territory-baseline-combine.md`
checklist E, frozen: expected saving <1 µs/layer, inside
run-to-run spread). This entry re-derives the bound for the
pragma kernel and posts it so the round-4 decode-margin
question has a verified ceiling instead of folklore.

## Cost

Per layer (peaks from `claude/GLOSSARY.md`; measured context:
`muse-spark/round-3/kernels/combine/attempt-2/grade.json`,
candidate 31.8 / 34.7 / 32.7 µs decode, 38.2 / 100.9 / 198.0
prefill):

- Kernels × fixed cost: unchanged — these terms add no launch;
  they live inside the single kernel whose floor is
  `ineff-baseline-combine-1`.
- Bytes ÷ peak: unique traffic unchanged at 9,232 B/row
  (**0.05 / 0.37 / 1.48 µs** decode; **5.86 / 46.9 / 94.5 µs**
  prefill). The redundant weight loads add no DRAM bytes;
  priced (incorrectly, as an upper bound) at DRAM peak they
  would be 12,288 B/row = 0.06 / 0.49 / 1.97 µs decode and
  7.8 / 62.4 / 125.8 µs prefill — but they are served from L1
  at far above DRAM bandwidth, so the true byte cost is a
  small fraction of that bound; the divide is pure ALU
  (rows×768 divides per layer call: 768 / 6,144 / 24,576 at
  decode, 97,536 / 780,288 / 1,572,864 at prefill).
- Measured: no isolated measurement exists — the terms are
  inside the 31.8–198.0 µs/layer totals above and below the
  ~5% same-kernel paired noise the brief reports
  (`claude/muse-agent-brief.md` *The piece timing*: ~1.6 µs
  at decode levels). Two controls bound them from outside:
  round 1's staged/vectorised organisation (weights staged
  once, u64 loads, one barrier) measured the same floor as the
  unstaged kernel (17.9 / 16.1 / 18.1 vs 18.6 / 17.0 / 18.9
  µs/layer, independent basis, frozen citation in the
  territory entry), and round 3's two barrier mechanisms
  (call vs pragma, same indexing) both won every size on
  same-run verdicts without either showing an indexing effect.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): neither bytes at peak
nor FLOPs at peak; issue/latency slivers under a dispatch
floor.

## Spans

`baseline-combine` only (index lines and load lines of the
single kernel). No other territory's code is touched by any
fix here.

## Predicted saving

Removing both terms bit-exactly is possible in principle —
row from `thread_position_in_grid.y` with a 2-D grid
(`slots.py:283-293` admits literal × `rows` products per grid
axis), weights staged once per threadgroup, or one thread per
2–4 outputs reusing one weight load set — and each variant
keeps the frozen 7-op fold untouched. Predicted saving if
removed (prediction, from the L1-bound above discounted by
the two controls): **≤0.5 µs per layer at decode 1 / 8 / 32
rows (≤0.01 ms per decode step) and ≤2 µs per layer at
prefill 127 / 1,016 / 2,048 rows**, i.e. at most a tenth of
the thinnest decode margin (5.1 µs/layer at 1 row) and below
paired-timing discrimination (needs ≥22/30 wins;
`bench/grader/check_candidate.py:59-63`). No phase trades
against the other: the same lines run at all sizes, and the
prefill ceiling is larger in µs but irrelevant to the
32×1024 set margin, which is decided in decode steps. The
entry's use is therefore negative: it caps what a
margin-widening combine attempt could win, so attempts are
not spent proving it kernel by kernel.

## Verdict

verifier, 2026-11-04T21:47:11Z — **confirmed**.

- **confirmed:** body terms in [block-0.metal:26-37](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/combine/attempt-2/block-0.metal:26) — `row = idx / 768u` once per output (768 divides/row/layer), `weights[wbase..wbase+3]` re-loaded by all 768 threads/row = 768×4×4 B = 12,288 B logical for 16 B unique, L1 broadcast not DRAM ✓; remainder of body (4 bf16 loads, 7-op fold, shared load/store) minimal per territory §B ✓; same micro-path bounded <1 µs/layer in frozen round-2 [territory-baseline-combine.md](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/exploration/territory-baseline-combine.md) checklist E ✓.
- **confirmed:** bounds arithmetic — unique traffic unchanged 9,232 B/row = 0.05/0.37/1.48 µs decode, 5.86/46.9/94.5 µs prefill at 200 GB/s ([GLOSSARY.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/GLOSSARY.md)); redundant loads priced (as stated, incorrectly) at DRAM peak = 12,288 B/row → 0.06/0.49/1.97 µs decode, 7.8/62.4/125.8 µs prefill, true L1 cost a small fraction ✓; divides 768/6,144/24,576 decode, 97,536/780,288/1,572,864 prefill ✓; no isolated measurement claimed — terms inside [grade.json](/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-3/kernels/combine/attempt-2/grade.json) totals, below ~5% paired noise (~1.6 µs decode, [muse-agent-brief.md](/Users/onur/Desktop/Github/Hackathon-muse/claude/muse-agent-brief.md) piece timing) ✓; controls check: staged/vectorised 17.9/16.1/18.1 vs unstaged 18.6/17.0/18.9 µs/layer same floor, both round-3 barrier mechanisms winning without indexing effect ✓; 2-D grid expressible via [slots.py:283-293](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/slots.py:283) `dim()` ✓.
- **confirmed:** predicted saving **≤0.5 µs/layer decode (≤0.01 ms/step), ≤2 µs/layer prefill** labelled prediction — a tenth of the thinnest 5.1 µs/layer margin, below 22/30-win discrimination ([check_candidate.py:59-63](/Users/onur/Desktop/Github/Hackathon-muse/bench/grader/check_candidate.py:59)), no phase trade (same lines all sizes) ✓; negative use (cap, do not chase) follows ✓.
