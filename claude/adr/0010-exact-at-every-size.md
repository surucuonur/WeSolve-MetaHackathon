---
status: accepted
---

# From round 2, a kernel must be bit-exact at every size it runs at, prefill included, and the piece timing chains the layers

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** Claude's measurement of round 1 ([findings](../findings/muse-round-1.md)) found that round 1's final attempts, which passed the grader, broke the brief's end-to-end rule in the model, and that one of them slowed the decode step it was meant to speed up. The grader missed three things:

1. **Sizes.** It tested a slot on a decode step's inputs only, at 1, 8 and 32 rows. The same `MoE.__call__` runs the prefill, at 127 to 2,048 rows with 128-token prompts. From 60 rows on, MLX's own experts kernel changes, from `gather_qvm` to `gather_qmm_rhs`, and so does its rounding. Round 1's experts kernel copied `gather_qvm` bit for bit, and differed in 62% of its outputs at each prefill size.
2. **Exactness.** G2 allowed small differences: 2 bfloat16 units in the last place, and 10⁻⁴ for routing weights. End to end, the brief asks for the same greedy tokens, and greedy decoding turns any difference into different tokens. Round 1's routing weights, within 10⁻⁵ of the reference, changed 7 to 9 of 32 sequences in 24 tokens.
3. **Latency.** The piece timing ran 20 layers' calls with no data between them, so the GPU could overlap them. At 1 row, round 1's experts kernel was 29% faster than the reference on overlapped calls and 66% slower on chained ones (`bench/micro/chain_timing.py`). In the model it slowed the decode step by 1.2 ms (22% in vLLM, 25% in plain MLX).

On 2026-11-04 surucuonur chose, before round 2:
- **every call:** the kernels run wherever `MoE.__call__` runs, the prefill included;
- **bit-exact** outputs;
- **chained** piece timing.

## How it works

- **G2 at two kinds of size.** `bench/grader/check_candidate.py` checks every layer at:
  - the captured decode inputs, at 1, 8 and 32 rows;
  - the first prefill chunk at 1, 8 and 32 requests (127, 1,016 and 2,048 rows), recorded while it runs (`slots.capture_prefill`), with the prompts `time_candidates.py` uses.
- **Bit-exact.** An output passes only if it equals the reference's:
  - for routing, the 4 experts and their weights, in the same order, since combine sums the pairs in that order;
  - for combine and experts, every element.

  The grader still reports the largest gap, in units in the last place, to help find the cause. The reference checked against itself shows no difference at any size.
- **Chained timing.** Each layer's call takes one input through `link`, a single fused kernel that adds 0 × one value of the previous call's output, so the call can't start before the previous one finishes. Every function timed (reference, compiled, candidate and baseline) gets the same link. The chained times match those of a real data chain within 10%.
  - **Why not `mx.depends`:** it orders only the graph. In a test on 2026-11-04, round 1's experts kernel took 46 µs per layer with it at 1 row, as fast as with no dependency, against 133 to 195 µs with a real data dependency in three runs.
- **The end-to-end check and G3 are unchanged.**
  - `bench/grader/time_candidates.py` now takes several `SLOT=FOLDER` attempts.
  - It times each slot alone, against its own compiled reference, and all of them together.

## Consequences

- **No round-1 attempt passes the new G2,** so round 2 has no baseline in any slot. Its bar is the compiled reference. Round 1's final attempts are its starting points, named in Claude's findings.
- **The experts slot needs a kernel exact at prefill sizes,** matching `gather_qmm_rhs`'s rounding from 60 rows on, or a single design whose rounding matches both of MLX's kernels.
- **A grader run takes longer.** It loads the model (760 MB) for every slot and runs three prefill chunks.
- **The piece timing now includes one small link kernel per layer** for every function timed. That brings the reference and the candidate closer in relative terms, but not in µs.
- **Not covered:** prefill speed. G3 times decode steps only, so a kernel slower at prefill sizes would go unnoticed, apart from the time to first token.

## Considered options

- **Decode calls only,** with MLX's code in the prefill. The kernels would need to be exact only at decode sizes, but they would no longer replace the whole MoE block. Not chosen: surucuonur asked for every call.
- **Keep the tolerances, and relax the token rule** to counting the sequences that differ. Simpler kernels, but the claim would become "the same model up to rounding", with a threshold to pick.
- **Keep independent timing,** and rely on G3 to catch latency. Round 2's builders would again rank attempts by a number that hid a 22% slowdown.
- **`mx.depends` for the chain.** No extra kernel, but it doesn't make the GPU wait.
