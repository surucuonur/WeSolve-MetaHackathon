---
status: accepted
---

# From round 3, an improvement must hold in both phases, decode and prefill, and an attempt may hand prefill-sized inputs to the compiled reference

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** Claude's comparison of every engine version over whole requests ([findings](../findings/muse-round-2.md), section 5) found that the grader and G3 timed only the decode step. They never timed the prefill, the phase in which a request's whole prompt runs through the model:
- **Experts attempt-7** looked like round 2's largest gain, a 19% shorter decode step at 32 requests. It processed prompts 6.6 to 14.5 times slower and cut throughput by 10% to 83%.
- **Routing attempt-4,** round 2's only baseline, processed prompts 7% to 10% slower. At 32 requests with 1,024-token prompts it cost 3.5% of throughput.

The Muse agent never saw either effect: its piece timing ran decode sizes only, and G3 measured the decode step only.

On 2026-11-04 surucuonur set the rule: "The improvement should be in the two phases of it: decode and prefill. Then, we can consider it as an improvement." surucuonur chose:
- **faster in one phase, slower in neither;**
- **the hand-off:** an attempt may leave prefill-sized inputs to the compiled reference. This replaces the part of [ADR 0010](0010-exact-at-every-size.md) that required the kernels to run in every call.

## How it works

- **The piece timing runs at the six G2 sizes** (`bench/grader/check_candidate.py`): decode at 1, 8 and 32 rows, and the first prefill chunk at 127, 1,016 and 2,048 rows. The layers are still chained.
- **The verdict at each size comes from paired repeats.** The candidate and the compiled reference run in alternating order, 30 times, and so does the baseline when one is given.
  - The candidate is `faster` when it wins at least 22 of the 30 pairs, a one-sided sign test at p < 0.01.
  - It is `slower` when it loses at least 22, and `same` otherwise.

  This replaces the comparison of medians for the decode sizes too, since the same kernel timed twice differs by up to 5%.
- **The verdict per phase:**
  - a phase is `faster` when every one of its sizes is;
  - it is `slower` when any size is;
  - otherwise it is `not slower`.

  `beats_compiled`, and `beats_baseline` against a baseline, are true when one phase is `faster` and neither is `slower`.
- **The hand-off** (`max_rows` in `spec.json`, `bench/grader/slots.py`). Inputs with more rows than `max_rows` run the slot's reference under `mx.compile`. G2 still checks every size.
  - On 2026-11-04 the compiled reference of each slot was bit-exact with the reference at all six sizes (`bench/micro/compiled_exact.py`, `results/runs/2026-11-04-handoff-check/`).
  - Above `max_rows` a hand-off runs the compiled reference itself, so it times `same` as the bar there, never better.
- **G3** (`bench/grader/time_candidates.py`, which writes `g3-vllm.json`):
  - **Workloads:** 1, 8 and 32 requests with 128-token prompts, and 32 requests with 1,024-token prompts.
  - **Measurements per run:** the decode step period, the prompt time and the whole-request time, over 5 shuffled repeats, up from 3.
  - **Faster and slower:** the two ranges over the repeats don't overlap.
  - **Why 5 repeats:** with 3 each, two identical engines' ranges miss each other by chance 1 time in 20 per direction. "Never slower" checks a phase 8 times (4 workloads, 2 bars), so a phase that changed nothing would fail about a third of the time. The first run under these rules showed it. A copy of experts attempt-7 with a hand-off at 59 rows came out `slower` on prompt time at 32 × 1,024, by 0.5% with ranges 16 ms apart, although above 59 rows it runs the compiled reference, whose own range overlapped the unchanged engine's (`results/runs/2026-11-04-handoff-check/`). With 5 repeats the chance is 1 in 252 per check, about 3% over 8.
  - **The pass rule:** against both the original model and the compiled reference, the decode step or the prompt time must be faster in every workload, neither may be slower in any, and no whole request may be slower.
- **The unchanged engine is vLLM's own.** The original model in G3 runs with Claude's step hooks, and with vLLM's engine inside the script's process. On 2026-11-04, alternating runs found it within −0.2% to +1.9% of vLLM with neither, inside the spread between repeats in every workload (`bench/micro/clean_vllm.py`, `results/runs/2026-11-04-clean-vllm/`).

## Consequences

- **Round 3 starts with no baseline.** Routing attempt-4 is `slower` at all three prefill sizes, 2.6 to 8.7 times the compiled reference's time. It fails G3 on prompt time at 8 and 32 requests. It becomes a starting point, so round 3's bar in every slot is the compiled reference.
- **A decode kernel no longer has to win at prefill sizes,** only to hand them off or not lose there. The attempt chooses the row count, so the boundary agent checks that the hand-off sits on the right side of 60 rows for the experts slot.
- **At prefill sizes, a slot with a hand-off is MLX's code, not the Muse agent's.** The claim for such an attempt is "faster decode, unchanged prefill".
- **Grader runs and G3 take longer.** Every attempt is now timed at three more sizes. G3 runs 5 repeats instead of 3, and its 32 × 1,024 workload adds at least 16 s per variant per repeat.
- **The decode verdict is stricter.** An attempt whose median beat the compiled reference by noise is now `same`.
- **Not covered:** prefill chunks of other sizes, and decode steps with other row counts. The boundary agent reports them; G3's four workloads are the end-to-end check. Not covered means not measured: from round 3 the swarm lists these as **Unverified**, never as free ([ADR 0013](0013-measure-what-you-claim.md)); round 2 read ADR 0010's line like this one as leave to slow the prefill.
- **At 1 request the whole-request time is noisy.** On 2026-11-04 its repeats spread 6% to 41%, so G3 there mostly says `same`, which passes the "never slower" test without showing a gain.

## Considered options

- **Faster in both phases.** The higher bar. A decode kernel would also have to beat MLX's large matrix kernels at 2,048 rows, which is not where the decode step loses its time. Not chosen.
- **Whole-request throughput alone.** Simple, but it lets a slower phase be traded for a faster one. surucuonur ruled that out.
- **Hand off to the uncompiled reference.** The prefill could then come out `slower` than the compiled reference wherever compiling helps; at 127 rows, routing's compiled reference takes 87 µs per layer against 102 µs uncompiled.
- **Keep the kernels in every call** (ADR 0010). The kernels could still branch on the row count inside themselves, but they couldn't use MLX's large-size kernels. surucuonur chose the hand-off.
