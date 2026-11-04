---
status: accepted
---

# From round 4, candidates that pass together are ready as a set, and an attempt may hand off small inputs too

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** Round 3 showed three gaps in the rules ([Claude's measurement](../findings/muse-round-3.md)). On 2026-11-04 surucuonur chose to fix all three before round 4:

1. **Ready as a set.** Round 3's routing and combine candidates each failed the ready check alone, on overlapping ranges, while the two together (`muse-all`) passed it. The skill made a candidate `ready` only when it passed alone, so the round submitted nothing. From round 4, when `muse-all` passes the ready check, all its candidates are `ready` together as a **set**. A set goes to G3 whole, and G3 judges it as `muse-all`.
2. **Hand off below a row count too: `min_rows`.** `max_rows` already sends inputs with more rows to `mx.compile` of the slot's reference ([ADR 0011](0011-decode-and-prefill.md)). The new `min_rows` in an attempt's `spec.json` does the same for inputs with fewer rows, so the kernels run only from `min_rows` to `max_rows`. Round 3's experts attempt-1 lost to the compiled reference only at 1 row: 96.9 against 86.6 µs per layer, while it took 168.9 against 238.2 at 8 rows and 554.7 against 883.4 at 32. Its only way to pass was to win at the size it is worst at.
3. **A handed-off size isn't timed.** The grader timed each handed-off size, which runs the compiled reference, against the compiled reference: the same code twice. In round 3 that noise judged routing attempt-1 `slower` at 2,048 rows, on 8 wins of 30, and cost a re-grade of a byte-identical copy. Now such a size is reported `handed off`. A phase is `faster` when every size the kernels run at is faster, and it needs at least one; `beats_compiled` is still one phase `faster` and none `slower`. G2 still checks every size, bit for bit.

## How it works

- **`bench/grader/slots.py`:**
  - `build()` reads `min_rows` as it reads `max_rows`: an integer of at least 1, at most `max_rows` when both are given.
  - The function it returns has `hands_off(rows)`, which says whether inputs with that many rows run the compiled reference.
- **`bench/grader/check_candidate.py`:**
  - At each size, a function that hands the rows off is not timed. It takes the compiled reference's times, so for `routing+combine` the part that hands off adds the same time to both sides.
  - A bar's verdict is `handed off` when neither side ran code of its own. Against a baseline that runs its own kernels there, the size is timed as before.
  - `judge()` leaves `handed off` sizes out of a phase. A phase whose sizes are all handed off is `handed off`.
- **`bench/grader/test_hand_off.py`:**
  - checks `build()` with a toy kernel at row counts on both sides of each bound, bit for bit, and that bad bounds are refused;
  - checks `judge()` on five sets of verdicts.
  - Passed on 2026-11-04.
- **A real attempt.** Claude re-graded round 3's routing attempt-6 (`max_rows` 32) with the changed grader ([results/runs/2026-11-04-grader-hand-off/](../../results/runs/2026-11-04-grader-hand-off/)):
  - G2 passes;
  - decode is `faster` at 1, 8 and 32 rows, on 29, 26 and 27 wins of 30;
  - the three prefill sizes are `handed off`;
  - `beats_compiled` is true.
- **The skill, version 9.**
  - The ready check makes a passing `muse-all` a set.
  - *Ready to test* names a set's attempts together.
  - Builders and boundary agents record `min_rows` and the sizes it hands off.
- **G3 and the ready check are unchanged.** They time whole requests, so a slot's kernels count wherever they run. With 1 request every decode step has 1 row, so an attempt that hands off 1 row runs MLX's own code for the whole 1-request workload. Alone it is then no faster than its compiled bar there, and it can pass G3 only in a set whose other kernels are faster with 1 request. The brief says so.

## Consequences

- **Round 3's pair qualifies for round 4.** Under the set rule, routing attempt-6 with combine attempt-2 would have been `ready` in round 3, because its `muse-all` passed the swarm's ready check. Claude's G3 failed the same pair against the compiled bar at 32 requests of 1,024 tokens, on overlapping ranges, so the pair is still no baseline.
- **A set is judged once.** Its parts are not judged alone, so a set can carry a part that only helps together with the others. G3 still requires the whole set to be faster than both bars, and to be slower nowhere.
- **The interaction of all three slots is untested.** In rounds 1 and 2, replacing all three slots at once lost about 2.8 ms with 8 requests, for an unknown reason ([round 2](../findings/muse-round-2.md), section 4). A set with an experts attempt meets that again.

## Considered options

- **A paired test on medians for the ready check and G3,** so that one outlier repeat can't sink a candidate, as it sank combine attempt-2 in round 3. Not chosen for round 4: it changes ADR 0011's rule for Claude's G3 too, and there was too little time to check it before the deadline.
- **Hand-off by an explicit list of row counts.** More flexible, but no attempt has needed more than one range, and two bounds are simple to check.
