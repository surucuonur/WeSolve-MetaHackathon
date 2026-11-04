---
status: accepted
---

# Rounds build on each other: each in its own folder, finished rounds frozen, the best attempt so far is the bar

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** On 2026-11-03, before round 2, surucuonur asked that round 1's files stay the same, and that round 2 "can see the round 1 and its results" and "do different plan considering that". Under the round skill's version 3, round 2 would have done neither:
- it would have continued round 1's documents in place, with `## Round 2` sections in its entries, and its README versions, logs and summary carried on;
- it would have judged attempts only against the original MLX code and its compiled version, a bar round 1 already passed.

Skill version 4 ([.agents/skills/muse-round/](../../.agents/skills/muse-round/SKILL.md)) changes both.

## How it works

- **A folder per round.** Round 1 stays at the top of `muse-spark/`. Round N writes in `muse-spark/round-<N>/`: `exploration/`, `implementation/`, `kernels/` and `summary.md`. Transcripts stay in `muse-spark/transcripts/`, one file per session.
- **Finished rounds are frozen.** Claude tags a round's last commit `muse-round-<N>` once its transcripts are committed. Round 1's tag is `muse-round-1`, on `60308aa`, and freezes its 71 files. From a tag on:
  - the pre-commit hook on `muse-agent` refuses any change to, or deletion of, a file the tag holds in `muse-spark/`. It also refuses a new file anywhere but the current round's folder and `muse-spark/transcripts/`;
  - the audit, `tools/provenance_check.py`, re-checks every commit after each tag, so a commit made around the hook is caught;
  - the transcript exporter skips sessions it has already exported, so running it again never rewrites a frozen transcript.
- **The starting point.** From round 2 on, the lead first writes a plan from:
  - the previous rounds' final documents;
  - the notes of each slot's best attempt;
  - Claude's report on the previous round, `claude/findings/muse-round-<N-1>.md`.

  The plan covers, per slot, the best attempt so far (the **baseline**) and its G3 effect against its prediction. It also covers what is left to build, and which territories to explore again. The round stops if Claude's report is missing.
- **Targeted exploration.** Explorers map:
  - the baselines' kernels, now the code in the slots;
  - where Claude's G3 differed from the piece timing;
  - only those round-1 territories the plan reopens.

  Earlier entries are cited, never rewritten.
- **The baseline is the bar.** Claude's report names each slot's baseline. A new attempt must beat it in the same grader run, with `bench/grader/check_candidate.py --baseline`.
  - **The test.** The two are timed in alternating repeats, so drift in the GPU's clocks reaches both alike. The candidate must be faster in at least 22 of the 30 paired repeats at each row count: a one-sided sign test at p < 0.01.
  - **Why a sign test.** In a test on 2026-11-03, round 1's combine attempt, timed separately against itself, differed by up to 5% between medians. In paired repeats it won only 15 to 19 of 30 against itself. Round 1's final experts attempt won 30 of 30 against its first. Comparing medians would have passed on noise.
- **A slot that nothing beats keeps its baseline** for G3.

## Consequences

- Round 1's files can't change, and every path in round 1's documents stays valid.
- Attempt numbers restart in each round's folder, and each round's summary counts attempts per round and over all rounds.
- The lead reads more at the start of a round, and the explorers repeat less.
- The grader's verdict gains `baseline`, `beats_baseline`, `wins_needed` and, per row count, `wins_vs_baseline`. Without `--baseline` it is unchanged.
- Claude's G3 for round 2 compares round 2's attempts with round 1's.

## Considered options

- **Carry on in round 1's folders** with `## Round N` sections, as version 3 did. Round 1's files would change, and only git history would keep their round-1 state.
- **Freeze by the skill's word alone,** with no hook. A slip would change round 1's files, unnoticed until the next audit.
- **Compare against round 1's recorded timings.** Timings drift between runs: in round 1 the compiled reference varied by up to 2 times at 1 row. The comparison has to happen in one run.
- **Compare medians within one run.** The 5% noise measured above would let an equal kernel pass.
