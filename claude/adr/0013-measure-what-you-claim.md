---
status: accepted
---

# From round 3, the swarm measures what it claims: an in-round ready check, a checker, and a rule to change approach

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. The design is the Muse agent's; see below. [Who made what](../../PROVENANCE.md).*

**The change.** Two rounds produced no real improvement, while the swarm's own reports predicted gains ([round 1](../findings/muse-round-1.md), [round 2](../findings/muse-round-2.md)). Claude traced this to five behaviours:
1. treating what wasn't measured as free;
2. claiming end-to-end gains from per-slot timing;
3. reasoning inconsistently, such as summing gains it had said don't add up;
4. using the grader as a search loop (round 1);
5. keeping a design that kept failing at the same point.

The root: inside a round the swarm never saw the real objective, only per-slot piece timing, and no one checked its claims before its summary. The rules even required an end-to-end number while forbidding a measurement of it: `tracking.md` said "Load no model", and *Ready to test* asked for the expected G3 change.

On 2026-11-04 surucuonur asked how to fix this. surucuonur decided:
- the swarm may run the end-to-end check itself, through the GPU queue;
- the design work is done by Meta's Muse: "Meta muse should be used due to the rules of the hackathon";
- the lead runs the check;
- my post-round G3 must agree with the swarm's in-round verdicts, verdict for verdict;
- the token count is recorded in every run.

## Who designed it

One headless Muse Code session, `01a1a701-10d9-7552-a2c1-bb9149781051`, on `muse-spark-1.3-contributor`, with approvals `on-request` and no cost or time limit. Its lead started 11 helpers:
- **four explorers:** the skill, the measurement tools, the round records, and Claude's findings;
- **three designers,** each starting from a different perspective: the existing system, the contracts, and failure modes;
- **three testers,** each testing one design live on round 2's attempts;
- **a blind reviewer;**
- **a tester of the synthesis.**

The reviewer passed one design on all six gates. It was the only design whose audit recomputes the G3 verdict from the log; the other two accepted a hand-edited verdict file, which their testers demonstrated. The lead added five ideas from the other two designs. Prototypes and test results stayed in the Muse worktree's untracked `design-lab/`, and the session's transcript records them.

Claude implemented the design on main without copying the Muse agent's code, since its work is committed only on `muse-agent` ([authorship](../../PROVENANCE.md)). The tools are new code following the design; the skill text is rewritten from it.

## How it works

- **The ready check, `bench/grader/check_ready.py`.**
  - **What it runs:** `time_candidates.py vllm` with G3's protocol fixed in code: the four workloads, 5 repeats and 256 tokens, with every submitted slot in one call, so a combination is measured as `muse-all`. No flag changes the protocol, and it stops before any GPU work if `time_candidates.py`'s G3 constants change.
  - **Its audit:** it writes a marker into the log, then audits the log instead of trusting it:
    - one marker with the protocol;
    - every run vLLM, at a G3 workload, with 256 tokens (a field every run now records, `repeat_fixes.py`) and all three measurements;
    - repeats 0–4 for each variant and workload, and exactly the expected variants;
    - the same greedy tokens as the unchanged model;
    - and `g3-vllm.json` equal to the verdict recomputed from the log.
  - **Its output:** `ready-verdict.json`, with medians and ranges.
  - **Exit codes:** 0 passes; 1 fails, and the verdict stands; 2 means no verdict, so the check runs again in a new folder.
  - **Re-checking:** `--audit` re-checks a folder without the GPU.
  - **Testing:** `bench/grader/test_check_ready.py` tries 13 ways of gaming it on a real check, plus the unchanged control.
- **Skill version 8:**
  - The goal is whole requests.
  - A hypothesis closes as a `candidate` on the grader's gates and its boundary check. Once candidates settle, the lead runs one ready check over them all, at most two per round, and only a passing verdict makes an attempt `ready`.
  - Builders keep `grade.json` in the attempt folder.
  - The number rule separates measured piece numbers, measured whole-request numbers and labelled predictions. Combinations are measured, never summed, and what no one measured is listed as **Unverified**.
  - A new **checker** checks every speed claim against its file, before a hypothesis closes and over the final claims.
  - After three attempts fail at the same phase for the same cause, the next changes approach, or the lead logs why not. A renamed cause counts as the same one.
  - Claude's G3 after the round stays the authority.
- **Checked on 2026-11-04,** on round 2's submission (routing attempt-4 and combine attempt-1):
  - the 14 cases (13 tricks and the control) on the Muse session's live log adapted to this tool's format, all as designed;
  - the live run itself ([results/runs/2026-11-04-ready-check/](../../results/runs/2026-11-04-ready-check/), 17:06–17:28 UTC): exit 1, with no protocol problem. Routing fails on prompt time, combine on the decode step, and the pair on all three, as in Claude's measurement of round 2; the tokens are identical. `--audit` agrees, and the 14 test cases pass on it too (`test-check-ready.txt`).
- **One rule recorded here for the first time:** one grader run is one attempt. Skill version 4 introduced it after round 1 ran the grader 28 times on 4 experts attempts, and no ADR recorded it.

## Consequences

- **A round takes longer.** One check runs about 10–15 minutes per slot, and about 21 minutes for two slots, during which builders' grader runs wait.
- **Fewer, true claims.** A candidate that loses on whole requests is no longer submitted. A real but small gain that noise hides at 1 request may be lost too, because a verdict stands.
- **Not covered:**
  - prefill chunks of other sizes and decode steps of other row counts;
  - a log invented whole, which the audit can't detect, though Claude's independent G3 can;
  - whether the contributor model follows the checker card, which only round 3 will show.
- **Old logs aren't evidence.** Logs from before this change have no marker or token count, and the audit refuses them.

## Considered options

- **A cheaper in-round check in plain MLX.** Its prompt time is always 0.0, so it would make "unmeasured is free" the official check.
- **Prediction rules alone, without an in-round measurement.** Round 2's records already had predictions next to misses of up to 60 times, and nothing changed.
- **A check per hypothesis during the round.** It gives earlier evidence, but costs two to three times as much GPU time and blocks builders.
- **A builder or a new runner role running the check.** surucuonur chose the lead.
- **Trusting `g3-vllm.json`.** Two of the three designs did, and their testers showed that a hand-edited file passed.
