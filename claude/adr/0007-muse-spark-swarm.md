---
status: accepted
---

# Muse Spark's round is a swarm: explorers share one analysis, then competing builders grade their own kernels

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** From 2026-11-03, before round 1, the round skill ([.agents/skills/muse-round/](../../.agents/skills/muse-round/SKILL.md)) runs a **swarm**. This replaces the lead with short-lived helpers that [ADR 0006](0006-muse-spark-in-muse-code.md) described. The round has two phases, in surucuonur's words:

- **Exploration.** "Point individual agents at individual codebases (modules within codebases, or even individual files if its too important) and measurement documentations."
  - Together they write one analysis that all of them can read and change, "so that they can change direction for more focusing".
  - Its final version holds "the context, analysis, relations, possible inefficiency and everything … for future improvements".
- **Implementation.** From the analysis, build "the best solution that can be implemented in here, by using multiple agents to build it".
  - It must be "aware of the full system" and must not "affect the previous or after steps".
  - "The best version of the solution should be ready to test and go."

surucuonur also asked that both phases be "trackable by the documents … so that I can track their findings".

## How the swarm works

- **One Muse Code session.**
  - The lead starts the helpers, keeps the phase documents current, settles clashes, and is the only agent that commits.
  - It neither explores nor grades.
  - The launcher raises Muse Code's limit from 8 agents at once to 12.
- **Exploration.**
  - One explorer per **territory**: a module, a single important file, or a measurement document.
  - There are 8 to start: the model, the expert path, MLX's quantized kernels, MLX's other slot kernels, vllm-metal's decode path, the grader's slot interface, Claude's findings, and the other runs.
  - The lead splits, merges or adds territories as costs show up.
  - A verifier marks each inefficiency confirmed or refuted against its sources.
  - Implementation starts once every territory is done and every inefficiency is settled.
- **Implementation.**
  - Confirmed inefficiencies are taken in order of predicted saving, at most two at a time and in different slots.
  - Per inefficiency, a boundary agent first writes what the solution must keep identical: the slot's inputs and outputs, dtypes, rounding points, and the steps before and after.
  - Then 2 or 3 builders each take a different direction, and each writes and grades its own attempt.
  - The best attempt is hardened one change at a time until it passes G1, G2 and the boundary check, and beats the compiled reference at 1, 8 and 32 rows.
  - No limit, as in ADR 0006: the lead ends the round when it judges further work won't pay, and says why.
- **Adapting.** Before every new task and after every post, an agent reads its phase's tracker and the newest log lines. It changes direction when:
  - another entry already covers its item;
  - its premise was refuted;
  - an unowned item predicts a larger saving.
  
  Savings are compared in µs per decode step at 1, 8 and 32 rows.

## The shared documents

`muse-spark/exploration/` and `muse-spark/implementation/` each hold:
- **`README.md`.** The phase's document, opening with a tracker table. Exploration's is the analysis; implementation's is the report, ending with the solution that is ready to test. Only the lead writes it, so concurrent edits never clobber it.
- **Entries,** one file per territory, inefficiency or hypothesis. Each has a status block its owner keeps current, and other agents append notes under `## From other agents` rather than rewriting.
- **`log.md`.** One line per event from every agent, appended through the shell: 2,000 concurrent appends from two writers, tested, came out whole.

The files are live in the worktree while the round runs (`tail -f …/log.md`), and the lead commits them at every checkpoint.

## Versions

surucuonur asked for version numbers that git commits control. Each version is raised by a single writer, so no number is taken twice:
- **Phase versions:**
  - each README opens with its version, vN, and the skill version the round runs;
  - the lead raises it at every checkpoint, and each version is exactly one commit, whose subject names it (`Muse Spark: exploration v5: …`);
  - the README ends with a changelog naming the entries each version changed.
- **Entry versions:**
  - each entry's status block has a version, which its owner raises with every change to its own text;
  - the verifier raises it with its verdict, and the lead raises it for hypothesis entries;
  - notes from other agents leave it alone.
- **Skill versions:**
  - the skill folder carries one number, raised by Claude with every change ([CHANGELOG.md](../../.agents/skills/muse-round/CHANGELOG.md)), and each round records the version it ran;
  - this design is version 2, and versioning arrived in version 3.
- **No tags for versions.** Tags stay for the era markers (`agent-era`, `muse-code-era`), and Muse Spark creates none.

## What changes in the brief, beyond ADR 0006

- **`strategy.md` and its targets JSON give way to hypothesis entries.** No code reads the JSON.
- **`diagnosis.md` gives way to the exploration folder.**
- **No drafts.** Every solution a builder finishes is graded and committed as an attempt, so the attempt count includes all of them.
- **Builders grade.** `bench/grader/gpu_lock.py` now waits for its turn instead of exiting when the GPU is busy. Runs queue and never overlap, so the timings stay clean. The grader is still the only GPU job, and helpers load no model.

The task, the slots, the grader, G1 and G2 stay.

## Consequences

- **More model calls:** about 10 agents in exploration, and up to 9 in implementation. Meta's terms for the contributor tier are unchanged (ADR 0006).
- **A GPU queue:** a builder may wait for others' grading. One run takes under 3 minutes.
- **Approvals:** all agents' shell commands surface in surucuonur's one terminal.
- **Lost edits:** two agents appending to one entry at the same moment can still lose a note. The lead re-reads at each checkpoint.
- **Only the lead commits,** because simultaneous commits from one checkout collide on git's index lock.

## Considered options

- **Perspectives by kind of cost** (bytes, launches, routing, occupancy, correctness, a skeptic). Proposed first; surucuonur chose territories of the code and measurements instead.
- **Separate Muse Code sessions with no lead.** True peers, but one terminal of approvals per session, split transcripts, and no one to settle clashes.
- **Only the lead grades.** No grader change, but the lead becomes the bottleneck.
- **One shared file, or one `analysis.md` written at the end.** One file lets simultaneous edits drop each other's changes, and an end-only document can't be followed live.
