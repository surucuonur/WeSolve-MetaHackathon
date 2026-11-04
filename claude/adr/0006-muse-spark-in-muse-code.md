---
status: accepted
---

# Muse Spark works in Muse Code, Meta's own coding agent, with helper agents and no attempt limit

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

*[ADR 0007](0007-muse-spark-swarm.md) replaces this record's round design on 2026-11-03, before round 1: a swarm of explorers and builders instead of short-lived helpers. That covers the helpers, the drafts, `diagnosis.md`, `strategy.md` and grading by the lead alone. Muse Code, the launcher, the isolation and the lack of an attempt limit stand.*

**The change.** From 2026-11-02, Muse Spark does its kernel work in **Muse Code** ([Meta's docs](https://dev.meta.ai/docs/muse-code)), not in a Claude Code session pointed at Meta. This replaces [ADR 0004](0004-muse-spark-as-claude-code-agent.md) and [ADR 0005](0005-muse-spark-launcher-only-contributor-tier.md) wherever they describe the harness.
- surucuonur starts it with [tools/muse-agent/start.sh](../../tools/muse-agent/start.sh), still the only way in.
- Every round follows the skill [.agents/skills/muse-round/SKILL.md](../../.agents/skills/muse-round/SKILL.md). Claude wrote the skill, and it is committed in the repository so anyone can read it. In each round:
  1. Muse's helper agents investigate the run evidence and the code, each with the surrounding system in view.
  2. Muse writes a diagnosis and a strategy.
  3. Helpers draft and review kernels, and the lead grades and commits them.

## Why

- **Only Meta's tools on the Muse side.** The hackathon scores "Use of Meta AI tooling", and its FAQ names Muse Code.
- **Claude Code's own prompt stops going to Meta.** With Claude Code as the harness, Anthropic's system prompt and tool definitions went to Meta with every request. On the contributor tier, Meta may train on them, which sits uncomfortably with Anthropic's terms (ADR 0004).
- **Helper agents.** surucuonur wants the investigation and the kernel work spread over several agents. Muse Code has native subagents: up to 8 at once by default, the lead included. Its scripted workflows are missing from the Apple Silicon build.

## What changes in the brief

The [brief](../muse-agent-brief.md) stays fixed as committed, and so do its task, slots, tools, files, grader and the pass criteria G1 and G2. Three things change:

1. **The harness.** The brief's "You are a Claude Code session" and "the co-author trailer your settings add" now read: a Muse Code session that writes the trailer itself. Muse Code has no setting that adds it, so the skill gives the exact commit command, and `commit-msg` refuses a `muse-agent` commit without it.
2. **No attempt limit and no stop rule.** In surucuonur's words: "if we limit the model, it should be limitless inside the muse code, it should stop with muse codes stopping conditions."
   - Muse works until it judges a slot done, or until Muse Code stops. No step cap is set, and surucuonur can interrupt at any time.
   - **What keeps it honest:** every graded attempt is committed, in order. The summary gives the number of graded attempts per slot, so readers see how many tries the best result came from.
3. **Drafts and a diagnosis.** Helpers write candidate kernels in `muse-spark/drafts/`. These are never graded on their own and are committed with the attempt they fed. `muse-spark/diagnosis.md` names the root cause per slot before any kernel is written.

## How the work stays apart

- **Starting over.** The `muse-agent` branch from the Claude Code harness was deleted on surucuonur's decision: one strategy commit and one ungraded attempt, "didnt start properly and didnt do much work". The new `muse-agent` starts from the tag `muse-code-era`, so every Muse result comes from Muse Code.
- **Place.**
  - Its own worktree, `../Hackathon-muse`, on `muse-agent`; writes only in `muse-spark/`, enforced by `pre-commit`.
  - The launcher trusts the worktree for each run, which loads the project skill and allows helpers.
  - It gives `muse` a home folder of its own, `~/.muse-spark`, which holds its settings and session logs. Muse Code looks for personal skills and rules under the home folder (`~/.agents/skills`, `~/.claude/skills`, `~/.claude/CLAUDE.md`) and loads them by default; in its own home it finds none, so the model sees only Muse Code's bundled skills and the project's skill.
- **Key.** `MODEL_API_KEY` from `legal-agent-workshop/agent/.env` goes only to the `muse` process, as `META_API_KEY`, and is never stored.
- **Seeing Claude's newest work.** `muse-agent` can't take merges. After a round, its branch is merged into main with `--no-ff` and Claude commits its results as `claude/findings/muse-round-<N>.md`. The next launch then fast-forwards `muse-agent` to main, which creates no commit and rewrites nothing, so the next round reads those results.
- **One GPU job at a time.** The grader takes an exclusive lock ([bench/grader/gpu_lock.py](../../bench/grader/gpu_lock.py)), so a second run, from any agent or from Claude, stops with a message instead of distorting the timings. Only the lead grades.
- **Evidence.** Muse Code logs every model call, tool run and edit for the lead, each helper and its own reminder agents, one log per agent, and `muse export` writes a log as JSON. [tools/export_transcript.py](../../tools/export_transcript.py) joins a session's logs into one transcript in `muse-spark/transcripts/`.
  - **What it keeps:** messages, tool calls and results, approvals, and the model of each call.
  - **What it drops:** Muse Code's own instructions.
  - It refuses a session that contains a key.
  - The audit checks that a Meta model (`muse-*`) made every model call.
- **Unchanged:** the authorship, the trailers, the hooks, the audit, and Claude's independent re-grading and G3 after each merge.

## Consequences

- **The model:** `muse-spark-1.3-contributor`, as in ADR 0005, for round 1; from round 2, Muse Wonder Full through a Muse Code subscription ([ADR 0008](0008-muse-wonder-full.md)). Meta may train on everything sent: the brief, the skill, the findings and every file Muse reads, all written by Claude or by us. Claude Code's own prompt is no longer among them. surucuonur accepted this for the lower price.
- **No sandbox.** Muse Code's sandbox keeps `.git` read-only, and a worktree commits into the main checkout's `.git`, so the launcher passes `--disable-sandbox`. Tool approvals stay on.
  - **Remaining risk:** an approved shell command can reach what the account can, and the key is in the session's environment. Every commit and exported transcript is scanned for it.
- **No fallback.** If Muse Code can't run the round, the fix is in the Muse Code setup. surucuonur ruled out returning to the Claude Code harness.
- **Superset's notification hooks,** set in surucuonur's own Muse Code settings, don't run in Muse Spark's sessions, which use their own home and settings.
- **The first smoke test leaked personal context.** It ran with Muse Code's settings for leaving personal skills and rules out, which turned out not to work. surucuonur's global instructions and the names and descriptions of 43 personal skills went to Meta. See [PROVENANCE.md](../../PROVENANCE.md).

## Considered options

- **Keep the Claude Code harness (ADR 0004/0005).** It already worked, but Anthropic's prompt goes to Meta, and the Muse side isn't Meta's tooling.
- **Keep the old branch as an archive,** renamed and pushed. Rejected: its work was too slight to keep beside a fresh start.
- **Keep the brief's 5 attempts per slot and its stop rule.** Rejected by surucuonur; the full commit history and the attempt counts replace the limit as the safeguard.
- **Let helpers grade too.** The GPU times one run at a time, so parallel grading adds coordination and no speed.
- **Muse Code's own settings for personal skills and rules,** `context.foreign_personal_skills` and `context.foreign_personal_rules`. Briefly adopted, then replaced after the first smoke test on 2026-11-03.
  - Muse Code 1.4.14 accepts them with `-c`, but personal skills and rules still load.
  - They name only other agents' folders. `~/.agents/skills` counts as Muse Code's own and would load anyway.
  - The flag `--no-foreign-personal-context` does drop `~/.claude`, but only for headless runs, and it also leaves `~/.agents/skills`.
