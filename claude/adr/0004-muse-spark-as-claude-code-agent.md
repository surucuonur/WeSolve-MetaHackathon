---
status: accepted
---

# Muse Spark works as a Claude Code agent session, kept apart by place, identity, hooks and audit

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

*[ADR 0005](0005-meta-settings-in-muse-folder-contributor-tier.md) changes three parts of this decision: Meta's settings live in the Muse Spark folder, the model is the contributor tier from the third session on, and there is no sandbox.*

**The change.** Muse Spark now does its kernel work as a coding agent of its own: a Claude Code session whose model is `muse-spark-1.3`, served by Meta's Anthropic-compatible API ([Meta's coding-agents guide](https://dev.meta.ai/docs/coding-agents)). surucuonur starts it with [tools/muse-agent/start.sh](../../tools/muse-agent/start.sh). It works from [muse-agent-brief.md](../muse-agent-brief.md): it reads the findings, chooses its targets, writes kernels, runs the grader itself and commits each attempt. No Claude-written loop stands between it and the task.

This replaces [ADR 0003](0003-muse-spark-writes-kernels.md)'s API pipeline:
- **Removed from the tree:** `bench/muse/run.py`, `client.py` and `verify.py`, and that stage's output (2 calls, a strategy and one routing+combine kernel).
- **Where it remains:** git history keeps it, in commits `d14c4a3` through `2c71eb4` (see [PROVENANCE.md](../../PROVENANCE.md)).
- **Why redo it:** so all of Muse Spark's results come from one method.

## How the two authors are kept apart

1. **Place.**
   - The agent works in its own git worktree on the branch `muse-agent`, created from the tag `agent-era`, which points at the commit registering its brief.
   - It writes only inside `muse-spark/`. Nothing else goes there.
   - Its Claude Code configuration lives in its own folder (`~/.claude-muse`), so it sees none of Claude's history, memory or settings.
   - Its worktree has no `.env`.
2. **Identity.**
   - Every commit is authored and committed by surucuonur.
   - The agent's settings ([tools/muse-agent/settings.json](../../tools/muse-agent/settings.json)) replace Claude Code's default trailer with `Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>`. For a non-Claude model, the default would be `Co-Authored-By: Claude Code <noreply@anthropic.com>`, which GitHub credits to its `claude` account.
   - The `.invalid` address is reserved and belongs to no one.
3. **Hooks** ([tools/githooks/](../../tools/githooks/)) refuse:
   - a commit on `muse-agent` without the Muse Spark co-author, or with Claude;
   - the Muse Spark co-author on any other branch;
   - a `muse-agent` commit outside `muse-spark/`;
   - a commit elsewhere that touches `muse-spark/`, unless it merges `muse-agent`;
   - any staged API key value.
4. **Audit.** [tools/provenance_check.py](../../tools/provenance_check.py) re-checks the whole history against the same rules. It also checks that every transcript turn came from a Meta model and that no key value is tracked. This catches anything committed around the hooks.
5. **Evidence.** [tools/export_transcript.py](../../tools/export_transcript.py) saves the agent session's conversation in `muse-spark/transcripts/`. Each assistant turn keeps its model, message id and request id. Claude Code's own system prompt and tool definitions are dropped, because they are Anthropic's, not ours to publish.
6. **Independent grading.** The grader ([bench/grader/](../../bench/grader/)) is Claude's work and was fixed before the session. The agent may run it but not change it. After `muse-agent` is merged with `--no-ff`, the grader and the decode-step timing run again from the main checkout, and Claude reports the results.

## Consequences

- **What judges can see:**
  - Muse Spark's whole engineering process: its choices, attempts, grader verdicts and commits;
  - on GitHub, the commits co-authored by "Muse Spark";
  - the transcript, which shows a Meta model wrote every turn.
- **The model and its price.** The default model is the standard tier, `muse-spark-1.3`: $1.25 per million input tokens and $4.25 per million output tokens. From the third session on, the contributor tier ([ADR 0005](0005-meta-settings-in-muse-folder-contributor-tier.md)).
  - On the contributor tier, Meta may train on everything sent. That would include Claude Code's own system prompt and tools, which sits uncomfortably with Anthropic's terms against using its services to train AI models.
  - The model is set in the Muse Spark folder's settings ([ADR 0005](0005-meta-settings-in-muse-folder-contributor-tier.md)); the choice shows in the transcript.
  - Claude Code may show $0 spend for a third-party model, so the real cost is on Meta's dashboard.
- **Anthropic doesn't support it.** Its docs say it "doesn't support routing Claude Code to non-Claude models", though no clause forbids it. Meta documents this setup.
- **Remaining risk.** An agent with a shell can reach what the user's account can.
  - The read and write blocks outside the worktree and the deny rules narrow that. (The sandbox first used here was removed: [ADR 0005](0005-meta-settings-in-muse-folder-contributor-tier.md).)
  - The hooks and the audit stop anything from being *committed* wrongly, but they can't undo what the agent sees.
  - The API key is in the session's environment, because Claude Code needs it. Every commit and transcript is scanned for it.

## Considered options

- **Keep the API pipeline** (ADR 0003). It's safer, because Muse Spark only returns text, but a Claude-written loop sits between Muse Spark and its work.
- **Muse Spark writes its own grader.** That's a student marking their own exam. Rejected.
- **Meta's own Muse Code agent.** Equally possible, but nothing in it controls commit attribution the way Claude Code's `attribution` setting does.
