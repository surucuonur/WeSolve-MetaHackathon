---
status: accepted
---

# Muse Spark runs only through its launcher, on the contributor tier, without a sandbox

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** Three parts of [ADR 0004](0004-muse-spark-as-claude-code-agent.md) change; the rest stands.

1. **Meta's settings exist only in the session the launcher starts.**
   - [start.sh](../../tools/muse-agent/start.sh) takes the API key (`MODEL_API_KEY`) from `legal-agent-workshop/agent/.env`. That is the file surucuonur set up for Meta's workshop; git ignores it.
   - It sets `ANTHROPIC_BASE_URL=https://api.meta.ai`, the key and every model alias only in the environment of the Claude Code process it starts.
   - No Claude Code configuration file points at Meta, Claude's own included.
2. **The contributor tier**, `muse-spark-1.3-contributor`, is the default from the third session on (`MUSE_MODEL` can change it). The first two sessions (`ec2d3d4f…` and `ef1b17e0…`, 2026-11-02) ran on the standard `muse-spark-1.3`.
3. **No sandbox.** The session's settings no longer turn on Claude Code's sandbox.

## Why

- **One way in.** A Claude Code session runs on Muse Spark whenever its configuration points at Meta.
  - While a switch could point any configuration at Meta, any session could become a Muse Spark session. On 2026-11-02 one did: Claude's own session in the main checkout ([PROVENANCE.md](../../PROVENANCE.md)).
  - Now only start.sh puts Meta's settings anywhere, and it starts the session in the worktree, on `muse-agent`, with the Muse Spark trailer and its permission rules.
  - A Claude Code started any other way talks to Anthropic, not Meta.
- **The sandbox didn't hold.**
  - It blocked `.venv/bin/python` and the git hooks, which live in the main checkout. Muse Spark re-ran those commands outside the sandbox, and the allow rules let them through. In practice it covered only the commands that didn't matter.
  - An airtight sandbox would need exceptions for the main checkout's `.venv`, `models` and `tools`, and possibly for the GPU. surucuonur chose to rely on the other layers instead.
- **Price.** The contributor tier costs about a twelfth of the standard tier ([ADR 0003](0003-muse-spark-writes-kernels.md)).

## What still keeps the work apart

- **Place:**
  - the worktree on `muse-agent` and the Muse Spark config folder `~/.claude-muse`;
  - edits are allowed only in `muse-spark/`, denied for Claude's folders, and blocked outside the worktree;
  - reads of both `.env` files are denied.
- **Identity:** the Muse Spark trailer. `commit-msg` refuses a commit on `muse-agent` without it.
- **Hooks:** `pre-commit` refuses paths outside `muse-spark/` and any staged API key from either `.env` file.
- **Audit and transcripts:** as in ADR 0004. [export_transcript.py](../../tools/export_transcript.py) now exports every session in the folder.

## Consequences

- **Meta may train on everything sent.** That includes Claude Code's own system prompt and tool definitions, the brief, the findings and every file Muse Spark reads.
  - As ADR 0004 noted, sending Claude Code's prompt for training sits uncomfortably with Anthropic's terms against using its services to train AI models.
  - surucuonur accepted this for the lower price. What has been sent can't be taken back.
- **The transcripts show the tier.** Each assistant turn records its model. The audit accepts any model that isn't a Claude model.
- **Remaining risk.**
  - Without the sandbox, shell commands outside the allow list run only when surucuonur approves them, and then they can reach what the account can.
  - The API key is in the session's environment, because Claude Code needs it. Every commit and exported transcript is scanned for it.

## Considered options

- **Meta's settings in the Muse Spark folder's `settings.json`.** Briefly adopted on 2026-11-02 (commit `433fea5`) and replaced the same day. It duplicated the key, and a session started by hand with that folder would have run on Muse Spark without the launcher's settings.
- **An airtight sandbox.** No escape from it, plus read access to the main checkout's `.venv`, `models` and `tools`. Rejected as more machinery than the risk warrants.
- **A guard hook in the main checkout** that blocks every tool call when a session points at Meta. Rejected: with Meta's settings only in the launcher, no other session points at Meta.
