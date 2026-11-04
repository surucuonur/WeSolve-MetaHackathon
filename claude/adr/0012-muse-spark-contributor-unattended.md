---
status: accepted
---

# From round 3, the Muse agent runs Muse Spark's contributor model, with tool approvals off

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** On 2026-11-04, after round 2, surucuonur decided: "from now on use muse-spark-contribution models, also make them fully automated (auto-accept) by default". From then on, `tools/muse-agent/start.sh` starts every session, round 3 included, with these settings:
- **The model** `muse-spark-1.3-contributor`, Muse Spark's contributor model, which round 1 ran. It still signs in with the API key (`MUSE_AUTH=key`), and the effort is still `max`.
- **Approval mode `never`**, Muse Code's documented mode in which "nothing stops for approval" ([permissions](https://dev.meta.ai/docs/muse-code/permissions)). Until now approvals were on (`on-request`): a person approved or declined, in the terminal, each tool call Muse Code judged risky. Now a session runs unattended.

This replaces [ADR 0008](0008-muse-wonder.md)'s choice of Muse Wonder. `MUSE_MODEL=muse-wonder-1.0` still selects it, and `MUSE_APPROVAL=on-request` or `untrusted` brings the approval prompts back.

## How it works

- **The launcher:**
  - `MODEL` defaults to `muse-spark-1.3-contributor`.
  - The new `APPROVAL` defaults to `never` and is passed as `--approval-mode`.
  - The banner prints both.
  - The sandbox stays off, as it has been since [ADR 0005](0005-muse-spark-launcher-only-contributor-tier.md).
- **Names and co-author lines** come from `tools/provenance.py`, which maps `muse-spark-` to "Muse Spark" and `Co-Authored-By: Muse Spark (Meta Model API) <…>`, as in round 1. The commit-message hook and `tools/provenance_check.py` already accept them.
- **Checked on 2026-11-04** in a headless session (`01a1a6ee-ca88-7281-83f4-e927246569a0`).
  - The launcher started `muse-spark-1.3-contributor`, and `muse exec` accepted `--approval-mode never` with `--reasoning-effort max`.
  - The session replied as asked, using 24,599 input and 15 output tokens.
  - The worktree stayed clean.
  - It made no tool call, so the approval mode's effect on one is not yet observed in a log.

## Consequences

- **Price.** The contributor model costs $0.002 per million input tokens, $0.10 cached and $0.20 output, against Muse Wonder's $4.00, $0.20 and $20.00 ([ADR 0008](0008-muse-wonder.md)). Round 1 cost about $0.70 and round 2 about $58.
- **Capability.** ADR 0008 chose Muse Wonder as Meta's most capable model. Round 3 runs the model round 1 ran, so the skill's instructions must be explicit and checkable, since a less capable model follows general advice less reliably.
- **No one watches the tool calls.** With both the sandbox and approvals off, a session can run any command on the Mac without stopping.
  - What still holds: the git hooks refuse a commit outside `muse-spark/`, a change to a finished round, a merge and a wrong co-author line; `tools/provenance_check.py` audits every commit; the skill's write rules.
  - surucuonur chose no sandbox before ([ADR 0005](0005-muse-spark-launcher-only-contributor-tier.md)).
- **Claude's own tooling may decline to start such a session.** On 2026-11-04 Claude Code's safety check let the first smoke test run, but blocked a second one that would make a tool call, calling it an unsafe agent. Sessions with approvals off are started by surucuonur, or by Claude only where its tooling allows it.
- **Rounds 1 and 2 are unchanged.** Their commits, co-authors and transcripts keep the models they ran.

## Considered options

- **Keep Muse Wonder (ADR 0008).** The more capable model, at about 60 times round 1's price per round. Not chosen: surucuonur asked for the contributor model from now on.
- **Keep approvals on.** A person approves risky tool calls, but a round then needs someone at the terminal for its whole length. Not chosen: surucuonur asked for fully automated sessions.
- **`--disable-approval` or `--yolo`.**
  - `--disable-approval` skips the prompts but keeps the sandbox on. The sandbox keeps `.git` read-only and so stops a worktree from committing, which is why ADR 0005 turned it off.
  - `--yolo` also disables both. The approval mode is the setting Muse Code documents for this, and `MUSE_APPROVAL` can turn it back.
