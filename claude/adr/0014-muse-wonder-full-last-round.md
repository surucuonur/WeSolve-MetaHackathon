---
status: accepted
---

# The last round, round 4, runs Muse Wonder Full on a Muse Code subscription

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** On 2026-11-04, after Claude measured round 3, surucuonur decided: "I want you to switch to the muse wonder full, and do the next round. for the last round". `tools/muse-agent/start.sh` now starts sessions with these settings:
- **The model** `muse-wonder-full-1.0`, Muse Wonder Full, which runs only on a Muse Code subscription. Round 3 ran Muse Spark's contributor model ([ADR 0012](0012-muse-spark-contributor-unattended.md)), and round 2 ran Muse Wonder on the API key ([ADR 0008](0008-muse-wonder.md)).
- **Sign-in `account`** (`MUSE_AUTH`), with contributor mode on. The launcher unsets `META_API_KEY`, because an API key always takes priority over the account login. surucuonur signs in once with `tools/muse-agent/start.sh login`.
- **Unchanged:** effort `max`, approval mode `never` and the sandbox off (ADR 0012), the swarm and its skill.

Round 4 is the last round before the hackathon's deadline.

## How it works

- **The launcher.** `MODEL` defaults to `muse-wonder-full-1.0`, and `AUTH` to `account`. `MUSE_MODEL=muse-spark-1.3-contributor MUSE_AUTH=key` brings back round 3's settings.
- **Names and co-author lines** come from `tools/provenance.py`. It already maps `muse-wonder-full-` to "Muse Wonder Full" and `Co-Authored-By: Muse Wonder Full (Muse Code) <muse-wonder-full@muse-code.invalid>`, and the longest prefix wins, so the model is never read as plain Muse Wonder. The commit-message hook and `tools/provenance_check.py` accept it.
- **The login** lives in the launcher's own home folder, `~/.muse-spark`, like the session logs. Claude never reads it.

## Consequences

- **Price.** A subscription pays for the session instead of the per-token API: round 2 on Muse Wonder's API cost about $58, round 3 on the contributor model about $1. `muse usage` shows the subscription's allowance.
- **Capability.** Muse Wonder Full is the model surucuonur first chose for round 2 ([ADR 0008](0008-muse-wonder.md)), before choosing the API instead to avoid a login.
- **Earlier rounds are unchanged.** Their commits, co-authors and transcripts keep the models they ran.

## Considered options

- **Keep the contributor model (ADR 0012).** Cheaper, and its round 3 ran to the end without help. Not chosen: surucuonur asked for Muse Wonder Full for the last round.
- **Muse Wonder on the API key (ADR 0008).** Needs no login, but costs about $58 a round, and surucuonur asked for Wonder Full.
