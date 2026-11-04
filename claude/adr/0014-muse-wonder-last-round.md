---
status: accepted
---

# The last round, round 4, runs Muse Wonder on the API key, as round 2 did

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** On 2026-11-04, after Claude measured round 3, surucuonur asked for Muse Wonder Full "for the last round", then changed it: "Dont use full version, use muse wonder as you have done in round 2". `tools/muse-agent/start.sh` now starts sessions with these settings:
- **The model** `muse-wonder-1.0`, Muse Wonder, on the Meta Model API with the API key (`MUSE_AUTH=key`), as in round 2 ([ADR 0008](0008-muse-wonder.md)). Round 3 ran Muse Spark's contributor model ([ADR 0012](0012-muse-spark-contributor-unattended.md)).
- **Unchanged:** effort `max`, approval mode `never` and the sandbox off (ADR 0012), the swarm and its skill.

Round 4 is the last round before the hackathon's deadline.

## How it works

- **The launcher.** `MODEL` defaults to `muse-wonder-1.0`; `AUTH` stays `key`. `MUSE_MODEL=muse-spark-1.3-contributor` brings back round 3's model, and `MUSE_MODEL=muse-wonder-full-1.0 MUSE_AUTH=account` Muse Wonder Full, which needs a subscription login.
- **Names and co-author lines** come from `tools/provenance.py`: "Muse Wonder" and `Co-Authored-By: Muse Wonder (Meta Model API) <muse-wonder@meta-model-api.invalid>`, as in round 2. The commit-message hook and `tools/provenance_check.py` accept them.

## Consequences

- **Price.** Muse Wonder costs $4.00 per million input tokens, $0.20 cached and $20.00 output ([ADR 0008](0008-muse-wonder.md)). Round 2 cost about $58, round 3 on the contributor model about $1.
- **Capability.** Muse Wonder is the model ADR 0008 chose as Meta's most capable on the API. Round 4 runs it unattended, which round 2 did not.
- **Earlier rounds are unchanged.** Their commits, co-authors and transcripts keep the models they ran.

## Considered options

- **Muse Wonder Full on a Muse Code subscription.** surucuonur's first choice for this round, and for round 2. Not chosen: it needs a login, and surucuonur asked for the setup round 2 ran.
- **Keep the contributor model (ADR 0012).** Cheaper, and its round 3 ran to the end without help. Not chosen: surucuonur asked for Muse Wonder for the last round.
