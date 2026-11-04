---
status: accepted
---

# From round 2, the Muse agent is Muse Wonder, on the Meta Model API

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** From round 2 (decided 2026-11-03), the launcher runs **Muse Wonder** (`muse-wonder-1.0`) instead of Muse Spark (`muse-spark-1.3-contributor`, round 1). surucuonur asked for Meta's best model, wanting "to see improved results after this step". Meta calls Muse Wonder "Meta's most capable model" ([Meta's model docs](https://dev.meta.ai/docs/models)).

surucuonur first chose Muse Wonder Full, "the higher-capability variant", and the launcher was set up for it. Full "is available only in Muse Code with a subscription", so it needs a Muse Code sign-in. surucuonur then chose to run without a sign-in: "without login switch to Muse Wonder in the api". No session ran with Full.

## How it runs

- **Sign-in.** The same pay-as-you-go API key as round 1 (`MUSE_AUTH=key`, the launcher's default), read from `legal-agent-workshop/agent/.env` and given only to `muse`.
- **The model id** is `muse-wonder-1.0` (`MUSE_MODEL`), from [Meta's price list](https://dev.meta.ai/docs/pricing-rate-limits). A one-prompt smoke test confirms that the key may call it before round 2.
- **Reasoning effort `max`** (`MUSE_EFFORT`): Muse Wonder "supports reasoning effort levels from "minimal" to "max"". Round 1 ran at Muse Code's default, `high`.
- **No contributor mode.** It is a subscription setting, so the launcher passes `--contributor-mode` only with `MUSE_AUTH=account`. Meta's standard API terms apply to what is sent; everything the agents read is public in this repository, as in round 1.
- The model and effort are passed on the command line, so the session log records them.
- **Full stays possible:** `MUSE_AUTH=account`, `MUSE_MODEL=muse-wonder-full-1.0` and one `MUSE_AUTH=account tools/muse-agent/start.sh login`. That id is unconfirmed: Meta's pages give none for Full, and it comes from a third-party page.

## Naming the author

surucuonur chose to name the real model wherever the author is identified.
- **The co-author line** for round 2 is `Co-Authored-By: Muse Wonder (Meta Model API) <muse-wonder@meta-model-api.invalid>`, and commit subjects start with "Muse Wonder:". Round 1's commits keep "Muse Spark (Meta Model API)".
- **Where it comes from.** `tools/provenance.py` maps each model id prefix to its name and co-author line (`MUSE_AUTHORS`); the longest matching prefix wins, so `muse-wonder-full-` and `muse-wonder-` stay apart. The launcher puts both into the lead's first prompt, so the lead needn't know its own model id, and the transcript shows what it was told. No tool call in round 1 used an environment variable, so a variable set by the launcher couldn't be relied on.
- **The hooks and the audit** accept any Muse agent's co-author. The audit counts the commits of each.
- **"The Muse agent"** is the name for every model that works on `muse-agent` in a round. The paths `muse-spark/`, `muse-agent` and `~/.muse-spark` keep their names, so round 1's history and every path in its documents stay valid.

## Consequences

- **Price.** Billed per token: $4.00 per million input tokens, $0.20 cached and $20.00 output ([Meta's prices](https://dev.meta.ai/docs/pricing-rate-limits)), against $0.002, $0.10 and $0.20 for round 1's contributor model.
  - Round 1's tokens (86.8M input, 83.5M of them cached, and 1.04M output) cost about $0.70. The same tokens at Muse Wonder's prices cost about $50.
  - At `max` effort the output, the dearest part, is likely to grow; the round has no attempt limit, so the cost has no fixed ceiling. The session log counts the tokens.
- **Rate limits:** 3,000 requests and 4M tokens per minute per team, enough for 12 agents.
- **Round 2 changes several things at once:** the model, the effort, the skill (version 5, [ADR 0009](0009-rounds-build-on-each-other.md)) and the grader's rules ([ADR 0010](0010-exact-at-every-size.md)). A difference between the rounds can't be put down to the model alone, and the write-up says so.
- **Muse Wonder is in preview,** in Meta's words.

## Considered options

- **Muse Wonder Full through a Muse Code subscription.** The higher-capability variant at a flat monthly price, but it needs a sign-in and a Power Usage or Maximum Usage plan, its id is unconfirmed, and its allowance could run out mid-round. Chosen first, then dropped by surucuonur for the API.
- **Standard `muse-spark-1.3`.** Round 1's model family at about 12 times the contributor price. Rejected: not the most capable.
- **Keep "Muse Spark" as the co-author.** The smallest change, but round 2's commits would misname their model.
- **Rename the agent everywhere,** folder and branch included. It would rewrite paths that round 1's documents already use.
