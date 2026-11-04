---
status: accepted
---

# From round 2, the Muse agent is Muse Wonder Full, signed in to a Muse Code subscription

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

**The change.** From round 2 (decided 2026-11-03), the launcher runs **Muse Wonder Full** instead of Muse Spark (`muse-spark-1.3-contributor`, round 1). surucuonur asked for it, wanting "to see improved results after this step". Meta calls Muse Wonder "Meta's most capable model", and Full "the higher-capability variant" ([Meta's model docs](https://dev.meta.ai/docs/models)).

## How it runs

- **Sign-in.** Full "is available only in Muse Code with a subscription", on the Power Usage and Maximum Usage plans. It isn't on the pay-as-you-go Model API, which round 1 used through its API key.
  - The launcher's sign-in is now the account by default (`MUSE_AUTH=account`), and it unsets `META_API_KEY`: "An API key always takes priority over a browser sign-in" ([Muse Code's sign-in docs](https://dev.meta.ai/docs/muse-code/auth)).
  - `tools/muse-agent/start.sh login` runs `muse login` with Muse's own home folder, `~/.muse-spark`, where the login stays.
  - `MUSE_AUTH=key` keeps round 1's API-key path.
- **The model id** is `muse-wonder-full-1.0` (`MUSE_MODEL`), which is **not yet confirmed**. Meta's pages give no id for Full; this one comes from a third-party page, and the built-in model list of Muse Code 1.4.14 (the newest in Homebrew) names no Muse Wonder model. A one-call smoke test after signing in confirms it before round 2. If it is wrong, `MUSE_MODEL` and `MUSE_AUTHORS` in `tools/provenance.py` change.
- **Reasoning effort `max`** (`MUSE_EFFORT`): Muse Wonder "supports reasoning effort levels from "minimal" to "max"". Round 1 ran at Muse Code's default, `high`.
- **Contributor mode on** (`MUSE_CONTRIBUTOR`), the subscription's default. It gives more usage in exchange for letting Meta train future models. Everything the agents read is public in this repository, as in round 1.
- The model, effort and contributor mode are passed on the command line, so the session log records them.

## Naming the author

surucuonur chose to name the real model wherever the author is identified.
- **The co-author line** for round 2 is `Co-Authored-By: Muse Wonder Full (Muse Code) <muse-wonder-full@muse-code.invalid>`, and commit subjects start with "Muse Wonder Full:". Round 1's commits keep "Muse Spark (Meta Model API)".
- **Where it comes from.** `tools/provenance.py` maps each model id prefix to its name and co-author line (`MUSE_AUTHORS`). The launcher puts both into the lead's first prompt, so the lead needn't know its own model id, and the transcript shows what it was told. No tool call in round 1 used an environment variable, so a variable set by the launcher couldn't be relied on.
- **The hooks and the audit** accept either Muse co-author. The audit counts the commits of each.
- **"The Muse agent"** is the name for both: the Meta model that works on `muse-agent` in a round. The paths `muse-spark/`, `muse-agent` and `~/.muse-spark` keep their names, so round 1's history and every path in its documents stay valid.

## Consequences

- **Price.** A flat monthly plan instead of billing per token.
  - Third-party pages put Power Usage at $50 and Maximum Usage at $200 a month, with allowances that reset every 5 hours and every week. The Meta pages we read don't state these.
  - For scale: round 1's tokens (86.8M input, 83.5M of them cached, and 1.04M output) cost about $0.70 on `muse-spark-1.3-contributor`. At `muse-wonder-1.0`'s pay-as-you-go prices ($4.00 input, $0.20 cached input and $20.00 output per million tokens; [Meta's prices](https://dev.meta.ai/docs/pricing-rate-limits)), the same tokens would cost about $50.
- **The allowance can run out mid-round,** at `max` effort with up to 12 agents. If it does, a later session carries on from the round's documents and commits (the skill's step 0).
- **Round 2 changes several things at once:** the model, the effort, and the skill (version 4, [ADR 0009](0009-rounds-build-on-each-other.md)). A difference between the rounds can't be put down to the model alone, and the write-up says so.
- **Muse Wonder is in preview,** in Meta's words.
- **Rate limits** for subscriptions aren't published. The Model API's limit for `muse-wonder-1.0` is 3,000 requests and 4M tokens per minute per team.

## Considered options

- **`muse-wonder-1.0` on the Model API, with the current key.** No subscription, but it isn't Full, a round of round 1's size would cost about $50, and it has no contributor mode. Rejected: surucuonur asked for Full.
- **Standard `muse-spark-1.3`.** Round 1's model family at about 12 times the contributor price, with `max` effort, which the contributor tier lacks. Rejected: not the most capable.
- **Keep "Muse Spark" as the co-author.** The smallest change, but round 2's commits would misname their model and how it was reached.
- **Rename the agent everywhere,** folder and branch included. It would rewrite paths that round 1's documents already use.
