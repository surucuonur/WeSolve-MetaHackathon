# Muse Spark's work

*This page was written by Claude Code (Claude Opus 5.5), directed by surucuonur. Everything Muse Spark wrote is marked below. [Who made what](../PROVENANCE.md).*

This folder holds Meta's Muse Spark (`muse-spark-1.3-contributor`) output, exactly as its API returned it, next to the harness's record of each call. Muse Spark chose which bottleneck to attack and wrote the kernels. The task and rules it was given are in [claude/muse-spark-brief.md](../claude/muse-spark-brief.md). The decision behind this setup is in [ADR 0003](../claude/adr/0003-muse-spark-writes-kernels.md).

## Whose each file is

| File | Written by | What it is |
|---|---|---|
| `calls/NNNN-*/response.json` | **Muse Spark** (raw API response, verbatim) | Everything the API returned for one call: the response id, the answer and token counts |
| `strategy.md` | **Muse Spark** | Its choice of target and its plan, extracted from the first call's answer |
| `kernels/<target>/attempt-N/kernel.metal` | **Muse Spark** | The Metal source of one attempt |
| `kernels/<target>/attempt-N/spec.json` | **Muse Spark** | The launch spec of that attempt: inputs, outputs, grid and threadgroup |
| `calls/NNNN-*/request.json` | the harness (Claude's code) | What was sent: model, settings and messages. Never the API key or headers. |
| `calls/NNNN-*/meta.json` | the harness | Time, response id, token counts, cost, and the SHA-256 of the answer |
| `kernels/<target>/attempt-N/result.json` | the harness | What the checks found: compile, correctness, timing |
| `claude-fixed/` | **Claude**, only if no attempt passes | A separately labelled version, with its diff against Muse Spark's original |

Files with Muse Spark's content start with a few lines marked `[harness]`, naming the call they came from. They are followed by the line `--- Muse Spark output below, unmodified ---`. Everything after that line is Muse Spark's answer, byte for byte. [bench/muse/verify.py](../bench/muse/verify.py) re-extracts it from `response.json` and checks the hash.
