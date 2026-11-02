---
status: superseded by 0004
---

# Muse Spark chooses a bottleneck and writes the kernels; its output is kept verbatim and apart

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

> **Superseded by [ADR 0004](0004-muse-spark-as-claude-code-agent.md) on 2026-11-02.** Muse Spark now works as a Claude Code agent session. This API pipeline and its output were removed from the tree; git history keeps them, in commits `d14c4a3` through `2c71eb4`.

Muse Spark (`muse-spark-1.3-contributor`, Meta Model API) does real engineering work in this project: it writes GPU kernels.

1. It reads the measured findings and chooses which bottleneck to attack, from the parts of the decode step that the test harness can swap out.
2. It writes fused Metal kernels for that part with `mx.fast.metal_kernel`, in a loop:
   - the harness compiles each kernel, checks it against the original on real inputs, and times it;
   - it sends Muse Spark back only the machine output: compiler errors, test failures, timings.

**The rules are fixed before the first call.** The task, the parts that can be swapped, the answer format, the pass criteria and the attempt caps are in `claude/muse-spark-brief.md` (removed; see commit `d14c4a3`). It was committed before any call, and its commit time proves it.

**Muse Spark's output is never edited.**
- Its answers are saved verbatim in `muse-spark/`, committed apart from any other work, and co-authored by Muse Spark.
- Each file's hash is checked against the raw API response.
- If no attempt passes, Claude may write one separate version, labelled as Claude's and kept as a diff against Muse Spark's original. It is reported apart from Muse Spark's results.

**Every commit is authored by surucuonur;** the co-author shows whose work it holds ([PROVENANCE.md](../../PROVENANCE.md)).
- Muse Spark has no GitHub account, so its co-author address uses the reserved `.invalid` domain, which belongs to no one.
- Its proof of origin is the response id Meta's API returned, which each commit records.

**The contributor tier** costs about a twelfth of the standard tier: $0.10 against $1.25 per million input tokens, and $0.20 against $4.25 per million output tokens. In exchange, Meta may train on the prompts and answers. Everything sent is already public in this repository, apart from the API key, which is never sent in a prompt.

## Consequences

- **What the project shows.** Muse Spark's contribution is a measured result: which target it chose, how many attempts passed each check, and whether its kernel beats both the original and an `mx.compile` of the same code. A negative result is reported as it is.
- **The harness is Claude's work.** That covers what Muse Spark sees and what it is checked against: the slot definitions, input capture, checks and timing. The brief makes those choices public before the calls.
- **Safety.** Model-written Python is never run. Muse Spark's kernel source runs only through MLX's Metal compiler, in a separate process with a time limit.

## Considered options

- **Muse Spark only explains the findings** (the October plan in [decode-bottlenecks.md](../findings/decode-bottlenecks.md#using-the-muse-spark-credits)). Cheap, but close to "summarise these documents", and nothing it says gets tested.
- **Muse Spark as Claude Code's model.** Anthropic doesn't support routing Claude Code to non-Claude models, and judges would not see what Muse Spark did.
- **Meta's Muse Code agent writes the code.** A coding tool used to build the project; its work would be hard to separate from ours.
- **We choose the target.** Lower risk, but the choice would then be ours, not Muse Spark's.
- **Claude fixes Muse Spark's kernels in place.** Faster, but nobody could tell who wrote the result.
