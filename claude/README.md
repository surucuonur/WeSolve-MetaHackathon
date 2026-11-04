# Claude Code's documents

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../PROVENANCE.md).*

Everything in this folder was written by Claude Code (Claude Opus 5.5), directed and reviewed by surucuonur. The numbers come from the scripts in [bench/](../bench/) and the raw data in [results/](../results/). The Muse agent's work is kept apart, in `muse-spark/`, on the `muse-agent` branch until it is merged.

| Path | What it is |
|---|---|
| [GLOSSARY.md](GLOSSARY.md) | The shared vocabulary |
| [adr/](adr/) | Decisions and why they were made |
| [benchmark/](benchmark/) | The baseline benchmark: findings, every measurement, the method |
| [findings/](findings/) | Where a decode step's time goes, and the fixes it suggests |
| [muse-agent-brief.md](muse-agent-brief.md) | The task, rules and pass criteria given to the Muse Spark agent, committed and tagged `agent-era` before its session |
| [muse-swarm.md](muse-swarm.md) | How the Muse agent's swarm runs a round, step by step: exploration and implementation, the shared documents and versions, how rounds build on each other, and how to follow a round live |
| [diagrams/](diagrams/) | The swarm's diagrams: `muse-swarm.drawio` (open in draw.io; overview, exploration and implementation pages) and an SVG export of each page |
| [../.agents/skills/muse-round/SKILL.md](../.agents/skills/muse-round/SKILL.md) | The round skill the Muse agent follows in Muse Code: a swarm of explorers writes a shared analysis, then competing builders write and grade kernels, tracked in shared documents; with one card per role and `tracking.md` beside it ([ADR 0006](adr/0006-muse-spark-in-muse-code.md), [ADR 0007](adr/0007-muse-spark-swarm.md)); from round 2 on, each round starts from the earlier rounds and must beat their best attempts ([ADR 0009](adr/0009-rounds-build-on-each-other.md)) |
