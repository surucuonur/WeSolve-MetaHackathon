# Who made what

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur.*

The hackathon rules require judges to tell what was built during the event, and by whom. This page lists every contributor to this repository, what each one made, and how to check it.

## The contributors

| Who | What they made | Where it is | How it is marked |
|---|---|---|---|
| **surucuonur** (the entrant) | Direction and every decision; ran the machine; started the Muse Spark sessions; reviewed the results | every commit | Author and committer of every commit |
| **Claude Code** (Claude Opus 5.5, Anthropic) | The benchmark, grader and provenance code, the measurements, every write-up, and Muse Spark's brief and round skill | `bench/`, `claude/`, `tools/`, `.agents/skills/`, `README.md`, this page, and the Markdown in `results/` | Commits co-authored by `Claude Opus 5.5 <noreply@anthropic.com>`, which GitHub shows as the `claude` account; a byline under each document's title |
| **Muse Spark** (`muse-spark-1.3-contributor`; Meta Model API), working in Muse Code, Meta's own coding agent | Its analysis of the code, hypotheses, kernels and notes, the logs of both phases, and the transcripts of its sessions, every agent of the swarm included | `muse-spark/`, committed on the branch `muse-agent` and merged with `--no-ff` | Commits co-authored by `Muse Spark (Meta Model API)`, subjects starting "Muse Spark:"; the transcript records the model behind every model call |
| **Others, before the event** | vLLM, vllm-metal, mlx-lm, MLX, and Meta's MobileMoE-S weights | `third_party/` (pinned submodules) and the installed wheels; the weights are downloaded, not committed | Their own authors and licences |

## How the work is kept apart

Every commit is authored and committed by surucuonur. The `Co-Authored-By` trailer says whose work the content is:

```
Claude's work                         Muse Spark's work
-------------                         -----------------
branch: main                          branch: muse-agent
Author: surucuonur                    Author: surucuonur
Co-Authored-By: Claude Opus 5.5       Co-Authored-By: Muse Spark (Meta Model API)
  <noreply@anthropic.com>               <muse-spark@meta-model-api.invalid>
Claude-Session: <session link>        changes only muse-spark/
never changes muse-spark/
```

Muse Spark works in Muse Code, Meta's own coding agent, which surucuonur starts with `tools/muse-agent/start.sh` ([ADR 0006](claude/adr/0006-muse-spark-in-muse-code.md)). That script is the only way to start one. Each round follows the skill [.agents/skills/muse-round/SKILL.md](.agents/skills/muse-round/SKILL.md), which Claude wrote and committed before the round that uses it. Its version number is in its header and its history in [its changelog](.agents/skills/muse-round/CHANGELOG.md), and each round records the version it ran. In it, Muse Spark works as a swarm ([ADR 0007](claude/adr/0007-muse-spark-swarm.md)):
- explorers map the code and the measurements into a shared analysis;
- builders write and grade competing kernels;
- the lead coordinates, and only the lead commits;
- both phases are tracked in shared documents in `muse-spark/`.

Six layers keep its work and Claude's from mixing:

1. **Place.** Muse Spark has its own git worktree on `muse-agent` and writes only in `muse-spark/`. Its worktree has no `.env`. The launcher gives `muse` a home folder of its own, `~/.muse-spark`, so Muse Code finds none of surucuonur's personal skills and rules, Claude's included, and keeps its settings and session logs there. The API key goes only to the `muse` process.
2. **Identity.** Every Muse Spark commit carries the Muse Spark co-author, written by Muse Spark as the skill shows. The `.invalid` address is reserved and belongs to no one (RFC 2606); Muse Spark has no GitHub account, so we neither claim nor invent one.
3. **Hooks.** `tools/githooks/` refuses mixed commits: Muse Spark's co-author off `muse-agent`, Claude's on it, `muse-agent` commits without Muse Spark's co-author or outside `muse-spark/`, other commits inside it (merges of `muse-agent` excepted), and any staged API key.
4. **Audit.** `tools/provenance_check.py` re-checks the whole history against the same rules, so a commit made around the hooks is caught.
5. **Evidence.** Muse Code logs every model call, tool run and edit of the lead, its helpers and Muse Code's own reminder agents. `tools/export_transcript.py` saves each session in `muse-spark/transcripts/`.
   - **What it keeps:** every agent's conversation, how each tool call was approved, and the model of every model call.
   - **What it drops:** Muse Code's own instructions.
   - It refuses a session that contains a key.
   - The audit checks that a Meta model made every call.
6. **Independent grading.** The grader (`bench/grader/`) is Claude's; Muse Spark may run it but not change it, and it runs one GPU job at a time. Its results in the write-up come from re-running it, and G3, on the main checkout after the merge.

**The removed API stage.** Before the agent session, Muse Spark was called through a Claude-written API pipeline: 2 calls, a strategy and one routing+combine kernel. That stage was removed so all of Muse Spark's results come from one method. Its commits remain in the history: `d14c4a3` (the brief), `48cb61f` and `8c0a9b0` (Muse Spark's answers), `6c0a0ba` and `2c71eb4` (their logs). The audit lists them under "before".

**The deleted Claude Code stage.** From the tag `agent-era` until the move to Muse Code, Muse Spark worked as a Claude Code session powered by Meta's API ([ADR 0004](claude/adr/0004-muse-spark-as-claude-code-agent.md), [ADR 0005](claude/adr/0005-muse-spark-launcher-only-contributor-tier.md)). Its three sessions produced one strategy commit and one kernel attempt that was never graded. On 2026-11-02, surucuonur deleted that unpushed `muse-agent` branch and started over in Muse Code, so no graded result comes from that stage.

## A Muse Spark model in Claude's session (2026-11-02)

On 2026-11-02, before Meta's settings were confined to the launcher ([ADR 0005](claude/adr/0005-muse-spark-launcher-only-contributor-tier.md)), a switch that points Claude Code at Meta put Claude's own session in the main checkout on `muse-spark-1.3-contributor` for several turns.
- **What it saw:** those turns ran with Claude's configuration. They loaded Claude's memory notes and surucuonur's global instructions, and read this page, ADR 0004, the brief, the launcher and its settings. All of it went to Meta on the tier where Meta may train on it.
- **What it did:** its shell commands failed, and it wrote and committed nothing. No file in this repository comes from those turns.
- **What prevents it now:** Muse Spark runs only in Muse Code, started by `tools/muse-agent/start.sh`; no Claude Code configuration points at Meta.

## Personal context in the first Muse Code smoke test (2026-11-03)

Before round 1, Claude ran two short smoke tests with the launcher, at surucuonur's request. Each was a headless Muse Spark session in the worktree that only ran read-only checks: git status, the brief's first line, one MLX operation on the GPU and one read-only helper agent. A third test, after the move to a swarm (session `01a1a2eb-a2fd-79f1-a880-1d7b36e56322`), started two explorers. They wrote entries and log lines in a scratch folder outside `muse-spark/`, each added a note to the other's entry, and the folder was then deleted. None of the three committed anything. Between the second and third, surucuonur started one round session as a trial (`01a1a2b8-03a0-7890-baf7-9febe7b8d245`), on the round skill's version 1, and stopped it after 4 minutes; it wrote and committed nothing.
- **What went wrong in the first** (session `01a1a2a2-76ea-7fe3-95d8-7df76e209c74`):
  - The launcher then relied on Muse Code's settings for leaving personal skills and rules out, and they didn't work ([ADR 0006](claude/adr/0006-muse-spark-in-muse-code.md)).
  - The context of the lead and its helper held surucuonur's global instructions (`~/.claude/CLAUDE.md`) and the names and descriptions of 43 personal skills from `~/.agents/skills` and `~/.claude/skills`.
  - All of it went to Meta on the tier where Meta may train on it.
- **What it did:** the read-only checks; it wrote and committed nothing.
- **What prevents it now:** the launcher gives `muse` its own home folder. In the second smoke test (session `01a1a2a7-d562-76c3-a410-07629daf9bbc`), the lead and the helper saw only Muse Code's 25 bundled skills, its bundled plugin's skill and `muse-round`, and no rules file.

## What existed before the event

| Component | Version | Licence | Our changes |
|---|---|---|---|
| vLLM | v0.32.0 (`third_party/vllm`, read-only) | Apache-2.0 | none |
| vllm-metal | `e177b67` (fork `surucuonur/vllm-metal`) | Apache-2.0 | none |
| mlx-lm | upstream `9e6acca` (fork `surucuonur/mlx-lm`, branch `mobilemoe`) | MIT | the MobileMoE-S port: `796e73d` and `a6b990a`, made during the event |
| MLX | 0.32.4 wheel | MIT | none |
| MobileMoE-S weights | `facebook/MobileMoE-S-QAT`, `facebook/MobileMoE-S-SFT` | Meta's licence on Hugging Face | converted locally to MLX's format (`bench/convert_qat.py`), never committed |

Everything else in this repository was made between 2026-10-28 and the submission.

## History re-authored on 2026-11-02

The first commits were made with a work email that GitHub links to another account (onur-portai), which is the same person. On 2026-11-02, every commit in this repository and the two port commits in the mlx-lm fork were re-authored to surucuonur. Dates, messages and contents were unchanged, and the submodule links were updated to match.

Commit ids changed as a result. Raw run records (`results/runs/2026-10-29-baseline/meta.json`, `results/port/*.json`) keep the ids they recorded at the time. Map them with these tables:

| Before | After | Commit |
|---|---|---|
| `1264006` | `dd9a7d7` | Initial commit |
| `f62e432` | `3c5a630` | Set up MobileMoE-S bandwidth study skeleton |
| `5ea530e` | `b25c6c2` | Document the verified prebuilt install |
| `9e59ac3` | `6ecbbd1` | Port MobileMoE-S to mlx_lm and verify it |
| `846bf63` | `44a9ca8` | Add the benchmark sweep and switch the baseline to int4 QAT |
| `7fc5cff` | `879c4c4` | Add the baseline sweep results |
| `953a0d2` | `d5a54c9` | Add the baseline benchmark write-up |
| `8a20dfa` | `95c452d` | Find where a decode step's time goes |
| `1739575` | `1a98f66` | Break a decode step into pieces and measure each one |
| `de102d0` | `02076b7` | Measure GPU memory traffic with Xcode's counters |
| `f657d36` | `be7bea7` | Explain where the decode bottleneck is, with worked numbers |

| mlx-lm fork, before | After | Commit |
|---|---|---|
| `854017b` | `796e73d` | Add MobileMoE model support |
| `d228505` | `a6b990a` | Support Meta's int4 MobileMoE checkpoints |

## How to check

```bash
# Who authored, committed and co-authored each commit
git log --all --format='%h %an | committer %cn | %(trailers:key=Co-Authored-By,valueonly,separator=%x2C )'

# Only Muse Spark's commits, and which files they change
git log --all --stat --grep='Co-Authored-By: Muse Spark'

# The full audit: identities, folder rules, transcript models, secret scan
.venv/bin/python tools/provenance_check.py
```
