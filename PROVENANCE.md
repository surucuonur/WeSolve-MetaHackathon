# Who made what

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur.*

The hackathon rules require judges to tell what was built during the event, and by whom. This page lists every contributor to this repository, what each one made, and how to check it.

## The contributors

| Who | What they made | Where it is | How it is marked |
|---|---|---|---|
| **surucuonur** (the entrant) | Direction and every decision; ran the machine; started the Muse Spark session; reviewed the results | every commit | Author and committer of every commit |
| **Claude Code** (Claude Opus 5.5, Anthropic) | The benchmark, grader and provenance code, the measurements, every write-up | `bench/`, `claude/`, `tools/`, `README.md`, this page, and the Markdown in `results/` | Commits co-authored by `Claude Opus 5.5 <noreply@anthropic.com>`, which GitHub shows as the `claude` account; a byline under each document's title |
| **Muse Spark** (`muse-spark-1.3`, Meta Model API), working as a Claude Code agent | Its choice of targets, its kernels and notes, and the transcript of its session | `muse-spark/`, committed on the branch `muse-agent` and merged with `--no-ff` | Commits co-authored by `Muse Spark (Meta Model API)`, subjects starting "Muse Spark:"; the transcript records the model behind every turn |
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

Muse Spark works in a separate Claude Code session powered by Meta's API, which surucuonur starts with `tools/muse-agent/start.sh` ([ADR 0004](claude/adr/0004-muse-spark-as-claude-code-agent.md)). Six layers keep its work and Claude's from mixing:

1. **Place.** Muse Spark has its own git worktree on `muse-agent`, writes only in `muse-spark/`, and runs with its own Claude Code configuration folder. It sees none of Claude's history, memory or settings, and its worktree has no `.env`.
2. **Identity.** Its settings (`tools/muse-agent/settings.json`) put the Muse Spark co-author on its commits instead of Claude Code's default, which would credit the `claude` account. The `.invalid` address is reserved and belongs to no one (RFC 2606); Muse Spark has no GitHub account, so we neither claim nor invent one.
3. **Hooks.** `tools/githooks/` refuses mixed commits: Muse Spark's co-author off `muse-agent`, Claude's on it, `muse-agent` commits outside `muse-spark/`, other commits inside it (merges of `muse-agent` excepted), and any staged API key.
4. **Audit.** `tools/provenance_check.py` re-checks the whole history against the same rules, so a commit made around the hooks is caught.
5. **Evidence.** The session's conversation is saved in `muse-spark/transcripts/` by `tools/export_transcript.py`. Every assistant turn keeps the model that wrote it and Meta's message and request ids. Claude Code's own system prompt and tool definitions are dropped.
6. **Independent grading.** The grader (`bench/grader/`) is Claude's, fixed before the session; Muse Spark may run it but not change it. Its results in the write-up come from re-running it on the main checkout after the merge.

**The removed API stage.** Before the agent session, Muse Spark was called through a Claude-written API pipeline: 2 calls, a strategy and one routing+combine kernel. That stage was removed so all of Muse Spark's results come from one method. Its commits remain in the history: `d14c4a3` (the brief), `48cb61f` and `8c0a9b0` (Muse Spark's answers), `6c0a0ba` and `2c71eb4` (their logs). The audit lists them under "before".

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
