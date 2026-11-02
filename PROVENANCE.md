# Who made what

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur.*

The hackathon rules require judges to tell what was built during the event, and by whom. This page lists every contributor to this repository, what each one made, and how to check it.

## The contributors

| Who | What they made | Where it is | How it is marked |
|---|---|---|---|
| **surucuonur** (the entrant) | Direction and every decision; ran the machine; reviewed the results | every commit | Author and committer of every commit |
| **Claude Code** (Claude Opus 5.5, Anthropic) | The benchmark and test-harness code, the measurements, every write-up | `bench/`, `claude/`, `README.md`, this page, and the Markdown in `results/` | Commits co-authored by `Claude Opus 5.5 <noreply@anthropic.com>`, which GitHub shows as the `claude` account; a byline under each document's title |
| **Muse Spark** (`muse-spark-1.3-contributor`, Meta Model API) | Its choice of which bottleneck to attack, its plan, and its GPU kernels, exactly as returned | `muse-spark/` | Commits co-authored by `Muse Spark (muse-spark-1.3-contributor)`, with Meta's response ids as trailers; a `[harness]` header above each file's verbatim content |
| **Others, before the event** | vLLM, vllm-metal, mlx-lm, MLX, and Meta's MobileMoE-S weights | `third_party/` (pinned submodules) and the installed wheels; the weights are downloaded, not committed | Their own authors and licences |

## How commits are marked

Every commit is authored and committed by surucuonur. The `Co-Authored-By` trailer says whose work the content is:

```
Claude's work                         Muse Spark's work
-------------                         -----------------
Author: surucuonur                    Author: surucuonur
Co-Authored-By: Claude Opus 5.5       Co-Authored-By: Muse Spark (muse-spark-1.3-contributor)
  <noreply@anthropic.com>               <muse-spark@meta-model-api.invalid>
Claude-Session: <session link>        Muse-Spark-Response: resp_...   (one per API call)
                                      Muse-Spark-Call: muse-spark/calls/...
                                      Claude-Session: <session link>
```

- **One author's content per commit.** Muse Spark's output never shares a commit with harness code or logs. Commits holding it have subjects starting with "Muse Spark:".
- **What `Claude-Session` means on a Muse Spark commit:** only that the commit was made through Claude Code. Claude is not a co-author of that content.
- **The Muse Spark address is not a real mailbox.** The `.invalid` domain is reserved and can never belong to anyone (RFC 2606). Muse Spark has no GitHub account, so we neither claim nor invent one.
- **Meta issues no signature for its answers.** The record is three things: the response id Meta's API returned, the raw response saved verbatim in `muse-spark/calls/`, and a SHA-256 hash of the content. `bench/muse/verify.py` re-checks the hashes.
- **Claude never edits Muse Spark's output.** If no Muse Spark kernel passes, Claude may write a separate version, labelled as Claude's, with its diff against Muse Spark's original ([ADR 0003](claude/adr/0003-muse-spark-writes-kernels.md)).

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
# Author, committer and co-authors of every commit
git log --format='%h %an | committer %cn | %(trailers:key=Co-Authored-By,valueonly,separator=%x2C )'

# Only the commits that hold Muse Spark's output
git log --format='%h %s' --grep='Co-Authored-By: Muse Spark'

# Muse Spark's files match the API responses they came from, and the total cost
.venv/bin/python bench/muse/verify.py
```
