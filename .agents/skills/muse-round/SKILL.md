---
name: muse-round
description: Run one Muse Spark round on the MobileMoE-S decode kernels as a swarm. Explorers map the code and measurements into a shared analysis; then competing builders write and grade fused Metal kernels for the top inefficiencies. Both phases are tracked in shared documents.
---

# One Muse Spark round: a swarm explores, then builds

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. Committed on main and tagged `muse-code-era` before the first Muse Code session; each later version is committed before the round that uses it. [Who made what](../../../PROVENANCE.md).*

**Skill version 3.** This file, the role cards and tracking.md are versioned together; [CHANGELOG.md](CHANGELOG.md) lists every version. Each round records the version it ran (see *Versions* in tracking.md).

You are Muse Spark, the **lead** of a Muse Code session in your own worktree on the branch `muse-agent`. `claude/muse-agent-brief.md` fixes the task, the four slots, the tools and limits, the grader command and the pass criteria (G1, G2). Two ADRs change it in four ways:

1. **Harness** ([ADR 0006](../../../claude/adr/0006-muse-spark-in-muse-code.md)). You are a Muse Code session, not a Claude Code one. Muse Code adds no trailer, so you write it on every commit yourself (see *Commits*).
2. **No attempt limit and no stop rule** (ADR 0006). You decide when a slot is done. Every graded attempt is committed, so the history shows every try.
3. **A swarm** ([ADR 0007](../../../claude/adr/0007-muse-spark-swarm.md)). Helpers explore and build; you coordinate and commit.
4. **Files** (ADR 0007). `muse-spark/exploration/` and `muse-spark/implementation/` hold the round's shared documents, and they replace `strategy.md`. Kernels and `summary.md` stay where the brief puts them.

The rest of the brief stands. The goal is unchanged: kernels that pass G1 and G2 and beat the compiled reference at 1, 8 and 32 rows, faster because they remove a cause you can name.

## The swarm

| Role | Name | Task | Card |
|---|---|---|---|
| **Lead** (you) | | Start the helpers, keep the two READMEs, settle clashes, commit | this file |
| **Explorer** | `explorer-<territory>` | Map one territory into entries, and find where it loses time | [explorer.md](explorer.md) |
| **Verifier** | `verifier` | Check every inefficiency against its sources | [verifier.md](verifier.md) |
| **Builder** | `builder-<n>-<letter>` | Write and grade one solution for one hypothesis: one attempt | [builder.md](builder.md) |
| **Boundary agent** | `boundary-<n>` | Write what a hypothesis's solution must keep identical, and check passing attempts against it | [boundary.md](boundary.md) |

You don't explore, build or grade yourself. Start each helper with a prompt of this shape:

> You are `<name>`, a `<role>` in this round's swarm. Read `.agents/skills/muse-round/<card>` and `.agents/skills/muse-round/tracking.md`, then: `<task>`.

[tracking.md](tracking.md) describes the shared documents (the two phase folders, their READMEs, entries and logs) and the rules every agent keeps. Read it before starting anyone. Up to 12 agents run at once, you included.

**A checkpoint** is what you do each time a helper returns, and whenever the log shows something to settle:
1. **Read the log.** Read the phase log since your last checkpoint.
2. **Settle clashes.** When two entries describe one cause, keep the fuller one and set the other's status to `superseded by <entry>`. When two helpers work on one item, redirect one of them (`subagent_send_message`). Log what you decided.
3. **Refresh the tracker.** Update the phase README's tracker table, and raise the version of each hypothesis entry that changed.
4. **Raise the version.** Raise the phase README to its next version, and add its line to `## Versions`, naming the entries that changed and their versions (see *Versions* in tracking.md).
5. **Commit** the phase folder, plus any attempt that came back, as that version (see *Commits*).

## The round

### 0. Find where you are

- Run `git log --oneline main -1`, then `git log --oneline --grep '^Muse Spark:' muse-code-era..HEAD`, which lists your earlier commits.
- **Round number:** `claude/findings/muse-round-<N>.md` holds Claude's measurement of round N, run after merging your branch. This round is one more than the highest N. With no such file, it is round 1.
- From round 2 on, read the newest of those files first. It has the G3 results of your attempts, and it is the first evidence for this round. The phase folders already exist:
  - explorers add a `## Round N` section to their entries rather than starting over;
  - you start explorers only for the territories the G3 results reopen;
  - the logs carry on.
- Read the brief.

Done when you can name the round number, your committed attempts per slot, and the latest G3 results, if any.

### 1. Exploration

1. **Set up the folder.** Write `muse-spark/exploration/README.md` with a heading, its version line (v1, with skill version 3), the tracker (one row per territory) and `## Versions`, and an empty `log.md`. From round 2 on, the README carries on from its last version.
2. **Start the explorers and the verifier,** all at once, with these starting territories:

   | Territory | Covers |
   |---|---|
   | `model` | `third_party/mlx-lm/mlx_lm/models/mobilemoe.py`: the decoder layer, `MoE.__call__` and everything around the slots |
   | `switch-layers` | `third_party/mlx-lm/mlx_lm/models/switch_layers.py`: the routed experts' path, `gather_qmm`, the sort from 16 rows |
   | `quantized-kernels` | `quantized.h` and `quantized_utils.h` in `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/`: the `affine_gather_*` kernels behind `gather_qmm` |
   | `other-kernels` | the other Metal kernels in the same folder that the routing and combine slots launch: sort, softmax, reductions, elementwise, copies |
   | `vllm-metal` | `third_party/vllm-metal/vllm_metal/`: how the engine runs a decode step around the model |
   | `grader` | `bench/grader/`: the slot interface, what G1, G2 and the piece timing measure, and so the boundary every solution keeps |
   | `findings` | `claude/findings/` and `results/runs/2026-11-01-breakdown/` |
   | `runs` | `results/runs/2026-11-02-gpu-counters/`, `2026-10-29-baseline/` and `2026-10-30-micro/` |

3. **Refocus as costs show up.** When an entry shows that one file holds most of a territory's cost, split it off as a territory of its own, with its own explorer. Merge or close a territory that turns out to hold nothing, and add one when an inefficiency points somewhere no one covers. Log each change and update the tracker.
4. **Checkpoint** each time a helper returns.

**Done** when every territory entry is `done`, the verifier has returned, and every inefficiency is `confirmed`, `refuted` or `superseded`.

### 2. The analysis

Write the final `muse-spark/exploration/README.md`. It is the round's report on the code, for this round's builders and for future improvements:

- **Tracker:** the final table.
- **Context:** the decode step, the MoE block and its slots, and what the measurements say about where the time goes.
- **Territory map:** how the territories feed each other. For each territory: what it receives, what it passes on, and its measured cost.
- **Inefficiencies, ranked:**
  - the confirmed ones inside the slots, ranked by predicted saving as a share of the decode step, averaged over 1, 8 and 32 rows;
  - each with its cost, kind and entry;
  - Claude's findings marked as Claude's, and the swarm's own reading marked as yours.
- **Recommendations:** the confirmed inefficiencies outside the slots, with their evidence. They're for surucuonur's team.
- **Refuted claims,** in one line each, so no one chases them again.
- **Files read:** links to each territory entry's list.

From round 2 on, add a `## Round N` section: what the G3 results confirm, contradict or reveal.

Commit it. **Done** when every ranked item links a confirmed entry and every number in the README has its source.

### 3. Implementation

Take the ranked inefficiencies in order, at most two at a time, in different slots so their attempts never touch. Write `muse-spark/implementation/README.md` with a heading, its version line, the tracker and `## Versions`, and an empty `log.md`. For each hypothesis:

1. **Open the hypothesis.** Write `muse-spark/implementation/hyp-<n>-<slug>.md`, with status `boundary`, the inefficiency and the slot. Add its tracker row and log it.
2. **Draw its boundary.** Start `boundary-<n>` to write the boundary, and wait for it.
3. **Compete.**
   - Set the status to `building`.
   - Start 2 or 3 builders, `builder-<n>-a`, `-b` and `-c`. Give each a different direction drawn from the analysis: for example, fusing kernels, reading each weight byte once, or a different threadgroup shape.
4. **Collect.** Each time a builder returns:
   - checkpoint, committing its attempt folder with the hypothesis entry, the README and the log;
   - if the attempt passed G2, start `boundary-<n>` to check it.
5. **Harden.**
   - Pick the best attempt: it passes G1, G2 and its boundary check, and it is the fastest against the compiled reference across the three row counts.
   - Set the status to `hardening`, and start a builder for that attempt's next single change.
   - Repeat while the attempt still has a cause you can name and remove.
6. **Close.**
   - **`ready`:** the best attempt passes G1, G2 and its boundary check, and beats the compiled reference at 1, 8 and 32 rows, and you can name no further change worth making. Write the **Decision**: the attempt, why it is best, and what G3 should show.
   - **`dropped`:** say why.

Then open the next-ranked inefficiency. **Done** when no confirmed inefficiency is left that you judge will pay. Write why in the implementation README.

### 4. Finish

1. **The implementation README:** the final tracker, one paragraph per hypothesis, and a **Ready to test** section. It names the attempt per slot that goes to G3, and the change in decode step period you expect at 1, 8 and 32 requests, with how you worked it out.
2. **The exploration README:** add *What building taught*, the analysis corrected by what the attempts showed.
3. **`muse-spark/summary.md`:** add a `## Round N` section with:
   - the skill version this round ran, and the final versions of both READMEs;
   - the final attempt per slot;
   - the number of graded attempts per slot, this round and in total;
   - the expected G3 change;
   - what you'd try next;
   - your recommendations outside the slots.
4. **Commit,** then reply with this round's commits and the words "Round N finished".

## Your own rules

- **Writes:**
  - you write the two READMEs, the hypothesis entries' status blocks and decisions, and `summary.md`;
  - every helper writes only what its card names;
  - all of it stays in `muse-spark/`, and a commit outside it is refused.
- **G3 and vLLM belong to surucuonur's team,** who run them after merging your branch.
- **The rules every agent keeps** are in tracking.md.

## Commits

You are the only agent that commits. Commit only on `muse-agent`.
- **What to add:** the phase folders that exist so far (only `exploration` until implementation starts) and the attempt folders that came back. An attempt another builder is still writing waits for its own commit. An entry that is mid-edit is committed as it stands, and finished in a later commit.
- **The message:**
  - a checkpoint's subject is `Muse Spark: <phase> v<N>: <what changed>`, the version it records;
  - other commits start with `Muse Spark:`;
  - every message ends with this exact trailer.

```bash
git add muse-spark/exploration muse-spark/implementation muse-spark/kernels/<slot-folder>/attempt-<N>
git commit -m "Muse Spark: implementation v4: experts attempt 3 graded, passes G2" \
  -m "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
```

Use no other co-author. The hooks refuse a commit without this trailer, a commit outside `muse-spark/`, and any merge. Push, merge, rebase, git config and hook skipping stay with surucuonur.
