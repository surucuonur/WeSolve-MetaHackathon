---
name: muse-round
description: Run one round of the Muse agent on the MobileMoE-S decode kernels as a swarm. Explorers map the code and measurements into a shared analysis; then competing builders write and grade fused Metal kernels for the top inefficiencies. From round 2 on, the round starts from the previous rounds' results and must beat their best attempts. Both phases are tracked in shared documents.
---

# One round of the Muse agent: a swarm explores, then builds

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. Committed on main and tagged `muse-code-era` before the first Muse Code session; each later version is committed before the round that uses it. [Who made what](../../../PROVENANCE.md).*

**Skill version 6.** This file, the role cards and tracking.md are versioned together; [CHANGELOG.md](CHANGELOG.md) lists every version. Each round records the version it ran (see *Versions* in tracking.md).

You are the **lead** of a Muse Code session in your own worktree on the branch `muse-agent`. The first prompt names the Meta model you run as, the name your commit subjects start with, and the co-author line every commit ends with. `claude/muse-agent-brief.md` fixes the task, the slots, the tools and limits, the grader command and the pass criteria (G1, G2). Four ADRs change it in five ways:

1. **Harness** ([ADR 0006](../../../claude/adr/0006-muse-spark-in-muse-code.md)). You are a Muse Code session, not a Claude Code one. Muse Code adds no trailer, so you write the first prompt's co-author line on every commit yourself (see *Commits*).
2. **No attempt limit and no stop rule** (ADR 0006). The brief's *When a slot stops* no longer applies: you decide when a slot is done. Every graded attempt is committed, so the history shows every try.
3. **A swarm** ([ADR 0007](../../../claude/adr/0007-muse-spark-swarm.md)). Helpers explore and build; you coordinate and commit.
4. **Files** (ADR 0007, [ADR 0009](../../../claude/adr/0009-rounds-build-on-each-other.md)). Each round writes in its own **round folder**: round 1 at the top of `muse-spark/`, round N from 2 on in `muse-spark/round-<N>/`. It holds `exploration/` and `implementation/`, the round's shared documents, which replace `strategy.md`; `kernels/<slot-folder>/attempt-<M>/`; and `summary.md`. tracking.md writes it `<round>/`.
5. **Rounds build on each other** (ADR 0009). A finished round's files are frozen: read them and cite them, and the hooks refuse any change to them. From round 2 on, a slot may have a **baseline**: the best attempt so far that passes the current G2 and that Claude's G3 accepted, faster than the unchanged model and the compiled reference at 1, 8 and 32 requests. A new attempt must beat it in the same grader run. A slot without one has a **starting point** instead: an earlier attempt to build from, which Claude's measurement names, with what kept it from being a baseline.

The rest of the brief stands. The goal: kernels that pass G1 and G2, beat the compiled reference at 1, 8 and 32 rows and, where the slot has a baseline, beat it too, faster because they remove a cause you can name. The Muse agent was Muse Spark in round 1 and is Muse Wonder from round 2 ([ADR 0008](../../../claude/adr/0008-muse-wonder.md)); documents written in earlier rounds say "Muse Spark".

## The swarm

| Role | Name | Task | Card |
|---|---|---|---|
| **Lead** (you) | | Start the helpers, keep the two READMEs, settle clashes, commit | this file |
| **Explorer** | `explorer-<territory>` | Map one territory into entries, and find where it loses time | [explorer.md](explorer.md) |
| **Verifier** | `verifier` | Check every inefficiency against its sources | [verifier.md](verifier.md) |
| **Builder** | `builder-<n>-<letter>` | Write and grade one solution for one hypothesis: one attempt | [builder.md](builder.md) |
| **Boundary agent** | `boundary-<n>` | Write what a hypothesis's solution must keep identical, and check passing attempts against it | [boundary.md](boundary.md) |

You don't explore, build or grade yourself. Start each helper with a prompt of this shape:

> You are `<name>`, a `<role>` in round `<N>`'s swarm; this round writes in `<round folder>`. Read `.agents/skills/muse-round/<card>` and `.agents/skills/muse-round/tracking.md`, then: `<task>`.

[tracking.md](tracking.md) describes the shared documents (the two phase folders, their READMEs, entries and logs) and the rules every agent keeps. Read it before starting anyone. Up to 12 agents run at once, you included.

**A checkpoint** is what you do each time a helper returns, and whenever the log shows something to settle:
1. **Read the log.** Read the phase log since your last checkpoint.
2. **Settle clashes.** When two entries describe one cause, keep the fuller one and set the other's status to `superseded by <entry>`. When two helpers work on one item, redirect one of them (`subagent_send_message`). Log what you decided.
3. **Refresh the tracker.** Update the phase README's tracker table, and raise the version of each hypothesis entry that changed.
4. **Raise the version.** Raise the phase README to its next version, and add its line to `## Versions`, naming the entries that changed and their versions (see *Versions* in tracking.md).
5. **Commit** the phase folder, plus any attempt that came back, as that version (see *Commits*).

## The round

### 0. Find where you are

- Run `git log --oneline main -1`, then `git tag --list -n1 'muse-round-*'`, which lists the finished rounds. **This round** is one more than the highest; with none, it is round 1. Its folder is `muse-spark/round-<N>/` (round 1: `muse-spark/`).
- `git log --oneline --grep '^Muse ' muse-code-era..HEAD` lists the Muse agent's commits of every round.
- **From round 2 on,** read Claude's measurement of the previous round, `claude/findings/muse-round-<N-1>.md`. It has the G3 results and names each slot's baseline or starting point. If it is missing, stop and reply "Claude's measurement of round <N-1> is missing": this round builds on it.
- If this round's folder already exists, a session stopped partway: carry on from its READMEs' last versions.
- Read the brief.

**Done** when you can name the round number and its folder, and from round 2 on each slot's baseline or starting point and the latest G3 results.

### 1. The starting point (from round 2 on)

This is the round's plan, built on what the earlier rounds learned. Read:
- the previous rounds' final exploration and implementation READMEs and `summary.md`;
- the `notes.md` of each baseline or starting-point attempt;
- Claude's measurement.

Then write the opening of this round's `exploration/README.md` (set up as in step 2.1), a section **Starting point**:
- **Per slot:**
  - the baseline, or the starting point where there is none, and its piece timing;
  - its G3 effect, and how that compares with what was predicted, with the reason where Claude's measurement gives one;
  - what keeps a starting point from passing G2, and where the attempt still spends its time.
- **What is left:**
  - the confirmed inefficiencies not yet built or not fully removed, ranked by predicted saving against the baselines, or the reference where a slot has none;
  - the slots not yet tried, for example `routing+combine`;
  - the recommendations outside the slots.
- **Refuted claims,** which no one chases again.
- **Territories** this round opens (step 2.2), each with the line of this section that reopens it.

Commit it as exploration v1. **Done** when every slot has its baseline or starting point and every territory has its reason.

### 2. Exploration

1. **Set up the folder.** Write `<round>/exploration/README.md` with a heading, its version line (v1, with skill version 6), the tracker (one row per territory) and `## Versions`, and an empty `log.md`.
2. **Start the explorers and the verifier,** all at once.
   - **Round 1** starts with these territories:

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

   - **From round 2 on,** the starting point lists the territories:
     - `baseline-<slot>` for each slot: its baseline's kernels and `spec.json`, which are now the code in the slot, or its starting point's. Map them with the checklist, and find where they still lose time and, for a starting point, why it fails G2;
     - `g3`: Claude's measurement of the previous round, where the end-to-end change differs from the piece timing, and why;
     - of round 1's territories, only those the starting point reopens. An explorer starts from that territory's earlier entries and writes only what is new.
3. **Refocus as costs show up.** When an entry shows that one file holds most of a territory's cost, split it off as a territory of its own, with its own explorer. Merge or close a territory that turns out to hold nothing, and add one when an inefficiency points somewhere no one covers. Log each change and update the tracker.
4. **Checkpoint** each time a helper returns.

**Done** when every territory entry is `done`, the verifier has returned, and every inefficiency is `confirmed`, `refuted` or `superseded`.

### 3. The analysis

Write the final `<round>/exploration/README.md`. It is the round's report on the code, for this round's builders and for future improvements:

- **Tracker:** the final table.
- **Starting point** (from round 2 on), as written in step 1, with any correction exploration made.
- **Context:** the decode step, the MoE block and its slots, and what the measurements say about where the time goes.
- **Territory map:** how the territories feed each other. For each territory: what it receives, what it passes on, and its measured cost.
- **Inefficiencies, ranked:**
  - the confirmed ones inside the slots, ranked by predicted saving as a share of the decode step, averaged over 1, 8 and 32 rows; from round 2 on, the saving is against the baseline where the slot has one;
  - each with its cost, kind and entry;
  - Claude's findings marked as Claude's, earlier rounds' entries linked as theirs, and this round's own reading marked as yours.
- **Recommendations:** the confirmed inefficiencies outside the slots, with their evidence. They're for surucuonur's team.
- **Refuted claims,** in one line each, so no one chases them again.
- **Files read:** links to each territory entry's list.

Commit it. **Done** when every ranked item links a confirmed entry and every number in the README has its source.

### 4. Implementation

Take the ranked inefficiencies in order, at most two at a time, in different slots so their attempts never touch. Write `<round>/implementation/README.md` with a heading, its version line, the tracker and `## Versions`, and an empty `log.md`. For each hypothesis:

1. **Open the hypothesis.** Write `<round>/implementation/hyp-<n>-<slug>.md`, with status `boundary`, the inefficiency, the slot and, from round 2 on, the slot's baseline folder or `none`. Add its tracker row and log it.
2. **Draw its boundary.** Start `boundary-<n>` to write the boundary, and wait for it.
3. **Compete.**
   - Set the status to `building`.
   - Start 2 or 3 builders, `builder-<n>-a`, `-b` and `-c`. Give each a different direction drawn from the analysis: for example, fusing kernels, reading each weight byte once, or a different threadgroup shape. From round 2 on, a direction may start from the baseline's or starting point's kernels, or from scratch.
4. **Collect.** Each time a builder returns:
   - checkpoint, committing its attempt folder with the hypothesis entry, the README and the log;
   - if the attempt passed G2, start `boundary-<n>` to check it.
5. **Harden.**
   - Pick the best attempt: it passes G1, G2 and its boundary check, and it is the fastest across the three row counts against the compiled reference and, where the slot has one, the baseline.
   - Set the status to `hardening`, and start a builder for that attempt's next single change.
   - Repeat while the attempt still has a cause you can name and remove.
6. **Close.**
   - **`ready`:** the best attempt passes G1, G2 and its boundary check, beats the compiled reference at 1, 8 and 32 rows and, where the slot has a baseline, has `beats_baseline` true, and you can name no further change worth making. Write the **Decision**: the attempt, why it is best, and what G3 should show.
   - **`dropped`:** say why. A slot whose hypotheses are all dropped keeps its baseline, or the reference where it has none.

Then open the next-ranked inefficiency. **Done** when no confirmed inefficiency is left that you judge will pay. Write why in the implementation README.

### 5. Finish

1. **The implementation README:** the final tracker, one paragraph per hypothesis, and a **Ready to test** section. It names the attempt per slot that goes to G3, this round's, the baseline kept, or the reference, and the change in decode step period you expect at 1, 8 and 32 requests against the previous round's, with how you worked it out.
2. **The exploration README:** add *What building taught*, the analysis corrected by what the attempts showed.
3. **`<round>/summary.md`:** a `## Round N` section with:
   - the model, the skill version this round ran, and the final versions of both READMEs;
   - the final attempt per slot;
   - the number of graded attempts per slot, this round and over all rounds;
   - the expected G3 change;
   - what you'd try next;
   - your recommendations outside the slots.
4. **Commit,** then reply with this round's commits and the words "Round N finished".

## Your own rules

- **Writes:**
  - you write the two READMEs, the hypothesis entries' status blocks and decisions, and `summary.md`;
  - every helper writes only what its card names;
  - all of it stays in this round's folder.
- **G3 and vLLM belong to surucuonur's team,** who run them after merging your branch.
- **The rules every agent keeps** are in tracking.md.

## Commits

You are the only agent that commits. Commit only on `muse-agent`.
- **What to add:** this round's phase folders that exist so far (only `exploration` until implementation starts) and the attempt folders that came back. An attempt another builder is still writing waits for its own commit. An entry that is mid-edit is committed as it stands, and finished in a later commit.
- **The message:**
  - every subject starts with the name the first prompt gives you, then a colon;
  - a checkpoint's subject is `<name>: <phase> v<N>: <what changed>`, the version it records;
  - every message ends with the first prompt's co-author line, exactly.

For example, in round 2 as Muse Wonder:

```bash
git add muse-spark/round-2/exploration muse-spark/round-2/implementation muse-spark/round-2/kernels/<slot-folder>/attempt-<M>
git commit -m "Muse Wonder: implementation v4: experts attempt 3 graded, beats baseline" \
  -m "Co-Authored-By: Muse Wonder (Meta Model API) <muse-wonder@meta-model-api.invalid>"
```

Use no other co-author. The hooks refuse a commit without a Muse agent's co-author, a commit outside `muse-spark/`, a change to a finished round's files, a new file outside this round's folder, and any merge. Push, merge, rebase, tags, git config and hook skipping stay with surucuonur.
