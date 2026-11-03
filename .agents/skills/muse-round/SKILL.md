---
name: muse-round
description: Run one Muse Spark round on the MobileMoE-S decode kernels. Helper agents investigate the run evidence and the code, then Muse diagnoses the root causes and writes and grades fused Metal kernels.
---

# One Muse Spark round: investigate, diagnose, build

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. Committed on main and tagged `muse-code-era` before the first Muse Code session; each later version is committed before the round that uses it. [Who made what](../../../PROVENANCE.md).*

You are Muse Spark, the **lead** of a Muse Code session in your own worktree on the branch `muse-agent`. `claude/muse-agent-brief.md` fixes the task, the four slots, the tools and limits, your files, the grader command and the pass criteria (G1, G2). [ADR 0006](../../../claude/adr/0006-muse-spark-in-muse-code.md) changes the brief in three ways:

1. **Harness.** You are a Muse Code session, not a Claude Code one. Muse Code adds no trailer, so you write the trailer on every commit yourself (see *Commits*).
2. **No attempt limit and no stop rule.** You decide when a slot is done. Every graded attempt is committed, so the history shows every try.
3. **More files.** `muse-spark/diagnosis.md` and the drafts folder `muse-spark/drafts/` join the files the brief lists.

The rest of the brief stands. The goal is unchanged: kernels that pass G1 and G2 and beat the compiled reference at 1, 8 and 32 rows, faster because they remove a cause you can name.

## The round

### 0. Find where you are

- Run `git log --oneline main -1`, then `git log --oneline --grep '^Muse Spark:' muse-code-era..HEAD`, which lists your earlier commits.
- **Round number:** `claude/findings/muse-round-<N>.md` holds Claude's measurement of round N, run after merging your branch. This round is one more than the highest N. With no such file, it is round 1.
- From round 2 on, read the newest of those files first. It has the G3 results of your attempts, and it is the first evidence for this round.
- Read the brief.

Done when you can name the round number, your committed attempts per slot, and the latest G3 results, if any.

### 1. Investigate (explore)

Start the **helpers**, one per **lens**, and run them in parallel. They are read-only: they read and report, and change no files. Give each one its lens, the *System-aware checklist* below, and the source rule: every claim cites a file and line, a table (path and row), a grader verdict (commit), or arithmetic shown in full. The lenses:

1. **Run evidence:** `claude/findings/` (all of it), the `tables.md` files under `results/runs/`, and the GPU counter tables. For each slot at 1, 8 and 32 rows: time, kernels, bytes measured against the fewest bytes needed, and achieved bandwidth.
2. **Model code path:** `MoE.__call__` in `third_party/mlx-lm/mlx_lm/models/mobilemoe.py`, through `switch_layers.py`, down to each MLX op it calls.
3. **MLX kernels:** the Metal kernels those ops launch, in `.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/`. `quantized.h` holds the `affine_gather_*` kernels behind `gather_qmm`. How many times each weight byte is read, threadgroup sizes, and occupancy.
4. **System context:**
   - how the decode step reaches the MoE block, and what runs just before and after each slot;
   - what your replacement must keep identical: outputs, dtypes, rounding points, which reorderings are allowed;
   - the grader's slot interface: `bench/grader/slots.py` and `check_candidate.py`.

Each lens ends with a **report** that covers every piece of code it inspected with the checklist, and lists every file it read.

When the reports are in, start a **verifier** helper. It takes each claim the diagnosis will rest on, checks it against its cited source, and marks it confirmed, wrong, or without a source.

### 2. Diagnose

From the verified reports, write `muse-spark/diagnosis.md`. For each slot it gives:
- what it costs at 1, 8 and 32 rows;
- whether it is overhead-bound, bandwidth-bound or compute-bound, in the terms of `claude/GLOSSARY.md`;
- the arithmetic behind that call: kernels × the fixed cost per kernel, against bytes ÷ peak bandwidth;
- the **root cause**: what the code does, where (file and lines), and what it costs per step in kernels or bytes. "The experts are slow" is a symptom, not a cause.

Mark Claude's findings as Claude's and your own reading and arithmetic as yours. List the causes outside the four slots as **recommendations** with their evidence; they're for surucuonur's team. End with every file read and what each one gave.

From round 2 on, add a `## Round N` section: what the G3 results confirm, contradict or reveal.

Commit `diagnosis.md`. Done when every slot has a named root cause with its cost and sources, and every claim the verifier checked is either confirmed or gone.

### 3. Strategy

Write or revise `muse-spark/strategy.md` in the brief's format, with the targets JSON. For each design, give per layer at 1, 8 and 32 rows:
- the kernels it launches;
- the bytes it reads, counting how many times each weight byte is read;
- the threads in flight, against what the GPU holds;
- the time this predicts;
- which root cause it removes.

Commit it.

### 4. Attempts (exploit)

For each target slot, in order, repeat:

1. **Draft.** Two drafting helpers each write a different design into `muse-spark/drafts/<slot-folder>/<draft-name>/`, using the brief's attempt layout plus a `notes.md` with the design and the cause it targets. Each drafter writes only in its own draft folder.
2. **Review.** A reviewing helper checks each draft:
   - against the brief's spec format and the GPU limits;
   - against the slot's shapes, dtypes and rounding points at 1, 8 and 32 rows;
   - against the *System context* report;
   - against the diagnosis.
   
   It reports the defects it finds and its predicted time.
3. **Choose and grade.** Pick the draft, or a combination of drafts, most likely to remove the cause, and write it as `muse-spark/kernels/<slot-folder>/attempt-<N>/`, numbering on from your last attempt. Before grading, its `notes.md` says what changed since the last attempt, which cause it addresses, and your predicted time per layer at 1, 8 and 32 rows. Run the brief's grader command on it and wait for it to finish.
4. **Record.** Add the verdict to `notes.md`: G1; G2 with the worst case; your timings next to the original's and the compiled reference's. Commit the attempt with the drafts behind it.
5. **Learn.** When the attempt fails, or its time misses your prediction by more than 20%, send helpers to find out why, and write the answer in `notes.md` before the next attempt. The gap usually points to the next cause.

Make one main change per attempt, so the grader shows what helped. Write the simplest kernel that removes the cause. Keep going while there is a cause you can name and still remove. Move on, or end the round, when you judge further attempts won't pay, and say why in the summary.

### 5. Finish

Add a `## Round N` section to `muse-spark/summary.md` with:
- your final attempt per slot;
- the number of graded attempts per slot, this round and in total;
- the change in decode step period you expect G3 to show at 1, 8 and 32 requests, and how you worked it out;
- what you'd try next;
- your recommendations outside the slots.

Commit it, then reply with this round's commits and the words "Round N finished".

## System-aware checklist

Every lens and every review fills this in for each piece of code it inspects, so that no piece is judged without its neighbours:

- **Callers:** who calls it, and with what shapes at 1, 8 and 32 rows.
- **Callees:** what it calls, down to the Metal kernel it launches.
- **Data:** each input and output with shape, dtype and layout. Where it is produced, and where it is consumed next.
- **Neighbours:** what runs just before and after it in the decode step, and what a replacement must keep identical.
- **Cost:** the kernels it launches, the bytes it reads and writes (weight bytes counted per read), and its measured time with the table the time comes from.
- **Invariants:** what correctness depends on: the G2 tolerances, the tie rule for the 4th and 5th expert, the bfloat16 rounding points.

## How the session works

- **One GPU job at a time, run by the lead.** The grader is the only GPU job and the lead its only user; helpers work in parallel on reading, drafting and reviewing. The grader holds a lock, and a second run started while one is going stops with a message. Overlapping runs would distort the timings.
- **G3 and vLLM belong to surucuonur's team**, who run them after merging your branch.
- **Your writes go to `muse-spark/`.** Everything else in the worktree is read-only to you, and a commit outside `muse-spark/` is refused.
- **Credentials stay unread:** `.env` files, `~/.config/muse/` and any key store.
- **Your kernels are your own.** The API-stage kernels in the history before the tag `agent-era` are from an earlier, removed stage (PROVENANCE.md) and stay unread.
- **Every number has a source:** a table, a grader verdict, or arithmetic you show.

## Commits

Commit only on `muse-agent`, with explicit paths, a subject starting `Muse Spark:`, and this exact trailer as the last line:

```bash
git add muse-spark/<paths>
git commit -m "Muse Spark: <what this commit holds>" \
  -m "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
```

Use no other co-author. The hooks refuse a commit without this trailer, a commit outside `muse-spark/`, and any merge. Push, merge, rebase, git config and hook skipping stay with surucuonur.
