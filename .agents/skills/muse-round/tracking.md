# The swarm's shared documents, and the rules every agent keeps

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. Part of the [muse-round skill](SKILL.md). [Who made what](../../../PROVENANCE.md).*

Every agent of the round reads this file. The documents below are how the swarm shares what it knows and how surucuonur follows the round while it runs, so keep them current as you work, not at the end.

## The round folder and its two phase folders

`<round>/` is this round's folder: `muse-spark/round-<N>/` from round 2 on (round 1's is `muse-spark/` itself). The lead's prompt names it.

| Folder | Phase | Holds |
|---|---|---|
| `<round>/exploration/` | Exploration | the analysis: `README.md`, territory and inefficiency entries, `log.md` |
| `<round>/implementation/` | Implementation | the report: `README.md`, hypothesis entries, `log.md` |

Kernels go in `<round>/kernels/<slot-folder>/attempt-<N>/`, numbered from 1 in each round; the grader reads any attempt folder it is given.

**Earlier rounds' folders are frozen.** Read them and cite them by path; write what you add in this round's entries. The hooks refuse a change to any file a finished round left.

Each folder has three kinds of document:

- **`README.md`.** The phase's document. Only the lead writes it, refreshing it and raising its version at every checkpoint. It opens with its version line and the tracker table below.
- **Entries.** One file per territory, inefficiency or hypothesis. The owner keeps it current, and other agents append notes.
- **`log.md`.** One line per event, from every agent, appended only.

## Versions

Every shared document carries a version number, so a reader knows which version they're looking at, and git holds every version.

**Phase versions.** Each README opens with its version line:

```markdown
**Version:** v5 · 2026-11-04T09:40Z · skill version 6
```

- **Who raises it:** the lead, at every checkpoint, and each version is exactly one commit.
- **The commit:** its subject names the version, `<name>: exploration v5: <what changed>`, where `<name>` is the Muse agent's name from the lead's first prompt. `git log --grep 'exploration v5:'` finds it, and `git diff` between two such commits shows what changed.
- **The changelog:** the README ends with `## Versions`, one line per version, newest first, naming the entries that changed and their new versions:

```markdown
- v5 · 2026-11-04T09:40Z · split quantized-kernels in two · changed: territory-quantized-kernels v4, ineff-quantized-kernels-2 v2
```

**Entry versions.** Each entry's status block has a `Version` line, starting at 1:
- **Territory and inefficiency entries:** the owner raises it with every change to its own text. The verifier raises it when it writes the verdict.
- **Hypothesis entries:** the lead raises it at each checkpoint where the entry changed, builders' and the boundary agent's sections included.
- **Notes:** a note under `## From other agents` leaves the version alone. The owner raises it when folding the note in.

**Times.** Every time you write, in a version line, a status block, a tracker row, a note or a log line, is the output of `date -u +%FT%TZ` run at that moment. Never estimate or round one.

## The README trackers

Exploration, `exploration/README.md`:

| Territory | Agent | Status | Inefficiencies | Updated |
|---|---|---|---|---|
| `quantized-kernels` | `explorer-quantized-kernels` | exploring | `ineff-quantized-kernels-1` (confirmed), `-2` (open) | 2026-11-04 09:40Z |

Implementation, `implementation/README.md`:

| Hypothesis | Slot | Builders | Attempts: G1, G2, µs per layer at 1 / 8 / 32 rows (compiled) | Status |
|---|---|---|---|---|
| `hyp-1-fused-gather` | `experts` | `builder-1-a`, `builder-1-b` | 3: pass, pass, 41 / 55 / 180 (48 / 70 / 260) | hardening |

From round 2 on, where the slot has a baseline, the attempts column also gives its time from the same run, and whether `beats_baseline` is true.

## Entries

Each entry opens with a status block that its owner keeps current:

```markdown
- **Owner:** explorer-quantized-kernels
- **Status:** exploring
- **Version:** 3
- **Updated:** 2026-11-04T09:40Z
- **Next:** read the 32-row dispatch of affine_gather_qmm
```

Entries are named so that two agents never pick the same name:
- `<round>/exploration/territory-<territory>.md`
- `<round>/exploration/ineff-<territory>-<n>.md`, numbered by the explorer of that territory
- `<round>/implementation/hyp-<n>-<slug>.md`, numbered by the lead

**Another agent's entry:** append under its `## From other agents` heading, starting with your name and the time. The owner's text stays as written, so two agents never overwrite each other. The owner reads these notes and folds in what holds.

### Territory entry

1. Status block (statuses: `exploring`, `done`).
2. **What it does** in the decode step, in a few lines.
3. **The checklist** below, filled in for each piece of code or table in the territory.
4. **Relations:** which territories feed it and which it feeds, with the data passed.
5. **Inefficiencies:** links to the entries this territory raised.
6. **Files read**, and what each one gave.

### Inefficiency entry

1. Status block. The statuses are `open`, `confirmed`, `refuted` and `superseded by <entry>`; only the verifier sets `confirmed` or `refuted`.
2. **Slot:** `routing`, `combine`, `experts`, or `outside the slots`.
3. **What happens:** what the code does, and where (file and lines).
4. **Cost** at 1, 8 and 32 rows of a decode step and at the prefill's 127, 1,016 and 2,048 rows: kernels × the fixed cost per kernel, bytes ÷ peak bandwidth, and the measured time with the table it comes from.
5. **Kind:** overhead-bound, bandwidth-bound or compute-bound, in the terms of `claude/GLOSSARY.md`.
6. **Spans:** the territories involved.
7. **Predicted saving** if removed, in µs per decode step at 1, 8 and 32 rows and per prefill chunk at 127, 1,016 and 2,048 rows, with the arithmetic. A saving in one phase that costs the other is not one.
8. **Verdict**, written by the verifier: each claim checked, and the result.

### Hypothesis entry

1. Status block. The lead owns it; the statuses are `boundary`, `building`, `hardening`, `ready` and `dropped`.
2. **Inefficiency** it removes, the slot, and from round 2 on the slot's baseline folder, or `none` and its starting point.
3. **Boundary**, written by the boundary agent: what the solution must keep identical (see `boundary.md`).
4. **One section per builder:** its direction, its attempts (folder, predicted and measured µs per layer at the six sizes with each size's verdict, G1, G2, `beats_compiled`, and where the slot has a baseline its time and `beats_baseline`), and what each attempt taught.
5. **Boundary checks** of the attempts that passed G2.
6. **Decision:** the best attempt and why. When the status is `ready`, it states what G3 should show.

## The checklist

Fill it in for each piece of code you inspect, so that no piece is judged without its neighbours:

- **Callers:** who calls it, and with what shapes at 1, 8 and 32 rows of a decode step and in a prefill.
- **Callees:** what it calls, down to the Metal kernel it launches.
- **Data:** each input and output with shape, dtype and layout. Where it is produced, and where it is consumed next.
- **Neighbours:** what runs just before and after it in the decode step, and what a replacement must keep identical.
- **Cost:** the kernels it launches, the bytes it reads and writes (weight bytes counted per read), and its measured time with the table the time comes from.
- **Invariants:** what correctness depends on: G2's bit-exact rule at every size, the tie rule for the 4th and 5th expert, the bfloat16 rounding points, and which MLX kernel the reference uses at each size.

## The log

Append one line per event with the shell, so lines from several agents never mix. In round 2:

```bash
printf '%s | %s | %s\n' "$(date -u +%FT%TZ)" "explorer-baseline-experts" "posted ineff-baseline-experts-2: each weight byte read 4 times at 32 rows" >> muse-spark/round-2/exploration/log.md
```

Log when you start, claim something, post or confirm an entry, change direction (what from, what to, and which entry made you), grade an attempt (with its verdict), and finish.

## Adapting to the others' work

Before every new piece of work, and after every post:
1. Read the phase README's tracker and the newest 30 lines of the phase log (`tail -n 30`).
2. Then change direction when:
   - **someone else covers it:** another entry already covers the same code or cause, so append to that entry instead of starting a second one;
   - **its premise fell:** the claim your work rests on was refuted;
   - **something pays more:** an open item in your territory or task predicts a larger saving than what you're on, and nobody owns it.
3. When you change direction, log it with the reason, and write it in your own entry's status block.

Two claims on the same item go to the lead, who settles them.

## Rules for every agent

- **Writes go only to this round's folder, in the files your role names.** Earlier rounds' folders, everything else in the worktree, and everything outside it are read-only.
- **Only the lead commits.** Helpers never run `git add` or `git commit`.
- **The grader is the only GPU job.** Run it only as a builder, through the brief's command; it waits its turn when another run has the GPU. Load no model and start no other GPU work: the machine has 16 GB and slows down sharply when memory runs short.
- **Every number has a source:** a file and line, a table (path and row), a grader verdict (attempt folder), or arithmetic shown in full.
- **Credentials stay unread:** `.env` files, `~/.config/muse/` and any key store.
- **The API-stage kernels** in the history before the tag `agent-era` are from an earlier, removed stage (PROVENANCE.md) and stay unread.
