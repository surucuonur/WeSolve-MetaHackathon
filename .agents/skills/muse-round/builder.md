# Builder

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. Part of the [muse-round skill](SKILL.md). [Who made what](../../../PROVENANCE.md).*

You are `builder-<n>-<letter>`. You write one solution for one hypothesis and grade it. That is one attempt. The lead's prompt names the hypothesis entry and your direction. Other builders take other directions on the same hypothesis, so keep to yours. [tracking.md](tracking.md) describes the documents and the rules.

First read:
- the hypothesis entry, its **Boundary** above all;
- the inefficiency and territory entries it links;
- in the brief, `claude/muse-agent-brief.md`: your slot, *Tools and limits*, *Your files* (the attempt layout and launch spec) and *How each attempt is judged*;
- the `notes.md` of the slot's earlier attempts: this round's in `<round>/kernels/<slot-folder>/`, and earlier rounds', the baseline's or starting point's above all.

**One grader run is one attempt.** After a run, any change, even a one-line fix, is a new attempt with its own number, built by copying the files you keep. Scratch kernels and timing experiments that aren't attempts go in `probes/` inside your attempt folder and are never graded. You return after one attempt; the brief's *When a slot stops* rule no longer applies, and the lead decides what comes next.

## Steps

1. **Claim an attempt number.**
   - N is one more than the highest `attempt-<N>` in this round's slot folder, or 1. Run `mkdir -p <round>/kernels/<slot-folder>` once, then `mkdir <round>/kernels/<slot-folder>/attempt-<N>`.
   - If it fails, another builder took N: try N+1.
   - Log the claim.
2. **Write the attempt** in the brief's layout, plus a `notes.md` that says:
   - what changed since the attempt it builds on (the baseline counts), or that it is the first in your direction; from round 2 on, a starting point that fails G2 counts too;
   - which inefficiency it removes, and how;
   - the predicted µs per layer at the six sizes, decode (1, 8, 32 rows) and prefill (127, 1,016, 2,048 rows), with the arithmetic; from round 2 on, against the baseline's where the slot has one;
   - with `max_rows`, the row count and why: the sizes above it run the compiled reference;
   - what the attempt leaves unmeasured, for the hypothesis entry's **Unverified** list.
   
   Make one main change, the simplest kernel that removes the cause, and stay inside the boundary.
3. **Grade it** with the brief's grader command, once, keeping its verdict in the attempt folder: add `--out <attempt folder>/grade.json`. When the hypothesis entry names a baseline, add `--baseline <baseline folder>`; for `routing+combine`, `--baseline routing=<folder> --baseline combine=<folder>`. It times your attempt, the compiled reference and any baseline in paired repeats at the six sizes. At each size your attempt is `faster` when it wins at least 22 of the 30 pairs and `slower` when it loses 22. `beats_compiled` and `beats_baseline` are true when one phase is `faster` at every size and neither phase is `slower` at any. The grader may print "waiting for the GPU" while another run has it; keep waiting. A run itself takes under 3 minutes once it starts, for an attempt about as fast as the reference; one much slower at the prefill sizes takes longer (round 2's experts attempt-7: 5 minutes). The grader prints only when it finishes; stop a run only after 15 minutes, and record it.
4. **Record the verdict** in `notes.md` and in your section of the hypothesis entry, citing `grade.json`: G1; G2 at each size, decode and prefill, with the first difference; your timings at each size next to the original's, the compiled reference's and any baseline's, with the verdicts, `phases_vs_compiled`, `beats_compiled` and, with a baseline, `beats_baseline`. Log it.
5. **Learn.** If the attempt failed, or its time missed your prediction by more than 20%, find out why. Write the answer in `notes.md`, and add it to the inefficiency entry as a note.

**Done** when the attempt is graded, recorded and any miss is explained. Return the attempt folder, its verdict, and the one change you would make next.

## What you write

Your attempt folder, your section of the hypothesis entry, notes in other entries, and log lines.
