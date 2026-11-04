# MobileMoE Bandwidth Study

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../PROVENANCE.md).*

How efficiently an inference engine turns an Apple Silicon GPU's memory bandwidth and compute into generated tokens when serving a small mixture-of-experts model.

## Hardware limits

**Peak bandwidth**:
The most bytes per second the GPU can read from unified memory, as rated by the manufacturer (about 200 GB/s on this project's M1 Pro).
_Avoid_: max bandwidth, memory speed

**Peak compute**:
The most floating-point operations per second the GPU can perform (about 5.2 TFLOPS on the 16-core M1 Pro GPU).
_Avoid_: max compute, TFLOPS (as the name of the limit)

**Achieved bandwidth**:
The bytes a decode step must read from memory (the weights it touches plus its share of cached attention state), divided by that step's duration.
_Avoid_: bandwidth usage, utilization (unqualified)

**Arithmetic intensity**:
Floating-point operations performed per byte read from memory during a decode step.
_Avoid_: compute density, FLOPs/byte ratio

**Ridge point**:
The arithmetic intensity at which peak bandwidth and peak compute meet (about 26 operations per byte here); below it a step is limited by bandwidth, above it by compute.

**Roofline**:
The chart of achievable throughput against arithmetic intensity, capped by peak bandwidth on the left and peak compute on the right, with measured decode steps plotted beneath it.

## Bottlenecks and fixes

**Bandwidth-bound**:
A decode step whose achieved bandwidth is close to peak bandwidth.

**Compute-bound**:
A decode step whose operation rate is close to peak compute.

**Overhead-bound**:
A decode step far below both limits, because its time goes to work other than reading weights or doing arithmetic, such as launching GPU jobs, routing tokens to experts or scheduling.
_Avoid_: latency-bound, CPU-bound (unless the CPU specifically is the cause)

**Bottleneck**:
The single cause, identified from measurements, that most limits throughput for a given workload.
_Avoid_: issue, slowness

**Kernel**:
One program the GPU runs as a unit, such as one matrix product or one element-wise add; a decode step is a chain of hundreds of them, each with a fixed start-up cost.
_Avoid_: GPU job, op (for a kernel launch)

**Occupancy**:
The share of the GPU's thread slots in use while it runs; low occupancy means most of the chip is idle even when the GPU counts as busy.
_Avoid_: GPU utilization (which only says the GPU had work)

**Knock-out**:
A measurement in which one piece of the decode step is replaced by a stand-in that returns the right shape without GPU work; the time and bytes the step saves are that piece's cost.
_Avoid_: ablation

**Baseline**:
The unmodified engine serving MobileMoE-S-QAT (int4), against which every fix is measured.

**Fix**:
One change to the engine, the model code or the configuration, measured against the baseline under the same workload.
_Avoid_: optimization (for a change that hasn't been measured), tweak

## Inference

**Prefill**:
Processing a request's whole prompt through the model before its first token is generated, in chunks of up to the engine's token budget per pass (2,048 tokens here).
_Avoid_: prompt phase (say prefill, or prompt processing)

**Decode step**:
One engine iteration that produces one new token for each request it includes.

**Decode rows**:
The number of requests the engine actually advanced in one decode step; the batch size that matters for the roofline.
_Avoid_: batch size (unqualified), concurrency

**Client concurrency**:
The number of requests the benchmark client keeps in flight; it caps decode rows but does not equal them.
_Avoid_: batch size

## Mixture of experts

**Expert**:
One of the interchangeable feed-forward blocks in a layer; MobileMoE-S has 60 routed experts per layer.

**Active experts**:
The experts the router picks for one token in one layer (4 for MobileMoE-S).
_Avoid_: selected experts, top-k experts

**Distinct experts**:
The number of different experts used by all decode rows together in one layer during one decode step; it decides how much expert weight data the step must read.
_Avoid_: unique experts, expert coverage

## Models

**MobileMoE-S**:
Meta's smallest on-device mixture-of-experts model (about 1.3B parameters in total, about 272M used per token), studied here in its SFT (full-precision) and QAT (int4) variants.

**Muse Spark**:
Meta's hosted model, and the Muse agent in round 1.

**Muse Wonder**:
Meta's most capable model family, in preview; its standard variant runs on Meta's Model API. The Muse agent in round 2.

**Muse Wonder Full**:
The higher-capability variant of Muse Wonder, available only in Muse Code with a subscription; not used in this project.

**Muse agent**:
The Meta model that chooses which pieces of the decode step to replace and writes their kernels, on the branch `muse-agent`: Muse Spark in rounds 1 and 3, Muse Wonder in rounds 2 and 4. It is never the model under study.
_Avoid_: Muse (alone), Meta agent

**Muse Code**:
Meta's own coding agent for the terminal; the only program in which the Muse agent does its work in this project.
_Avoid_: Muse CLI, Meta Code

**Muse agent session**:
A Muse Code session in which the Muse agent writes every turn, the lead's and every helper's, kept apart from Claude's work by its own branch, folder and commit trailer.
_Avoid_: Muse Spark session (round 2's model was Muse Wonder), guardrail (for the session itself), Meta mode

## The Muse agent's work

**Slot**:
One replaceable piece of the mixture-of-experts block (routing, combine, or the routed experts), with fixed inputs and outputs, that the Muse agent may replace with its own kernels.
_Avoid_: target (that is a slot the Muse agent has chosen), module

**Round**:
One Muse agent session in which a swarm explores and then builds, writing in its own round folder, followed by Claude's independent measurement of what it produced. Once a round is finished, its files are frozen.
_Avoid_: iteration, cycle

**Round folder**:
Where one round writes all its files: round 1 at the top of `muse-spark/`, each later round in `muse-spark/round-<N>/`.

**Starting point**:
The plan each round from round 2 on begins with, drawn from the earlier rounds and Claude's measurement: each slot's baseline, or the attempt to start from where it has none, with its end-to-end effect; what is left to build; the refuted claims; and the territories to explore again.
_Avoid_: recap, retrospective

**Improvement**:
An attempt that is faster than its bar in one phase, prefill or decode, and slower in neither; a faster decode step bought with a slower prefill is not one.
_Avoid_: speedup (for a change measured in one phase only)

**Hand-off**:
An attempt's choice to run the compiled reference instead of its own kernels outside a range of row counts it names: above it, so prefill-sized inputs stay on MLX's code, and below it, so the smallest decode steps do. A size handed off is not timed against the compiled reference, since both run the same code.
_Avoid_: fallback

**Ready check**:
The lead's in-round run of G3 over every candidate attempt at once, with its protocol fixed and its own log audited; only its passing verdict makes an attempt ready.
_Avoid_: G3 (that is Claude's run after the round), end-to-end test

**Set**:
Candidates in different slots that pass the ready check together, and so are ready and go to G3 together, even where one fails alone.
_Avoid_: pair, combination (for the ready ones)

**Candidate**:
An attempt that passes the grader's gates and its boundary check, with its claims checked, and waits for the ready check.
_Avoid_: ready (that needs the ready check)

**Prediction**:
A speed number no one measured for exactly what it describes, labelled as such, counting only the candidate's own measured effect and shown beside the measured value once there is one.
_Avoid_: expected gain, estimate (unlabelled)

**Unverified**:
What no one measured, listed as such in a hypothesis and in the round's final claims, and never assumed free, neutral or zero.

**Baseline** (of a slot):
A slot's best attempt so far that passes the current G2 and that Claude's G3 accepted, named in Claude's measurement of the previous round; every new attempt for that slot must beat it in the same grader run. A slot may have none.
_Avoid_: reference (that is the original MLX code and its compiled version)

**Swarm**:
All the agents of one round, the lead and its helpers, working toward one goal through shared documents that each of them reads and adds to.
_Avoid_: team, crew

**Lead agent**:
The one agent of a swarm that starts the helpers, keeps the shared documents current, settles clashes and commits; it neither explores nor grades.
_Avoid_: orchestrator, main agent

**Helper agent**:
An agent the lead starts: an explorer, the verifier, a builder, a boundary agent or a checker. It reports back and never commits.
_Avoid_: worker, subagent (when the role matters, not the mechanism)

**Reminder agent**:
One of Muse Code's own background agents, which watch the lead's progress and may nudge it, for example to check a claim before it says it is done; the Muse agent's model writes their turns too, but they take no part in the work.
_Avoid_: helper (that is one the lead starts)

**Exploration phase**:
The first part of a round, in which explorers map the code and the measurements and find where the decode step loses time.

**Implementation phase**:
The second part of a round, in which builders write and grade solutions to the confirmed inefficiencies.
_Avoid_: hypothesis phase

**Territory**:
The part of the code or the measurements that one explorer covers: a module, a single important file, or a measurement document.
_Avoid_: area, lens

**Analysis**:
The swarm's shared account of the code, written during exploration: the context, how the territories feed each other, and the ranked inefficiencies, each with its cost and sources.
_Avoid_: diagnosis, findings (those are Claude's)

**Inefficiency**:
One cause of lost time in the decode step: where it is in the code, what it costs, and what removing it should save. It is confirmed or refuted against its sources before anyone builds on it.
_Avoid_: issue, bottleneck (that is the single largest cause)

**Hypothesis**:
A proposed solution to one inefficiency: the slot it replaces, the boundary it keeps and the saving it predicts, tested by competing attempts.
_Avoid_: strategy, plan

**Boundary**:
What a solution must keep identical so that the steps before and after its slot work exactly as before: the slot's inputs and outputs, their dtypes, and the rounding points.
_Avoid_: interface (unqualified), contract

**Attempt**:
One grader run of a solution a builder wrote, committed by the lead; attempts for a slot are numbered in order within each round and have no limit.
_Avoid_: try, submission, draft
