2026-11-04T21:41:22Z | verifier | started; waiting for explorers to post ineff entries
2026-11-04T21:41:38Z | explorer-baseline-combine | started: mapping round-3 combine attempt-2 (pragma devolatilized) as code in slot — pragma binary, decode margin, chained floor
2026-11-04T21:42:08Z | explorer-baseline-experts | started territory baseline-experts: round-3 experts attempt-1 under min_rows 2
2026-11-04T21:42:23Z | explorer-grader | started: mapping round-4 grader delta (ADR 0015 min_rows, handed off, set rule)
2026-11-04T21:42:24Z | explorer-baseline-routing | started: mapping round-3 routing attempt-6 (staged scalar, max_rows 32) as code in slot — decode margin at 32x1024, per-row w_router re-read, staged scalar costs at decode
2026-11-04T21:42:38Z | explorer-g3 | started: mapping Claude round-3 G3 (pair at 32x1024, min_rows set prediction, three-slot interaction)
2026-11-04T21:43:26Z | explorer-baseline-routing | opened territory-baseline-routing v1: attempt-6 mapped, decode margins 17.7/15.8/19.7 us/layer (hand-off re-grade), G3-alone is why no baseline
2026-11-04T21:43:42Z | explorer-baseline-experts | posted ineff-baseline-experts-1: per-pair weight re-reads at 8/32 rows (552,960 B/pair, no cross-pair sharing)
2026-11-04T21:43:42Z | explorer-baseline-experts | posted ineff-baseline-experts-2: serial K-chain + idle simdgroups hold 8/32 rows 80-201 us/layer above own byte floor
2026-11-04T21:43:44Z | explorer-baseline-routing | posted ineff-baseline-routing-1: per-row w_router re-read at decode, excess 0/6.5/28.6 us/layer vs margins 17.7/15.8/19.7, upper bound 0/0.13/0.57 ms/step overlap-discounted
2026-11-04T21:43:47Z | explorer-baseline-experts | finished territory baseline-experts (done): min_rows 2 window 2-59, 1-row handed off, 33-59 exact-but-untimed, scalar branch still dead; 2 ineff entries open
2026-11-04T21:43:58Z | explorer-g3 | posted ineff-g3-1: pair keeps -0.36 of -0.66 ms/step piece win vs compiled at 32x1024, ranges overlap, verdict flips
2026-11-04T21:43:58Z | explorer-g3 | posted ineff-g3-2: min_rows set meets untested three-slot interaction +2.13..2.86 ms/step at 8 requests
2026-11-04T21:44:04Z | explorer-g3 | finished territory g3 (done): pair 32x1024 gap, min_rows set prediction at 4 workloads, three-slot interaction carried
2026-11-04T21:44:27Z | explorer-baseline-routing | posted ineff-baseline-routing-2: staged scalar shape at decode (27 barriers, 23%-duty tail third, dead parts-zero at 1 row, serial tails), saving bounded single-digit us/layer
2026-11-04T21:44:47Z | explorer-grader | finished territory grader (done): ADR 0015 boundary, min_rows 2/max_rows 59 six-size grading, set-rule reading; 0 ineff entries (grader runs no decode kernels); notes to baseline-experts, g3, baseline-routing
2026-11-04T21:44:49Z | explorer-baseline-routing | done: territory-baseline-routing v2 (grader note folded), ineff-baseline-routing-1..2 open; decode margin pools priced for 32x1024 set
2026-11-04T21:45:18Z | explorer-baseline-combine | posted ineff-baseline-combine-1: chained floor ~32 us/layer decode (bytes+compute <=1.5), achievable saving 0 within slot
2026-11-04T21:45:18Z | explorer-baseline-combine | posted ineff-baseline-combine-2: idx/768 divide + 768x weight broadcast loads bounded <=0.5 us/layer decode, below paired noise
2026-11-04T21:45:29Z | explorer-baseline-combine | done: territory-baseline-combine v2, ineff-baseline-combine-1..2 open; pragma leaves no FMA effect (AIR unverified, no toolchain), no bit-exact decode-margin widening, floor priced
2026-11-04T21:48:24Z | verifier | verdict ineff-baseline-experts-1: refuted — 32-row re-read mixes pair-model 70.78 MB/layer with counter distinct 22.15, floor is 2.33 ms/step counter (23.25 MB/layer) not 4.86; restate on one basis
2026-11-04T21:48:24Z | verifier | verdict ineff-baseline-experts-2: confirmed — chain, floors 80.4/200.8 us/layer above pair-byte floor, headroom 1.61/4.02 ms/step, 0 banked
2026-11-04T21:48:24Z | verifier | verdict ineff-baseline-routing-1: confirmed — per-row w_router excess 0/6.5/28.6 us/layer vs margins 17.7/15.8/19.7, upper bound 0/0.13/0.57 ms/step overlap-discounted
2026-11-04T21:48:24Z | verifier | verdict ineff-baseline-routing-2: confirmed — 27 barriers, 23%-duty tail third, residuals and single-digit us/layer bound, prediction only
2026-11-04T21:48:24Z | verifier | verdict ineff-baseline-combine-1: confirmed — chained floor ~32 us/layer decode, bytes+compute <=1.5 us, achievable saving 0
2026-11-04T21:48:24Z | verifier | verdict ineff-baseline-combine-2: confirmed — divide + broadcast loads <=0.5 us/layer decode, below paired noise, do not chase
2026-11-04T21:48:24Z | verifier | verdict ineff-g3-1: confirmed — pair keeps -0.36 of -0.66 ms/step piece win at 32x1024, 0.9% gap inside spread, needs ~0.17 ms/step more
2026-11-04T21:48:24Z | verifier | verdict ineff-g3-2: confirmed — three-slot interaction +2.13..2.86 ms/step at 8 requests vs -2.16 additive prediction, ready check is first test
2026-11-04T21:48:24Z | verifier | duplicates checked: none — experts bytes vs chain, routing bytes vs shape, combine floor vs body, g3 pair-margin vs triple-interaction are distinct causes (overlaps flagged do-not-add in entries)
2026-11-04T21:48:24Z | verifier | done: all 5 explorers finished (territories done), 8 ineff entries checked — 7 confirmed, 1 refuted, 0 open
