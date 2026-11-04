# Inefficiency: the kernel's sigmoid rounds differently from MLX's at every score

- **Owner:** explorer-baseline-routing
- **Status:** open
- **Version:** 1
- **Updated:** 2026-11-04T03:19:36Z
- **Next:** verifier to confirm against installed `unary_ops.h:308-314`

## Slot

`routing`

## What happens

The starting point computes `scores[e] = 1.0f / (1.0f + metal::exp(-acc))`
(`muse-spark/kernels/routing/attempt-1/block-0.metal:35`): the fast
(non-`precise`) Metal `exp` of `-acc`, then one division, for all signs.
The reference's `mx.sigmoid` (`bench/grader/slots.py:45`) is MLX's unary
`Sigmoid` (`mlx/include/mlx/backend/metal/kernels/unary_ops.h:308-314`,
installed MLX 0.32.4):

```metal
auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
return (x < 0) ? y : 1 - y;
```

Three rounding differences, each independently able to move a score by
1 ULP: (a) `metal::precise::exp` vs `metal::exp`; (b) the exponent argument
is `|x|`, never negative; (c) for `x >= 0` the result is the subtraction
`1 - y` rather than a direct division. With 60 scores per row and 4 selected,
even a bit-exact matmul behind this sigmoid would still leave most rows
differing — this cause is independent of `ineff-baseline-routing-1`, and
both must be fixed for G2. (That the 4-term weight sum is *not* a third
cause is shown in `territory-baseline-routing.md` section 3.3.)

## Cost

Correctness only; no time cost. It contributes to the same measured failure
as ineff-baseline-routing-1: rows differing 20/159/638 (decode) and
2,518/16,187/32,743 (prefill), max relative error up to 8.7e-06
(`results/runs/2026-11-04-muse-round-1/grade-exact-routing.json`). The split
of blame between this entry and the matmul order cannot be measured without
a candidate that fixes one of them; the grader's worst-row examples show
individual selected weights alternately equal and 1-ULP-off within one row
(e.g. prefill 127, layer 0 row 0: three weights equal, the fourth off by
3e-08 relative), consistent with per-score last-bit flips from either cause.

## Kind

Overhead-bound slot blocked by a correctness (rounding-formula) cause.

## Spans

`baseline-routing`

## Predicted saving

Zero µs directly — copying MLX's formula verbatim is the same instruction
count class (one `exp`, one division, one conditional subtract). Its value
is that it is a *necessary* half of unlocking the fused-vs-compiled headroom
quantified in `ineff-baseline-routing-1` (0.12/0.37/0.37 ms per step at
1/8/32 rows on the independent timing): with the matmul fixed and this left
in place, G2 still fails at essentially every row and the saving stays 0.

## Verdict

(empty — verifier writes here)

## From other agents

(none yet)
