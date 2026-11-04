// Fused combine with contraction-free exact sum: thread-per-output kernel,
// identical to the starting point
// (muse-spark/round-2/kernels/combine/attempt-1/block-0.metal) except the
// seven `volatile float` rounding barriers are replaced by
// `__attribute__((noinline))` device-function barriers (combine_mul /
// combine_add, block-1.metal, via header_block). Grid is (rows*768, 1, 1),
// threadgroup (256, 1, 1), no threadgroup memory.
//
// Reference (slots.py:51-52): (y * weights[..., None]).sum(axis=-2)
//   .astype(bf16) + shared. y is [rows,4,768] bf16 row-contiguous, weights
//   [rows,4] f32, shared [rows,768] bf16, out [rows,768] bf16.
//
// Exact sequence reproduced (hyp-1-combine-devolatilize Boundary, delta to
// round 2's hyp-3-combine-exact Boundary):
//   1. cast y bf16->f32 exactly;
//   2. each product p_j = float(y_j) * weights_j rounded once to f32;
//   3. fold ((p0+p1)+p2)+p3 with a rounding after every add;
//   4. round the sum once to bfloat16;
//   5. add shared in bfloat16 with a single rounding.
// Steps 2-3 are forced by the noinline call barriers: each call returns
// one f32-rounded value in a register, so the compiler cannot keep a
// product unrounded into the next add (explicit rounding barriers,
// fma-free), with no thread-private memory traffic. The final
// bfloat(acc) + shared line is verbatim from the starting point.

uint idx = thread_position_in_grid.x;
uint row = idx / 768u;
uint d = idx - row * 768u;

uint wbase = row * 4u;
uint ybase = row * 3072u + d; // 4*768
uint sbase = row * 768u + d;

float p0 = combine_mul(float(y[ybase]), weights[wbase]);
float p1 = combine_mul(float(y[ybase + 768u]), weights[wbase + 1u]);
float p2 = combine_mul(float(y[ybase + 1536u]), weights[wbase + 2u]);
float p3 = combine_mul(float(y[ybase + 2304u]), weights[wbase + 3u]);
float s01 = combine_add(p0, p1);
float s012 = combine_add(s01, p2);
float acc = combine_add(s012, p3);

out[sbase] = bfloat(acc) + shared[sbase];
