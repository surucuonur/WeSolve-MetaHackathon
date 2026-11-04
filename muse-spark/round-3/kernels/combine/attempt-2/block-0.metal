// Fused combine with contraction-free exact sum: thread-per-output kernel
// identical to the starting point
// (muse-spark/round-2/kernels/combine/attempt-1) in grid, indexing,
// threadgroup and final line. The only change is the barrier mechanism:
// the seven `volatile float` rounding barriers are seven plain `float`
// locals, and FP contraction is disabled by the header pragmas
// (block-1.metal) instead of by forced memory traffic. Grid is
// (rows*768, 1, 1), threadgroup (256, 1, 1), no threadgroup memory.
//
// Reference (slots.py:51-52): (y * weights[..., None]).sum(axis=-2)
//   .astype(bf16) + shared. y is [rows,4,768] bf16 row-contiguous, weights
//   [rows,4] f32, shared [rows,768] bf16, out [rows,768] bf16.
//
// Exact sequence reproduced (hyp-1-combine-devolatilize Boundary):
//   1. cast y bf16->f32 exactly;
//   2. each product p_j = float(y_j) * weights_j rounded once to f32;
//   3. fold ((p0+p1)+p2)+p3 with a rounding after every add;
//   4. round the sum once to bfloat16;
//   5. add shared in bfloat16 with a single rounding.
// Steps 2-3 are exact because the header's contract(off) pragmas forbid
// the backend from fusing any mul into its consumer add; each statement
// below is then one rounded f32 op on register values, with no FMA, no
// reassociation and no threadgroup memory. The final
// bfloat(acc) + shared line is verbatim from the starting point.

uint idx = thread_position_in_grid.x;
uint row = idx / 768u;
uint d = idx - row * 768u;

uint wbase = row * 4u;
uint ybase = row * 3072u + d; // 4*768
uint sbase = row * 768u + d;

float p0 = float(y[ybase]) * weights[wbase];
float p1 = float(y[ybase + 768u]) * weights[wbase + 1u];
float p2 = float(y[ybase + 1536u]) * weights[wbase + 2u];
float p3 = float(y[ybase + 2304u]) * weights[wbase + 3u];
float s01 = p0 + p1;
float s012 = s01 + p2;
float acc = s012 + p3;

out[sbase] = bfloat(acc) + shared[sbase];
