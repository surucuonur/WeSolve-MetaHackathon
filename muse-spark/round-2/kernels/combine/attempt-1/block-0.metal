// Fused combine with contraction-free exact sum: thread-per-output kernel
// identical to the starting point (muse-spark/kernels/combine/attempt-1)
// except the 4-term sum is evaluated with precise stepwise rounding so no
// mul+add is FP-contracted into an FMA. Grid is (rows*768, 1, 1),
// threadgroup (256, 1, 1), no threadgroup memory.
//
// Reference (slots.py:51-52): (y * weights[..., None]).sum(axis=-2)
//   .astype(bf16) + shared. y is [rows,4,768] bf16 row-contiguous, weights
//   [rows,4] f32, shared [rows,768] bf16, out [rows,768] bf16.
//
// Exact sequence reproduced (hyp-3-combine-exact Boundary):
//   1. cast y bf16->f32 exactly;
//   2. each product p_j = float(y_j) * weights_j rounded once to f32;
//   3. fold ((p0+p1)+p2)+p3 with a rounding after every add;
//   4. round the sum once to bfloat16;
//   5. add shared in bfloat16 with a single rounding.
// Steps 2-3 are forced by volatile thread variables: each volatile store
// materialises its f32 value, so the compiler cannot keep a product
// unrounded into the next add (explicit rounding barriers, fma-free).
// The Meta qualifier spelling `precise float` does not compile under
// MLX's header (metal::precise is a function namespace there), so
// volatile stepwise barriers provide the precise semantics. The final
// bfloat(acc) + shared line is verbatim from the starting point.

uint idx = thread_position_in_grid.x;
uint row = idx / 768u;
uint d = idx - row * 768u;

uint wbase = row * 4u;
uint ybase = row * 3072u + d; // 4*768
uint sbase = row * 768u + d;

volatile float p0 = float(y[ybase]) * weights[wbase];
volatile float p1 = float(y[ybase + 768u]) * weights[wbase + 1u];
volatile float p2 = float(y[ybase + 1536u]) * weights[wbase + 2u];
volatile float p3 = float(y[ybase + 2304u]) * weights[wbase + 3u];
volatile float s01 = p0 + p1;
volatile float s012 = s01 + p2;
volatile float acc = s012 + p3;

out[sbase] = bfloat(acc) + shared[sbase];
