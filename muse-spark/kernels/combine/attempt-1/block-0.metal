// Fused combine: weight the 4 experts' bf16 outputs by fp32 weights, sum in
// fp32, round once to bf16, add shared — one kernel launch per layer, one
// thread per output element. Grid is (rows*768, 1, 1), threadgroup (256, 1, 1).
//
// Reference (slots.py:51-52): (y * weights[..., None]).sum(axis=-2)
//   .astype(bf16) + shared. y is [rows,4,768] bf16 row-contiguous, weights
//   [rows,4] f32, shared [rows,768] bf16, out [rows,768] bf16. The bfloat
//   conversion and the + shared use the same single-rounding ops as MLX's
//   copy_v/binary_vv kernels, and the 4-term fp32 sum order is free per the
//   hyp-3 boundary.

uint idx = thread_position_in_grid.x;
uint row = idx / 768u;
uint d = idx - row * 768u;

uint wbase = row * 4u;
uint ybase = row * 3072u + d; // 4*768
uint sbase = row * 768u + d;

float acc = float(y[ybase]) * weights[wbase]
    + float(y[ybase + 768u]) * weights[wbase + 1u]
    + float(y[ybase + 1536u]) * weights[wbase + 2u]
    + float(y[ybase + 2304u]) * weights[wbase + 3u];

out[sbase] = bfloat(acc) + shared[sbase];
