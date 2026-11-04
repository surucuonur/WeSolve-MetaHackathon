// hyp-2-routing-exact hardening, round-2 attempt-3 (builder-2-c).
// Builds on round-2 attempt-1 (the only full-G2 attempt): keeps its exact
// regime branches, partition boundaries and sequential partial order, its
// GEMV shuffle tree, verbatim sigmoid, top-4 tie rule and weights sum.
// ONE main change: the simdgroup-MMA threadgroup staging (48 chunks x 2
// barriers/row, 8x8 tiles with 7/8 waste) is replaced by scalar partition
// accumulation with no per-chunk barriers. Attempt-2 proved scalar sums
// round identically to MMA over the same K order at decode; this applies
// that inside attempt-1's full six-size regime logic.
//   rows == 1      GEMV, verbatim from attempt-1: per expert, 32 lanes hold
//                  d=i*128+lane*4+0..3 (i=0..5) serial partials, shuffle
//                  tree 16,8,4,2,1.
//   1 < rows <=768 steel split-K geometry, verbatim from attempt-1:
//                  partitions from next_pow2(48/((rows+31)/32*2)) clamped
//                  to [2,32] (8 partitions at 127 rows), base 48/partitions
//                  BK=16 chunks, last partition to chunk 48. Each
//                  (expert, partition) partial is one thread's serial
//                  scalar sum over its K range in increasing d; partials
//                  are then summed sequentially in partition order.
//   rows > 768     regular steel: partitions=1, one serial scalar
//                  accumulator over all 48 BK=16 chunks in order.
// One threadgroup (256 threads) per row; grid rows*256. Only two barriers
// on the scalar path: after staging x, and after writing partials (plus
// the shared barriers around the sigmoid/top-4 phase, as in attempt-1).
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int rows = x_shape[0];

threadgroup float xsh[768];
threadgroup float scores[60];
threadgroup float parts[1920]; // 60 experts x up to 32 partitions

for (uint i = tid; i < 768u; i += 256u) {
  xsh[i] = float(x[row * 768u + i]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (rows == 1) {
  for (uint e = sg; e < 60u; e += 8u) {
    float acc = 0.0f;
    for (uint i = 0u; i < 6u; ++i) {
      uint base = i * 128u + lane * 4u;
      uint wb = e * 768u + base;
      acc += xsh[base + 0u] * w_router[wb + 0u];
      acc += xsh[base + 1u] * w_router[wb + 1u];
      acc += xsh[base + 2u] * w_router[wb + 2u];
      acc += xsh[base + 3u] * w_router[wb + 3u];
    }
    acc += simd_shuffle_down(acc, 16);
    acc += simd_shuffle_down(acc, 8);
    acc += simd_shuffle_down(acc, 4);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 1);
    if (lane == 0u) {
      scores[e] = acc;
    }
  }
} else {
  uint partitions = 1u;
  uint base_iters = 48u;
  if (rows <= 768) {
    uint tm = (uint(rows) + 31u) / 32u;
    uint ratio = 48u / (tm * 2u);
    uint np = 1u;
    while (np < ratio) {
      np *= 2u;
    }
    if (np < 2u) {
      np = 2u;
    }
    partitions = np > 32u ? 32u : np;
    base_iters = 48u / partitions;
    if (base_iters == 0u) {
      base_iters = 1u;
    }
  }
  uint total_pairs = 60u * partitions;
  for (uint idx = tid; idx < total_pairs; idx += 256u) {
    uint e = idx / partitions;
    uint p = idx - e * partitions;
    uint c0 = p * base_iters;
    uint c1 = (p + 1u == partitions) ? 48u : (p + 1u) * base_iters;
    float part = 0.0f;
    for (uint d = c0 * 16u; d < c1 * 16u; ++d) {
      part += xsh[d] * w_router[e * 768u + d];
    }
    parts[e * 32u + p] = part;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < 60u; e += 256u) {
    float acc = 0.0f;
    for (uint p = 0u; p < partitions; ++p) {
      acc += parts[e * 32u + p];
    }
    scores[e] = acc;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// MLX Sigmoid verbatim (unary_ops.h): y on |x|, branch on sign, float32.
for (uint e = tid; e < 60u; e += 256u) {
  float s = scores[e];
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
  scores[e] = (s < 0.0f) ? y : 1.0f - y;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
  float topv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint topi[4] = {0, 0, 0, 0};
  for (uint e = 0u; e < 60u; ++e) {
    float b = scores[e] + bias[e];
    if (b > topv[0]) {
      topv[3] = topv[2]; topi[3] = topi[2];
      topv[2] = topv[1]; topi[2] = topi[1];
      topv[1] = topv[0]; topi[1] = topi[0];
      topv[0] = b; topi[0] = e;
    } else if (b > topv[1]) {
      topv[3] = topv[2]; topi[3] = topi[2];
      topv[2] = topv[1]; topi[2] = topi[1];
      topv[1] = b; topi[1] = e;
    } else if (b > topv[2]) {
      topv[3] = topv[2]; topi[3] = topi[2];
      topv[2] = b; topi[2] = e;
    } else if (b > topv[3]) {
      topv[3] = b; topi[3] = e;
    }
  }
  float s0 = scores[topi[0]];
  float s1 = scores[topi[1]];
  float s2 = scores[topi[2]];
  float s3 = scores[topi[3]];
  float denom = s0 + s1 + s2 + s3 + 1e-20f;
  uint obase = row * 4u;
  inds[obase + 0] = topi[0];
  inds[obase + 1] = topi[1];
  inds[obase + 2] = topi[2];
  inds[obase + 3] = topi[3];
  weights[obase + 0] = (s0 / denom) * 2.5f;
  weights[obase + 1] = (s1 / denom) * 2.5f;
  weights[obase + 2] = (s2 / denom) * 2.5f;
  weights[obase + 3] = (s3 / denom) * 2.5f;
}
