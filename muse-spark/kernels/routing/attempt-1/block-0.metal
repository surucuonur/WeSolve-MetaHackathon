// Fused routing: router scores (matmul + sigmoid + bias), top-4 select, weight
// normalize/scale — one kernel launch per layer, one threadgroup (32 threads)
// per row. Grid is (rows*32, 1, 1), threadgroup (32, 1, 1).
//
// Per row: 60 experts, thread t computes experts t and (t+32 if t<28).
// x (768 bf16) is staged once into threadgroup memory; w_router rows are read
// straight from device memory. Thread 0 then does the serial top-4 scan over
// the 60 threadgroup scores, normalizes, and writes inds/weights.

uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;

threadgroup float xsh[768];
threadgroup float scores[60];

// Stage x[row] (bfloat -> float) cooperatively: 32 threads x 24 elements.
uint xbase = row * 768;
for (uint i = 0; i < 24; i++) {
  uint d = tid * 24 + i;
  xsh[d] = float(x[xbase + d]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Two experts per thread (threads 0..27), one for threads 28..31.
for (uint k = 0; k < 2; k++) {
  uint e = tid + k * 32;
  if (e >= 60) {
    break;
  }
  uint wbase = e * 768;
  float acc = 0.0f;
  for (uint d = 0; d < 768; d++) {
    acc += xsh[d] * w_router[wbase + d];
  }
  scores[e] = 1.0f / (1.0f + metal::exp(-acc));
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0) {
  // Top-4 of biased scores, strictly-greater insertion (earliest index wins
  // ties; exact ties are G2-exempt near-tie rows).
  float topv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint topi[4] = {0, 0, 0, 0};
  for (uint e = 0; e < 60; e++) {
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
  // Weights from the UNBIASED scores, same op order as the reference:
  // w / (sum + 1e-20) * 2.5.
  float s0 = scores[topi[0]];
  float s1 = scores[topi[1]];
  float s2 = scores[topi[2]];
  float s3 = scores[topi[3]];
  float denom = s0 + s1 + s2 + s3 + 1e-20f;
  uint obase = row * 4;
  inds[obase + 0] = topi[0];
  inds[obase + 1] = topi[1];
  inds[obase + 2] = topi[2];
  inds[obase + 3] = topi[3];
  weights[obase + 0] = (s0 / denom) * 2.5f;
  weights[obase + 1] = (s1 / denom) * 2.5f;
  weights[obase + 2] = (s2 / denom) * 2.5f;
  weights[obase + 3] = (s3 / denom) * 2.5f;
}
