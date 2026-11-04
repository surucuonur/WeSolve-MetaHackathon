// Fused routing, direction (b): minimal-change exact fix starting from
// muse-spark/kernels/routing/attempt-1's single kernel. Keep one 32-thread
// threadgroup per row and the serial top-4 scan/tie rule and weights sum
// exactly as attempt-1; change ONLY the scores:
//   (1) sigmoid to MLX verbatim y=1/(1+precise::exp(abs(acc))); (acc<0)?y:1-y
//   (2) product accumulation reordered scalar-wise, no MMA intrinsics:
//       rows==1: GEMV tree order (32 lane partials, each lane l serially sums
//                d = i*128 + l*4 + 0..3 for i=0..5, then shuffle tree
//                sn=16,8,4,2,1 emulated on the partial array);
//       rows>1: split-K partition order (32 partitions of BK=16 chunks:
//               partitions 0..30 cover 16 elements each, partition 31 covers
//               the remaining 272, matching steel split-K at 8/32 rows with
//               gemm_k_iterations=(768/16)/32=1 and the last partition
//               absorbing the rest; each partial summed scalar-wise in
//               increasing d, partials summed sequentially in partition
//               order starting from 0, as in gemm_splitk_accum).
// Grid is (rows*32, 1, 1), threadgroup (32, 1, 1).
//
// Per row: 60 experts, thread t computes experts t and (t+32 if t<28).
// x (768 bf16) is staged once into threadgroup memory; w_router rows are read
// straight from device memory. Thread 0 then does the serial top-4 scan over
// the 60 threadgroup scores, normalizes, and writes inds/weights.

uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
int rows = x_shape[0];

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
  if (rows == 1) {
    // GEMV tree order, scalar emulation in this thread.
    float lane_part[32];
    for (uint l = 0; l < 32; l++) {
      float p = 0.0f;
      for (uint i = 0; i < 6; i++) {
        uint base = i * 128 + l * 4;
        p += xsh[base + 0] * w_router[wbase + base + 0];
        p += xsh[base + 1] * w_router[wbase + base + 1];
        p += xsh[base + 2] * w_router[wbase + base + 2];
        p += xsh[base + 3] * w_router[wbase + base + 3];
      }
      lane_part[l] = p;
    }
    // Shuffle tree sn=16,8,4,2,1: in-place ascending is exact, since
    // lane_part[l+sn] is read before it is overwritten in this round.
    for (uint sn = 16; sn >= 1; sn >>= 1) {
      for (uint l = 0; l + sn < 32; l++) {
        lane_part[l] += lane_part[l + sn];
      }
    }
    acc = lane_part[0];
  } else {
    // Split-K partition order, scalar emulation in this thread.
    float part[32];
    for (uint pIdx = 0; pIdx < 32; pIdx++) {
      uint k_start = pIdx * 16;
      uint k_len = (pIdx == 31) ? 272 : 16;
      float p = 0.0f;
      for (uint d = k_start; d < k_start + k_len; d++) {
        p += xsh[d] * w_router[wbase + d];
      }
      part[pIdx] = p;
    }
    acc = 0.0f;
    for (uint pIdx = 0; pIdx < 32; pIdx++) {
      acc += part[pIdx];
    }
  }
  // MLX sigmoid verbatim (unary_ops.h Sigmoid).
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(acc)));
  scores[e] = (acc < 0.0f) ? y : 1.0f - y;
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
