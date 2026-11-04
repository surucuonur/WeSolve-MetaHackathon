// hyp-2-routing-exact direction (a), round-2 attempt-1 (builder-2-a).
// Faithful MLX-regime replica of routing in ONE fused kernel.
// One threadgroup (256 threads, 8 simdgroups) per row; grid rows*256.
// Branches on rows in-Metal (function of rows, not the six graded values):
//   rows == 1      GEMV: per expert, 32 lanes hold d=i*128+lane*4+0..3
//                  (i=0..5) serial partials, shuffle tree 16,8,4,2,1.
//   1 < rows <=768 steel split-K: K=768 in BK=16 chunks grouped into
//                  partitions (formula below), each partition accumulated
//                  with simdgroup 8x8 MMA (two 8-wide ops per chunk),
//                  partials added to the total in partition order.
//   rows > 768     regular steel: BK=16 chunks in order, MMA accumulation
//                  chunk after chunk (one partition covering all chunks).
// Only row 0 of each 8x8 A tile is the real row (other rows staged zero);
// per-element K order is the steel order regardless of the 8-row tile.
// Sigmoid is MLX verbatim; top-4 and weights keep the starting point's
// strictly-greater insertion and ((w0+w1)+w2)+w3 association.
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int rows = x_shape[0];

threadgroup float xsh[768];
threadgroup float scores[60];
threadgroup float Ash[128];
threadgroup float Bsh[1024];
threadgroup float Csh[512];

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
  uint ebase = sg * 8u;
  simdgroup_float8x8 tot = make_filled_simdgroup_matrix<float, 8>(0.0f);
  for (uint p = 0u; p < partitions; ++p) {
    uint c0 = p * base_iters;
    uint c1 = (p + 1u == partitions) ? 48u : (p + 1u) * base_iters;
    simdgroup_float8x8 part = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint c = c0; c < c1; ++c) {
      uint ks = c * 16u;
      if (tid < 128u) {
        uint rr = tid / 16u;
        uint cc = tid % 16u;
        Ash[tid] = (rr == 0u) ? xsh[ks + cc] : 0.0f;
      }
      for (uint t = 0u; t < 4u; ++t) {
        uint idx = lane + t * 32u;
        uint kk = idx / 8u;
        uint jj = idx % 8u;
        uint ee = ebase + jj;
        Bsh[sg * 128u + idx] =
            (ee < 60u) ? w_router[ee * 768u + ks + kk] : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_float8x8 af0, bf0, af1, bf1;
      simdgroup_load(af0, Ash, 16, ulong2(0, 0), false);
      simdgroup_load(bf0, Bsh + sg * 128u, 8, ulong2(0, 0), false);
      simdgroup_multiply_accumulate(part, af0, bf0, part);
      simdgroup_load(af1, Ash + 8, 16, ulong2(0, 0), false);
      simdgroup_load(bf1, Bsh + sg * 128u + 64u, 8, ulong2(0, 0), false);
      simdgroup_multiply_accumulate(part, af1, bf1, part);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tot.thread_elements()[0] += part.thread_elements()[0];
    tot.thread_elements()[1] += part.thread_elements()[1];
  }
  simdgroup_store(tot, Csh + sg * 64u, 8, ulong2(0, 0), false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    for (uint j = 0u; j < 8u; ++j) {
      uint e = ebase + j;
      if (e < 60u) {
        scores[e] = Csh[sg * 64u + j];
      }
    }
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
