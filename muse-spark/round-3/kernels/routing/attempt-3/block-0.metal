// hyp-3-routing-prefill direction (c), round-3 attempt-3, kernel-A (general path).
// Attempt-4's block-0 verbatim except: (1) uniform early exit for rows > 768
// (kernel-B serves those); (2) outputs renamed inds->inds_a, weights->weights_a
// (kernel-B copies them through on this path). Regime branches, partition
// formula, staging, scalar orders, sigmoid, top-4 and weights fold unchanged.
// Threadgroup: (816 + 60 + 2048 + 4096) * 4 B = 28,080 B of 32 KB.
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int rows = x_shape[0];
if (rows > 768) {
  return;
}

threadgroup float xpad[816];
threadgroup float scores[60];
threadgroup float parts[2048];
threadgroup float wsh[4096];

for (uint d = tid; d < 768u; d += 256u) {
  xpad[(d >> 4) * 17u + (d & 15u)] = float(x[row * 768u + d]);
}
for (uint i = tid; i < 2048u; i += 256u) {
  parts[i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (rows == 1) {
  for (uint e = sg; e < 60u; e += 8u) {
    float acc = 0.0f;
    for (uint i = 0u; i < 6u; ++i) {
      uint base = i * 128u + lane * 4u;
      uint xb = (base >> 4) * 17u + (base & 15u);
      uint wb = e * 768u + base;
      acc += xpad[xb + 0u] * w_router[wb + 0u];
      acc += xpad[xb + 1u] * w_router[wb + 1u];
      acc += xpad[xb + 2u] * w_router[wb + 2u];
      acc += xpad[xb + 3u] * w_router[wb + 3u];
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
  uint thr = (partitions - 1u) * base_iters;
  for (uint gbase = 0u; gbase < 48u; gbase += 4u) {
    uint gn = 48u - gbase;
    if (gn > 4u) {
      gn = 4u;
    }
    for (uint lv = tid; lv < gn * 240u; lv += 256u) {
      uint s = lv / 240u;
      uint rem = lv - s * 240u;
      uint e = rem / 4u;
      uint kv = rem - e * 4u;
      uint c = gbase + s;
      float4 v = *(const device float4 *)(w_router + e * 768u + c * 16u + kv * 4u);
      uint dst = (s * 16u + kv * 4u) * 64u + e;
      wsh[dst] = v.x;
      wsh[dst + 64u] = v.y;
      wsh[dst + 128u] = v.z;
      wsh[dst + 192u] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint clast = gbase + gn - 1u;
    uint p_first = (gbase < thr) ? (gbase / base_iters) : (partitions - 1u);
    uint p_last = (clast < thr) ? (clast / base_iters) : (partitions - 1u);
    uint dparts = p_last - p_first + 1u;
    if (tid < dparts * 60u) {
      uint dp = tid / 60u;
      uint e = tid - dp * 60u;
      uint p = p_first + dp;
      uint pc0 = p * base_iters;
      uint pc1 = (p + 1u == partitions) ? 48u : (p + 1u) * base_iters;
      uint c_lo = pc0 > gbase ? pc0 : gbase;
      uint gend = gbase + gn;
      uint c_hi = pc1 < gend ? pc1 : gend;
      float acc = parts[p * 64u + e];
      for (uint c = c_lo; c < c_hi; ++c) {
        uint s = c - gbase;
        uint xb = c * 17u;
        uint wb = s * 1024u + e;
        for (uint k = 0u; k < 16u; ++k) {
          acc += xpad[xb + k] * wsh[wb + k * 64u];
        }
      }
      parts[p * 64u + e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint e = tid; e < 60u; e += 256u) {
    float acc = 0.0f;
    for (uint p = 0u; p < partitions; ++p) {
      acc += parts[p * 64u + e];
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
  inds_a[obase + 0] = topi[0];
  inds_a[obase + 1] = topi[1];
  inds_a[obase + 2] = topi[2];
  inds_a[obase + 3] = topi[3];
  weights_a[obase + 0] = (s0 / denom) * 2.5f;
  weights_a[obase + 1] = (s1 / denom) * 2.5f;
  weights_a[obase + 2] = (s2 / denom) * 2.5f;
  weights_a[obase + 3] = (s3 / denom) * 2.5f;
}
