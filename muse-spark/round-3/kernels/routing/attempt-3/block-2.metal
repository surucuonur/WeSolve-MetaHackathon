// hyp-3-routing-prefill direction (c), round-3 attempt-3, kernel-B.
// Single-partition path (rows > 768, the only regime where attempt-4's
// partition formula yields partitions == 1): each expert's whole 768-product
// chain runs in one thread-private accumulator, so the 2,048-float `parts`
// array is deleted. Staging, xpad layout, product order, sigmoid, top-4 and
// weights fold are attempt-4's verbatim; at rows <= 768 this kernel instead
// copies kernel-A's outputs through (bit-exact) and returns.
// Threadgroup: (816 + 60 + 4096) * 4 B = 19,888 B of 32 KB (60.7%).
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
int rows = x_shape[0];
if (rows <= 768) {
  for (uint j = tid; j < 4u; j += 256u) {
    uint o = row * 4u + j;
    inds[o] = inds_a[o];
    weights[o] = weights_a[o];
  }
  return;
}

threadgroup float xpad[816];
threadgroup float scores[60];
threadgroup float wsh[4096];

for (uint d = tid; d < 768u; d += 256u) {
  xpad[(d >> 4) * 17u + (d & 15u)] = float(x[row * 768u + d]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// partitions == 1, base_iters == 48: thread e owns expert e's full chain.
float acc = 0.0f;
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
  if (tid < 60u) {
    uint e = tid;
    uint gend = gbase + gn;
    for (uint c = gbase; c < gend; ++c) {
      uint s = c - gbase;
      uint xb = c * 17u;
      uint wb = s * 1024u + e;
      for (uint k = 0u; k < 16u; ++k) {
        acc += xpad[xb + k] * wsh[wb + k * 64u];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid < 60u) {
  scores[tid] = acc;
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
