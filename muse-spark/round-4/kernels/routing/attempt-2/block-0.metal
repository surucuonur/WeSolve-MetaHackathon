// hyp-2-routing-margin round-4 attempt-2 (builder-2-a, direction a).
// Builds on round-3 attempt-6 (byte-identical round-2 attempt-4 staged
// scalar kernels, max_rows 32). ONE main change: at rows > 16, share
// each w_router byte across 4 rows. One 256-thread threadgroup covers
// 4 rows (240 (row,expert) threads + 16 staging helpers); w chunks are
// staged once per group into wsh and reused for all 4 rows. Groups
// beyond ceil(rows/4) return immediately. At rows <= 16 the original
// staged scalar path (and at rows == 1 the GEMV path) runs verbatim,
// only remapped onto the shared threadgroup buffer tg[7616].
// Per-row order is bit-identical in both paths: each partition is a
// serial sum in increasing d, partitions are folded in increasing
// order, the sigmoid is MLX's verbatim form, the top-4 scan is the
// strictly-greater insertion scan, and the weights fold is unchanged.
// Threadgroup: 7616 * 4 B = 30,464 B of 32 KB; 256 threads.
uint gpos = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int rows = x_shape[0];

threadgroup float tg[7616];

if (rows > 16) {
  uint gbase_row = gpos * 4u;
  if (gbase_row >= uint(rows)) {
    return;
  }
  uint Rg = uint(rows) - gbase_row;
  if (Rg > 4u) {
    Rg = 4u;
  }
  // Preload the 4 rows' x into the padded layout: row lr at lr*816,
  // chunk c at c*17+k (same padding as the staged path).
  for (uint idx = tid; idx < Rg * 768u; idx += 256u) {
    uint lr = idx / 768u;
    uint d = idx - lr * 768u;
    uint r = gbase_row + lr;
    tg[lr * 816u + (d >> 4) * 17u + (d & 15u)] = float(x[r * 768u + d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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

  uint lr = tid / 60u;
  uint e = tid - lr * 60u;
  bool active = (tid < 240u) && (lr < Rg);
  uint r = gbase_row + lr;
  float tot = 0.0f;
  float pacc = 0.0f;
  uint curP = 4294967295u;

  for (uint gbase = 0u; gbase < 48u; gbase += 4u) {
    uint gn = 48u - gbase;
    if (gn > 4u) {
      gn = 4u;
    }
    for (uint lv = tid; lv < gn * 240u; lv += 256u) {
      uint s = lv / 240u;
      uint rem = lv - s * 240u;
      uint ee = rem / 4u;
      uint kv = rem - ee * 4u;
      uint c = gbase + s;
      float4 v = *(const device float4 *)(w_router + ee * 768u + c * 16u + kv * 4u);
      uint dst = 3520u + (s * 16u + kv * 4u) * 64u + ee;
      tg[dst] = v.x;
      tg[dst + 64u] = v.y;
      tg[dst + 128u] = v.z;
      tg[dst + 192u] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint c = gbase; c < gbase + gn; ++c) {
        uint p = (c < thr) ? (c / base_iters) : (partitions - 1u);
        if (p != curP) {
          if (curP != 4294967295u) {
            tot += pacc;
          }
          curP = p;
          pacc = 0.0f;
        }
        uint s = c - gbase;
        uint xb = lr * 816u + c * 17u;
        uint wb = 3520u + s * 1024u + e;
        for (uint k = 0u; k < 16u; ++k) {
          pacc += tg[xb + k] * tg[wb + k * 64u];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (active) {
    tot += pacc;
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(tot)));
    tg[3264u + lr * 64u + e] = (tot < 0.0f) ? y : 1.0f - y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (active && e == 0u) {
    float topv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint topi[4] = {0, 0, 0, 0};
    uint sb = 3264u + lr * 64u;
    for (uint ee = 0u; ee < 60u; ++ee) {
      float b = tg[sb + ee] + bias[ee];
      if (b > topv[0]) {
        topv[3] = topv[2]; topi[3] = topi[2];
        topv[2] = topv[1]; topi[2] = topi[1];
        topv[1] = topv[0]; topi[1] = topi[0];
        topv[0] = b; topi[0] = ee;
      } else if (b > topv[1]) {
        topv[3] = topv[2]; topi[3] = topi[2];
        topv[2] = topv[1]; topi[2] = topi[1];
        topv[1] = b; topi[1] = ee;
      } else if (b > topv[2]) {
        topv[3] = topv[2]; topi[3] = topi[2];
        topv[2] = b; topi[2] = ee;
      } else if (b > topv[3]) {
        topv[3] = b; topi[3] = ee;
      }
    }
    float s0 = tg[sb + topi[0]];
    float s1 = tg[sb + topi[1]];
    float s2 = tg[sb + topi[2]];
    float s3 = tg[sb + topi[3]];
    float denom = s0 + s1 + s2 + s3 + 1e-20f;
    uint obase = r * 4u;
    inds[obase + 0] = topi[0];
    inds[obase + 1] = topi[1];
    inds[obase + 2] = topi[2];
    inds[obase + 3] = topi[3];
    weights[obase + 0] = (s0 / denom) * 2.5f;
    weights[obase + 1] = (s1 / denom) * 2.5f;
    weights[obase + 2] = (s2 / denom) * 2.5f;
    weights[obase + 3] = (s3 / denom) * 2.5f;
  }
  return;
}

// Staged scalar path, verbatim from round-3 attempt-6 apart from the
// tg offsets: xpad tg[0..815], scores tg[816..875],
// parts tg[876..2923], wsh tg[2924..7019].
uint row = gpos;

for (uint d = tid; d < 768u; d += 256u) {
  tg[(d >> 4) * 17u + (d & 15u)] = float(x[row * 768u + d]);
}
for (uint i = tid; i < 2048u; i += 256u) {
  tg[876u + i] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (rows == 1) {
  for (uint e = sg; e < 60u; e += 8u) {
    float acc = 0.0f;
    for (uint i = 0u; i < 6u; ++i) {
      uint base = i * 128u + lane * 4u;
      uint xb = (base >> 4) * 17u + (base & 15u);
      uint wb = e * 768u + base;
      acc += tg[xb + 0u] * w_router[wb + 0u];
      acc += tg[xb + 1u] * w_router[wb + 1u];
      acc += tg[xb + 2u] * w_router[wb + 2u];
      acc += tg[xb + 3u] * w_router[wb + 3u];
    }
    acc += simd_shuffle_down(acc, 16);
    acc += simd_shuffle_down(acc, 8);
    acc += simd_shuffle_down(acc, 4);
    acc += simd_shuffle_down(acc, 2);
    acc += simd_shuffle_down(acc, 1);
    if (lane == 0u) {
      tg[816u + e] = acc;
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
      uint dst = 2924u + (s * 16u + kv * 4u) * 64u + e;
      tg[dst] = v.x;
      tg[dst + 64u] = v.y;
      tg[dst + 128u] = v.z;
      tg[dst + 192u] = v.w;
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
      float acc = tg[876u + p * 64u + e];
      for (uint c = c_lo; c < c_hi; ++c) {
        uint s = c - gbase;
        uint xb = c * 17u;
        uint wb = 2924u + s * 1024u + e;
        for (uint k = 0u; k < 16u; ++k) {
          acc += tg[xb + k] * tg[wb + k * 64u];
        }
      }
      tg[876u + p * 64u + e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint e = tid; e < 60u; e += 256u) {
    float acc = 0.0f;
    for (uint p = 0u; p < partitions; ++p) {
      acc += tg[876u + p * 64u + e];
    }
    tg[816u + e] = acc;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// MLX Sigmoid verbatim (unary_ops.h): y on |x|, branch on sign, float32.
for (uint e = tid; e < 60u; e += 256u) {
  float s = tg[816u + e];
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
  tg[816u + e] = (s < 0.0f) ? y : 1.0f - y;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
  float topv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint topi[4] = {0, 0, 0, 0};
  for (uint e = 0u; e < 60u; ++e) {
    float b = tg[816u + e] + bias[e];
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
  float s0 = tg[816u + topi[0]];
  float s1 = tg[816u + topi[1]];
  float s2 = tg[816u + topi[2]];
  float s3 = tg[816u + topi[3]];
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
