// hyp-3-routing-prefill direction (b), round-3 attempt-5 (builder-3-e),
// kernel 1 of 2: GEMM-shaped prefill kernel sharing w_router across rows.
// Copied from attempt-4 (builder-3-d) with one change: 8 rows per
// threadgroup (480 threads) instead of 10 (600), under the 576 pipeline cap.
// Grid rows*60, threadgroup 480: one threadgroup spans 8 rows with one
// thread per (row, expert). 60 divides 480, so every threadgroup covers
// whole rows (the last may cover fewer than 8) and no bounds check is
// needed: every thread's (row, expert) is valid by construction.
//   wsh: 4 K-chunks staged at once as [slot][k][expert] with expert
//        stride 64 (attempt-4's exact staging, coalesced float4 loads),
//        read once per 8 rows instead of once per row.
//   xst: the staged groups' 64 d-values per row (8 rows x 64 floats).
//   sc:  sigmoid scores per row (8 rows x 64 stride) for the top-4.
// Each thread accumulates its (row, expert) score with the exact
// rows-branched per-row order: split-K partition partials in registers
// (partitions from the rows-only formula, each partial a serial scalar
// sum in increasing d), folded in partition order; a single serial sum
// above 768 rows. Sharing changes where operands are read from, never
// the order, so G2 follows attempt-4's proof. At rows <= 32 this kernel
// only copies kernel 0's (attempt-4 verbatim) outputs.
// Threadgroup: (4096 + 512 + 512) * 4 B = 20,480 B of 32 KB.
uint gid = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint g = threadgroup_position_in_grid.x;
int rows = x_shape[0];

if (rows <= 32) {
  uint n = uint(rows) * 4u;
  if (gid < n) {
    inds[gid] = inds0[gid];
    weights[gid] = weights0[gid];
  }
  return;
}

uint lr = tid / 60u;
uint e = tid - lr * 60u;
uint r = g * 8u + lr;

threadgroup float wsh[4096];
threadgroup float xst[512];
threadgroup float sc[512];

uint Rg = uint(rows) - g * 8u;
if (Rg > 8u) {
  Rg = 8u;
}
uint nt = Rg * 60u;

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

float part[16];
for (uint p = 0u; p < 16u; ++p) {
  part[p] = 0.0f;
}

for (uint gbase = 0u; gbase < 48u; gbase += 4u) {
  for (uint lv = tid; lv < 960u; lv += nt) {
    uint s = lv / 240u;
    uint rem = lv - s * 240u;
    uint ee = rem / 4u;
    uint kv = rem - ee * 4u;
    uint c = gbase + s;
    float4 v = *(const device float4 *)(w_router + ee * 768u + c * 16u + kv * 4u);
    uint dst = (s * 16u + kv * 4u) * 64u + ee;
    wsh[dst] = v.x;
    wsh[dst + 64u] = v.y;
    wsh[dst + 128u] = v.z;
    wsh[dst + 192u] = v.w;
  }
  uint xd = gbase * 16u;
  for (uint lx = tid; lx < Rg * 64u; lx += nt) {
    uint lrr = lx / 64u;
    uint jj = lx - lrr * 64u;
    uint rr = g * 8u + lrr;
    xst[lrr * 64u + jj] = float(x[rr * 768u + xd + jj]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = gbase; c < gbase + 4u; ++c) {
    uint p = (c < thr) ? (c / base_iters) : (partitions - 1u);
    uint s = c - gbase;
    float acc = part[p];
    uint xb = lr * 64u + s * 16u;
    uint wb = s * 1024u + e;
    for (uint k = 0u; k < 16u; ++k) {
      acc += xst[xb + k] * wsh[wb + k * 64u];
    }
    part[p] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

float tot = 0.0f;
for (uint p = 0u; p < partitions; ++p) {
  tot += part[p];
}

// MLX Sigmoid verbatim (unary_ops.h): y on |x|, branch on sign, float32.
float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(tot)));
sc[lr * 64u + e] = (tot < 0.0f) ? y : 1.0f - y;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (e == 0u) {
  float topv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  uint topi[4] = {0, 0, 0, 0};
  uint sb = lr * 64u;
  for (uint ee = 0u; ee < 60u; ++ee) {
    float b = sc[sb + ee] + bias[ee];
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
  float s0 = sc[sb + topi[0]];
  float s1 = sc[sb + topi[1]];
  float s2 = sc[sb + topi[2]];
  float s3 = sc[sb + topi[3]];
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
