// hyp-2-experts-prefill direction (c), round-3 attempt-3 (builder-2-c).
// Prefill kernel: fused grouped replica of gather_qmm_rhs issue. Each
// threadgroup handles one (expert, 16-pair tile): it gathers the tile's
// pairs from the count kernel's sorted list, streams gate/up weights once
// per tile (shared across <=16 rows), accumulates with 8-wide frag MMA in
// BK=32 order, runs swiglu on-chip, then streams down weights the same
// way. No empty tiles, offsets computed once, x/hidden never leave the
// chip per tile. Rows < 60 return immediately (decode kernel owns them).
uint gid = threadgroup_position_in_grid.x;
uint thread_idx = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint rows = x_shape[0];
if (rows < 60u) return;
uint total_tiles = tile_offsets[60];
if (gid >= total_tiles) return;
uint e = 0u;
uint t = 0u;
for (uint ee = 0u; ee < 60u; ++ee) {
  if (gid < tile_offsets[ee + 1u]) {
    e = ee;
    t = gid - tile_offsets[ee];
    break;
  }
}
uint cnt_e = expert_offsets[e + 1u] - expert_offsets[e];
uint m = cnt_e - t * 16u;
if (m > 16u) m = 16u;

threadgroup uint tg_pairs[16];
threadgroup bfloat tg_h[16 * 384];
threadgroup bfloat Xs[16 * 40];
threadgroup bfloat Ws0[32 * 40];
threadgroup bfloat Ws1[32 * 40];

if (thread_idx < 16u) {
  if (thread_idx < m) {
    tg_pairs[thread_idx] =
        sorted_pairs[expert_offsets[e] + t * 16u + thread_idx];
  } else {
    tg_pairs[thread_idx] = 0u;
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

uint fm;
uint fn;
frag_coord(lane, fm, fn);

ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;

// Gate + up, interleaved per K-tile, swiglu on-chip into tg_h.
for (uint g = 0u; g < 12u; ++g) {
  float Cg[4][2];
  float Cu[4][2];
  for (uint f = 0u; f < 4u; ++f) {
    Cg[f][0] = 0.0f;
    Cg[f][1] = 0.0f;
    Cu[f][0] = 0.0f;
    Cu[f][1] = 0.0f;
  }
  for (uint k = 0u; k < 24u; ++k) {
    uint k0 = k * 32u;
    load_x_tile(x, tg_pairs, m, k0, 768u, Xs, thread_idx);
    dequant_tile_4bit(
        gate_w + eW, gate_scales + eS, gate_biases + eS, k0, g, 48u, 12u,
        Ws0, thread_idx);
    dequant_tile_4bit(
        up_w + eW, up_scales + eS, up_biases + eS, k0, g, 48u, 12u, Ws1,
        thread_idx);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint kk = 0u; kk < 32u; kk += 8u) {
      mma_step(Xs, Ws0, sg, fm, fn, kk, Cg);
      mma_step(Xs, Ws1, sg, fm, fn, kk, Cu);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  for (uint f = 0u; f < 4u; ++f) {
    for (uint j = 0u; j < 2u; ++j) {
      uint row = sg * 8u + fm;
      uint col = g * 32u + f * 8u + fn + j;
      if (row < m) {
        bfloat gb = (bfloat)Cg[f][j];
        bfloat ub = (bfloat)Cu[f][j];
        float gg = (float)gb;
        bfloat e1 = (bfloat)metal::precise::exp(metal::abs(gg));
        bfloat d = (bfloat)(1.0f + (float)e1);
        bfloat yy = (bfloat)(1.0f / (float)d);
        bfloat ss = (gg < 0.0f) ? yy : (bfloat)(1.0f - (float)yy);
        bfloat pp = (bfloat)(gg * (float)ss);
        tg_h[row * 384u + col] = (bfloat)((float)pp * (float)ub);
      }
    }
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

// Down: hidden (tg_h) x dequantized weights, positional stores.
for (uint g = 0u; g < 24u; ++g) {
  float Cd[4][2];
  for (uint f = 0u; f < 4u; ++f) {
    Cd[f][0] = 0.0f;
    Cd[f][1] = 0.0f;
  }
  for (uint k = 0u; k < 12u; ++k) {
    uint k0 = k * 32u;
    load_h_tile(tg_h, m, k0, Xs, thread_idx);
    dequant_tile_4bit(
        down_w + eW, down_scales + eS, down_biases + eS, k0, g, 96u, 24u,
        Ws0, thread_idx);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint kk = 0u; kk < 32u; kk += 8u) {
      mma_step(Xs, Ws0, sg, fm, fn, kk, Cd);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  for (uint f = 0u; f < 4u; ++f) {
    for (uint j = 0u; j < 2u; ++j) {
      uint row = sg * 8u + fm;
      uint col = g * 32u + f * 8u + fn + j;
      if (row < m) {
        uint pair = tg_pairs[row];
        y_prefill[pair * 768u + col] = (bfloat)Cd[f][j];
      }
    }
  }
}
