// Header for hyp-2-experts-prefill direction (c), round-3 attempt-3 (builder-2-c).
// Grouped prefill helpers: MLX gather_qmm_rhs-shaped tiling (BM=16, BN=32,
// BK=32) with bfloat-rounded dequant and 8-wide frag MMA in BK=32 order.
// Dequant replicates MLX dequantize bits=4 exactly (s*(low)+b,
// (s/16)*(high masked)+b); the loader mapping replicates
// QuantizedBlockLoader (bi=thread/2 pairs sharing each K-row).
#include <metal_simdgroup_matrix>

// Frag coordinate within an 8x8 simdgroup_matrix tile (MLX BaseMMAFrag).
inline void frag_coord(uint lane, thread uint &fm, thread uint &fn) {
  uint qid = lane >> 2;
  fm = (qid & 4u) + ((lane >> 1) & 3u);
  fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
}

// Dequantize one 32x32 weight tile into Ws (stride 40) from packed 4-bit.
// 64 threads: pair (2i,2i+1) shares K-row bi=i, writing outputs 0-15/16-31.
inline void dequant_tile_4bit(
    const device uint *W_e,
    const device bfloat *S_e,
    const device bfloat *B_e,
    uint k0,
    uint g,
    uint Ww,
    uint G,
    threadgroup bfloat *Ws,
    uint thread_idx) {
  uint bi = thread_idx >> 1;
  uint half = thread_idx & 1u;
  uint k = k0 + bi;
  uint wb = g * 4u;
  float s = (float)S_e[(ulong)k * (ulong)G + (ulong)g];
  float b = (float)B_e[(ulong)k * (ulong)G + (ulong)g];
  float s0 = s;
  float s1 = s / 16.0f;
  ulong base = (ulong)k * (ulong)Ww + (ulong)wb;
  uint w_lo = W_e[base + half * 2u];
  uint w_hi = W_e[base + half * 2u + 1u];
  for (uint j = 0u; j < 8u; ++j) {
    uint w = (j < 4u) ? w_lo : w_hi;
    uint by = (w >> ((j & 3u) * 8u)) & 0xffu;
    Ws[bi * 40u + half * 16u + 2u * j] =
        (bfloat)(s0 * (float)(by & 0x0fu) + b);
    Ws[bi * 40u + half * 16u + 2u * j + 1u] =
        (bfloat)(s1 * (float)(by & 0xf0u) + b);
  }
}

// Gather one 16x32 x tile into Xs (stride 40) via the tile's pair list.
// Threads (4i..4i+3) own tile-row i; rows past m are zero-padded.
inline void load_x_tile(
    const device bfloat *x,
    const threadgroup uint *pairs,
    uint m,
    uint k0,
    uint K,
    threadgroup bfloat *Xs,
    uint thread_idx) {
  uint bi = thread_idx >> 2;
  uint bj = (thread_idx & 3u) * 8u;
  if (bi >= m) {
    for (uint j = 0u; j < 8u; ++j) {
      Xs[bi * 40u + bj + j] = (bfloat)0.0f;
    }
  } else {
    uint r = pairs[bi] >> 2;
    ulong src = (ulong)r * (ulong)K + (ulong)(k0 + bj);
    for (uint j = 0u; j < 8u; ++j) {
      Xs[bi * 40u + bj + j] = x[src + j];
    }
  }
}

// Load one 16x32 hidden tile from tg_h (stride 384) into Xs (stride 40).
inline void load_h_tile(
    const threadgroup bfloat *tg_h,
    uint m,
    uint k0,
    threadgroup bfloat *Xs,
    uint thread_idx) {
  uint bi = thread_idx >> 2;
  uint bj = (thread_idx & 3u) * 8u;
  if (bi >= m) {
    for (uint j = 0u; j < 8u; ++j) {
      Xs[bi * 40u + bj + j] = (bfloat)0.0f;
    }
  } else {
    for (uint j = 0u; j < 8u; ++j) {
      Xs[bi * 40u + bj + j] = tg_h[bi * 384u + k0 + bj + j];
    }
  }
}

// One 8-wide MMA step over Xs/Ws tiles into C[4][2] (sg's 8 rows x 32 cols).
inline void mma_step(
    const threadgroup bfloat *Xs,
    const threadgroup bfloat *Ws,
    uint sg,
    uint fm,
    uint fn,
    uint kk,
    thread float (*C)[2]) {
  uint a_row = sg * 8u + fm;
  float a0 = (float)Xs[a_row * 40u + kk + fn];
  float a1 = (float)Xs[a_row * 40u + kk + fn + 1u];
  simdgroup_barrier(metal::mem_flags::mem_none);
  float B[4][2];
  for (uint f = 0u; f < 4u; ++f) {
    uint b_row = kk + fm;
    uint b_col = f * 8u + fn;
    B[f][0] = (float)Ws[b_row * 40u + b_col];
    B[f][1] = (float)Ws[b_row * 40u + b_col + 1u];
  }
  simdgroup_barrier(metal::mem_flags::mem_none);
  for (uint f = 0u; f < 4u; ++f) {
    metal::simdgroup_matrix<float, 8, 8> Am;
    metal::simdgroup_matrix<float, 8, 8> Bm;
    metal::simdgroup_matrix<float, 8, 8> Cm;
    metal::simdgroup_matrix<float, 8, 8> Dm;
    Am.thread_elements()[0] = a0;
    Am.thread_elements()[1] = a1;
    Bm.thread_elements()[0] = B[f][0];
    Bm.thread_elements()[1] = B[f][1];
    Cm.thread_elements()[0] = C[f][0];
    Cm.thread_elements()[1] = C[f][1];
    metal::simdgroup_multiply_accumulate(Dm, Am, Bm, Cm);
    C[f][0] = Dm.thread_elements()[0];
    C[f][1] = Dm.thread_elements()[1];
  }
}
