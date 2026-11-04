#include <metal_simdgroup_matrix>
// Shared header (builder-1-a, hyp-1-experts-exact direction a).
// qvm_block: byte-identical copy of the starting point's helper
// (muse-spark/kernels/experts/attempt-4/block-1.metal): bit-faithful
// replica of MLX affine_gather_qvm for one 32-output block: each lane
// accumulates its strided K elements serially in float32 with the
// dequantized weight kept in float32, then a simd_sum tree reduces
// across the 32 lanes. Used below 60 rows.
inline void qvm_block(
    const device uint* W,
    const device bfloat* S,
    const device bfloat* B,
    const threadgroup bfloat* xv,
    uint K,
    uint G,
    uint Ww,
    uint g,
    uint wb,
    uint lane,
    thread float* p) {
  for (uint j = 0; j < 32u; ++j) {
    p[j] = 0.0f;
  }
  for (uint k = lane; k < K; k += 32u) {
    float x = (float)xv[k];
    float scale = (float)S[k * G + g];
    float bias = (float)B[k * G + g];
    float s0 = scale;
    float s1 = scale / 16.0f;
    ulong row = (ulong)k * (ulong)Ww + (ulong)wb;
    uint w4[4];
    for (uint j = 0; j < 4u; ++j) {
      w4[j] = W[row + j];
    }
    for (uint i = 0; i < 16u; ++i) {
      uint by = (w4[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
      p[2u * i] += x * (s0 * (float)(by & 0x0fu) + bias);
      p[2u * i + 1u] += x * (s1 * (float)(by & 0xf0u) + bias);
    }
  }
#pragma clang loop unroll(full)
  for (uint j = 0; j < 32u; ++j) {
    p[j] = simd_sum(p[j]);
  }
}

// qmm_block: per-row replica of MLX affine_gather_qmm_rhs arithmetic
// for one 32-output block of one pair (used at 60 rows and above).
// Weights are dequantized exactly as QuantizedBlockLoader/dequantize
// bits=4 does (quantized.h:617-623): (bfloat)(s*q + b) with
// sc = {s, s/16} on the byte nibbles, i.e. the dequantized weight is
// rounded to bfloat16 BEFORE the multiply. Accumulation follows
// BlockMMA (steel/gemm/mma.h:513-537) with BK=32: K tiles of 32 in
// increasing order, each tile four sequential 8-wide
// simdgroup_multiply_accumulate steps on float 8x8 fragments. This
// threadgroup holds a single row (row 0 of the fragment; rows 1-7
// are zero padding in separate accumulators, as schedule_row_tile's
// padded rows are in the reference), so per-row K order matches the
// reference while row grouping stays free. o32 receives the 32
// float accumulators; the caller rounds once to bfloat16.
inline void qmm_block(
    const device uint* W,
    const device bfloat* S,
    const device bfloat* B,
    const device bfloat* xv,
    uint Kdim,
    uint G,
    uint Ww,
    uint blk,
    uint lane,
    threadgroup float* Xs,
    threadgroup float* Ws,
    threadgroup float* tmp,
    thread float* o32) {
  simdgroup_matrix<float, 8, 8> acc[4];
  for (uint j = 0; j < 4u; ++j) {
    acc[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
  }
  for (uint kt = 0; kt < Kdim; kt += 32u) {
    for (uint i = lane; i < 256u; i += 32u) {
      uint row = i >> 5;
      uint col = i & 31u;
      Xs[i] = (row == 0u) ? (float)xv[kt + col] : 0.0f;
    }
    for (uint i = lane; i < 1024u; i += 32u) {
      uint k = i >> 5;
      uint n = i & 31u;
      uint kg = kt + k;
      float s = (float)S[kg * G + blk];
      float b = (float)B[kg * G + blk];
      uint bidx = n >> 1;
      uint word = W[(ulong)kg * (ulong)Ww + blk * 4u + (bidx >> 2)];
      uint byte = (word >> ((bidx & 3u) * 8u)) & 0xffu;
      float v = ((n & 1u) == 0u) ? (s * (float)(byte & 0x0fu) + b)
                                 : ((s / 16.0f) * (float)(byte & 0xf0u) + b);
      Ws[i] = (float)(bfloat)v;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 4u; ++kk) {
      simdgroup_matrix<float, 8, 8> A;
      simdgroup_load(A, Xs + kk * 8u, 32);
      for (uint j = 0; j < 4u; ++j) {
        simdgroup_matrix<float, 8, 8> Bm;
        simdgroup_load(Bm, Ws + kk * 256u + j * 8u, 32);
        simdgroup_multiply_accumulate(acc[j], A, Bm, acc[j]);
      }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < 4u; ++j) {
    simdgroup_store(acc[j], tmp + j * 8u, 32);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint j = 0; j < 32u; ++j) {
    o32[j] = tmp[j];
  }
}

// silu(gate) * up on bfloat16 values, computed entirely in bfloat16
// after every op: the starting point's proven lowering of the compiled
// swiglu (muse-spark/kernels/experts/attempt-4/block-0.metal:53-64).
inline bfloat silu_mul_bf16(bfloat gb, bfloat ub) {
  float g = (float)gb;
  bfloat e1 = (bfloat)metal::precise::exp(metal::abs(g));
  bfloat d = (bfloat)(1.0f + (float)e1);
  bfloat y = (bfloat)(1.0f / (float)d);
  bfloat s = (g < 0.0f) ? y : (bfloat)(1.0f - (float)y);
  bfloat p = (bfloat)(g * (float)s);
  return (bfloat)((float)p * (float)ub);
}
