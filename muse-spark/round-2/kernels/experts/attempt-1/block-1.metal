// Header for hyp-1-experts-exact direction (b), round-2 attempt-1.
// Two per-block arithmetics behind one fused kernel (branch in block-0):
//  - qvm_block: bit-faithful replica of MLX affine_gather_qvm (unchanged
//    copy of the starting point muse-spark/kernels/experts/attempt-4).
//  - qmm_accum: prefill (rows>=60) scalar mimic of affine_gather_qmm_rhs:
//    dequantize to bfloat16 first, then accumulate K contiguously in
//    BK=32 tiles (see notes.md).

// Decode arithmetic (rows < 60): each lane accumulates its strided K
// elements serially in float32, dequantizing on the fly as
// x * (scale * q + bias), then a simd_sum tree reduces across lanes.
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

// Prefill arithmetic (rows >= 60): one output (lane-th of the 32-output
// block g), accumulated by a single thread so the add order is exactly
// the contiguous K order, tile by tile (BK=32), mimicking BlockMMA's
// sequential tile order. The weight is rounded to bfloat16 before the
// multiply, as QuantizedBlockLoader/dequantize does into Ws:
//   w = bfloat(float(scale) * q + float(bias))
// (MLX writes the odd nibble as (s/16)*(byte&0xf0)+b; that is the same
// float value: scaling by powers of two is exact and s*q is exact in
// float32 for 4-bit q, so both forms round identically.)
inline float qmm_accum(
    const device uint* W,
    const device bfloat* S,
    const device bfloat* B,
    const threadgroup bfloat* xv,
    uint K,
    uint G,
    uint Ww,
    uint g,
    uint wb,
    uint lane) {
  uint word = wb + (lane >> 3u);
  uint shift = (lane & 7u) * 4u;
  float acc = 0.0f;
  for (uint tile = 0u; tile < K; tile += 32u) {
    for (uint kk = 0u; kk < 32u; ++kk) {
      uint k = tile + kk;
      uint w = W[(ulong)k * (ulong)Ww + (ulong)word];
      uint q = (w >> shift) & 0xfu;
      float wq = (float)S[k * G + g] * (float)q + (float)B[k * G + g];
      acc += (float)xv[k] * (float)(bfloat)wq;
    }
  }
  return acc;
}
