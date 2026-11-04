// Header for hyp-1-experts-exact hardening, round-2 attempt-6 (builder-1-f).
// Builds on attempt-5 (fused single kernel, 512 threads/pair, interleaved
// gate/up + parallel block stores).
// One main change: decode down interleaved via qvm_down_pair below.
//  - qvm_block: bit-faithful replica of MLX affine_gather_qvm (unchanged
//    from attempt-1; kept verbatim, now unused — its per-block arithmetic
//    is copied into the two pair helpers).
//  - qvm_gate_up_pair: gate and up for the same 32-output block in one
//    K loop — unchanged from attempt-4/5.
//  - qvm_down_pair: two down 32-output blocks in one K loop — both
//    blocks' weight/scale loads issued before either accumulates,
//    per-block order unchanged (new in this attempt).
//  - qmm_accum: prefill (rows>=60) scalar mimic of affine_gather_qmm_rhs,
//    unchanged from attempt-1 (see notes.md).

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

// Decode gate/up pair (rows < 60): same per-block arithmetic as two
// qvm_block calls for block g, interleaved. For each strided k the
// gate and up weight words, scales and biases are all loaded before
// either block accumulates, so the two blocks' load latencies overlap.
// Each block's adds still run in qvm_block's order — k = lane,
// lane+32, ... serial, same x*(s*q+bias) terms in the same sequence
// per output — pg and pu are separate accumulators, and each gets
// the same simd_sum tree afterwards, so every block's bits are
// identical to the sequential pair of qvm_block calls.
inline void qvm_gate_up_pair(
    const device uint* Wg,
    const device bfloat* Sg,
    const device bfloat* Bg,
    const device uint* Wu,
    const device bfloat* Su,
    const device bfloat* Bu,
    const threadgroup bfloat* xv,
    uint K,
    uint G,
    uint Ww,
    uint g,
    uint wb,
    uint lane,
    thread float* pg,
    thread float* pu) {
  for (uint j = 0; j < 32u; ++j) {
    pg[j] = 0.0f;
    pu[j] = 0.0f;
  }
  for (uint k = lane; k < K; k += 32u) {
    float x = (float)xv[k];
    float scale_g = (float)Sg[k * G + g];
    float bias_g = (float)Bg[k * G + g];
    float s0g = scale_g;
    float s1g = scale_g / 16.0f;
    ulong row_g = (ulong)k * (ulong)Ww + (ulong)wb;
    uint w4g[4];
    for (uint j = 0; j < 4u; ++j) {
      w4g[j] = Wg[row_g + j];
    }
    float scale_u = (float)Su[k * G + g];
    float bias_u = (float)Bu[k * G + g];
    float s0u = scale_u;
    float s1u = scale_u / 16.0f;
    ulong row_u = (ulong)k * (ulong)Ww + (ulong)wb;
    uint w4u[4];
    for (uint j = 0; j < 4u; ++j) {
      w4u[j] = Wu[row_u + j];
    }
    for (uint i = 0; i < 16u; ++i) {
      uint by = (w4g[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
      pg[2u * i] += x * (s0g * (float)(by & 0x0fu) + bias_g);
      pg[2u * i + 1u] += x * (s1g * (float)(by & 0xf0u) + bias_g);
    }
    for (uint i = 0; i < 16u; ++i) {
      uint by = (w4u[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
      pu[2u * i] += x * (s0u * (float)(by & 0x0fu) + bias_u);
      pu[2u * i + 1u] += x * (s1u * (float)(by & 0xf0u) + bias_u);
    }
  }
#pragma clang loop unroll(full)
  for (uint j = 0; j < 32u; ++j) {
    pg[j] = simd_sum(pg[j]);
  }
#pragma clang loop unroll(full)
  for (uint j = 0; j < 32u; ++j) {
    pu[j] = simd_sum(pu[j]);
  }
}

// Decode down pair (rows < 60): two down blocks g0/g1 in one K loop,
// mirroring qvm_gate_up_pair. For each strided k both blocks' weight
// words, scales and biases are loaded before either block accumulates,
// into separate accumulators p0/p1. Each block keeps qvm_block's exact
// order — k = lane, lane+32, ... serial, same x*(s*q+bias) terms in the
// same per-output sequence — and the same simd_sum tree afterwards, so
// every block's bits are identical to two sequential qvm_block calls.
inline void qvm_down_pair(
    const device uint* W,
    const device bfloat* S,
    const device bfloat* B,
    const threadgroup bfloat* xv,
    uint K,
    uint G,
    uint Ww,
    uint g0,
    uint wb0,
    uint g1,
    uint wb1,
    uint lane,
    thread float* p0,
    thread float* p1) {
  for (uint j = 0; j < 32u; ++j) {
    p0[j] = 0.0f;
    p1[j] = 0.0f;
  }
  for (uint k = lane; k < K; k += 32u) {
    float x = (float)xv[k];
    float scale0 = (float)S[k * G + g0];
    float bias0 = (float)B[k * G + g0];
    float s00 = scale0;
    float s10 = scale0 / 16.0f;
    ulong row0 = (ulong)k * (ulong)Ww + (ulong)wb0;
    uint w40[4];
    for (uint j = 0; j < 4u; ++j) {
      w40[j] = W[row0 + j];
    }
    float scale1 = (float)S[k * G + g1];
    float bias1 = (float)B[k * G + g1];
    float s01 = scale1;
    float s11 = scale1 / 16.0f;
    ulong row1 = (ulong)k * (ulong)Ww + (ulong)wb1;
    uint w41[4];
    for (uint j = 0; j < 4u; ++j) {
      w41[j] = W[row1 + j];
    }
    for (uint i = 0; i < 16u; ++i) {
      uint by = (w40[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
      p0[2u * i] += x * (s00 * (float)(by & 0x0fu) + bias0);
      p0[2u * i + 1u] += x * (s10 * (float)(by & 0xf0u) + bias0);
    }
    for (uint i = 0; i < 16u; ++i) {
      uint by = (w41[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
      p1[2u * i] += x * (s01 * (float)(by & 0x0fu) + bias1);
      p1[2u * i + 1u] += x * (s11 * (float)(by & 0xf0u) + bias1);
    }
  }
#pragma clang loop unroll(full)
  for (uint j = 0; j < 32u; ++j) {
    p0[j] = simd_sum(p0[j]);
  }
#pragma clang loop unroll(full)
  for (uint j = 0; j < 32u; ++j) {
    p1[j] = simd_sum(p1[j]);
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
