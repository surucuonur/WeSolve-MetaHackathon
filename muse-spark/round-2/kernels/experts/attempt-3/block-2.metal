// Shared header for hyp-1-experts-exact hardening attempt-3 (builder-1-c).
// qvm_block: byte-identical copy of attempt-1's helper (itself a copy of
// muse-spark/kernels/experts/attempt-4/block-1.metal): bit-faithful
// replica of MLX affine_gather_qvm for one 32-output block — each lane
// accumulates its strided K elements serially in float32, dequantizing
// on the fly as x * (scale * q + bias), then a simd_sum tree reduces
// across the 32 lanes. Used below 60 rows.
// qmm_accum: byte-identical copy of attempt-1's prefill helper
// (muse-spark/round-2/kernels/experts/attempt-1/block-1.metal):
// scalar mimic of affine_gather_qmm_rhs — weight rounded to bfloat16
// first, then K accumulated contiguously in BK=32 tile order by a
// single thread per output. Used at 60 rows and above. No MMA tiles,
// so no extra threadgroup tile memory is allocated in either branch.
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

// silu(gate) * up on bfloat16 values, computed entirely in bfloat16
// after every op: attempt-4's proven lowering of the compiled swiglu
// (muse-spark/kernels/experts/attempt-4/block-0.metal:53-64), shared
// by both size branches, as in attempt-1 and attempt-2.
inline bfloat silu_mul_bf16(bfloat gb, bfloat ub) {
  float g = (float)gb;
  bfloat e1 = (bfloat)metal::precise::exp(metal::abs(g));
  bfloat d = (bfloat)(1.0f + (float)e1);
  bfloat y = (bfloat)(1.0f / (float)d);
  bfloat s = (g < 0.0f) ? y : (bfloat)(1.0f - (float)y);
  bfloat p = (bfloat)(g * (float)s);
  return (bfloat)((float)p * (float)ub);
}
