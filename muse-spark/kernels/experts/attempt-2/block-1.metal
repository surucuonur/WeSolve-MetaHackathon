// Bit-faithful replica of MLX affine_gather_qvm's per-output computation
// (quantized.h qvm_impl + qouter, bits=4 path) for one 32-output block:
// each lane accumulates its strided K elements serially, then a simd_sum tree
// reduces across the 32 lanes, exactly as the reference grid does.
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
