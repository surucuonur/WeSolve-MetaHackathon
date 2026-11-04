// hyp-2-experts-prefill direction (e), round-3 attempt-5 (builder-2-e).
// K1 down_small: reference-shaped unfused down for rows == 1.
// One 128-thread group per (pair, 32-output block): 96 groups at 1 row.
// Each of the 4 simdgroups computes 8 down outputs of its block with
// qvm_block's exact per-output arithmetic (k = lane, lane+32, ...
// serial over K=384, same x*(s*q+bias) terms, same simd_sum tree) and
// stores bfloat16. Reads h from K0's h_small. At rows != 1 every
// thread returns immediately (uniform).
uint gid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
if (inds_shape[0] != 1u) { return; }
uint pair = gid / 24u;
uint b = gid % 24u;
uint e = inds[pair];

threadgroup bfloat tg_h[384];
for (uint i = tid; i < 384u; i += 128u) {
  tg_h[i] = h_small[pair * 384u + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;
uint g = b;
uint wb = b * 4u;
const device uint* W = down_w + eW;
const device bfloat* S = down_scales + eS;
const device bfloat* B = down_biases + eS;
float pd[8];
for (uint j = 0; j < 8u; ++j) {
  pd[j] = 0.0f;
}
for (uint k = lane; k < 384u; k += 32u) {
  float hv = (float)tg_h[k];
  float scale = (float)S[k * 24u + g];
  float bias = (float)B[k * 24u + g];
  float s0 = scale;
  float s1 = scale / 16.0f;
  uint w = W[(ulong)k * 96ul + (ulong)(wb + sg)];
  for (uint t = 0; t < 4u; ++t) {
    uint by = (w >> (t * 8u)) & 0xffu;
    pd[2u * t] += hv * (s0 * (float)(by & 0x0fu) + bias);
    pd[2u * t + 1u] += hv * (s1 * (float)(by & 0xf0u) + bias);
  }
}
#pragma clang loop unroll(full)
for (uint j = 0; j < 8u; ++j) {
  pd[j] = simd_sum(pd[j]);
}
if (lane < 8u) {
  y_small[pair * 768u + b * 32u + sg * 8u + lane] = (bfloat)pd[lane];
}
