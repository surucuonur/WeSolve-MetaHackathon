// hyp-2-experts-prefill direction (e), round-3 attempt-5 (builder-2-e).
// K0 gate_up_small: reference-shaped unfused gate/up for rows == 1.
// One 128-thread group per (pair, 32-output block): 48 groups at 1 row.
// Each of the 4 simdgroups computes 8 gate + 8 up outputs of its block
// with qvm_block's exact per-output arithmetic (k = lane, lane+32, ...
// serial, same x*(s*q+bias) terms, same simd_sum tree), rounds to
// bfloat16, runs the fused kernel's exact silu-mul op sequence, and
// stores h. At rows != 1 every thread returns immediately (uniform).
uint gid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
if (x_shape[0] != 1u) { return; }
uint pair = gid / 12u;
uint b = gid % 12u;
uint r = pair >> 2u;
uint e = inds[pair];

threadgroup bfloat tg_x[768];
for (uint i = tid; i < 768u; i += 128u) {
  tg_x[i] = x[r * 768u + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;
uint g = b;
uint wb = b * 4u;
const device uint* Wg = gate_w + eW;
const device bfloat* Sg = gate_scales + eS;
const device bfloat* Bg = gate_biases + eS;
const device uint* Wu = up_w + eW;
const device bfloat* Su = up_scales + eS;
const device bfloat* Bu = up_biases + eS;
float pg[8];
float pu[8];
for (uint j = 0; j < 8u; ++j) {
  pg[j] = 0.0f;
  pu[j] = 0.0f;
}
for (uint k = lane; k < 768u; k += 32u) {
  float xv = (float)tg_x[k];
  float scale_g = (float)Sg[k * 12u + g];
  float bias_g = (float)Bg[k * 12u + g];
  float s0g = scale_g;
  float s1g = scale_g / 16.0f;
  uint wg = Wg[(ulong)k * 48ul + (ulong)(wb + sg)];
  float scale_u = (float)Su[k * 12u + g];
  float bias_u = (float)Bu[k * 12u + g];
  float s0u = scale_u;
  float s1u = scale_u / 16.0f;
  uint wu = Wu[(ulong)k * 48ul + (ulong)(wb + sg)];
  for (uint t = 0; t < 4u; ++t) {
    uint by = (wg >> (t * 8u)) & 0xffu;
    pg[2u * t] += xv * (s0g * (float)(by & 0x0fu) + bias_g);
    pg[2u * t + 1u] += xv * (s1g * (float)(by & 0xf0u) + bias_g);
  }
  for (uint t = 0; t < 4u; ++t) {
    uint by = (wu >> (t * 8u)) & 0xffu;
    pu[2u * t] += xv * (s0u * (float)(by & 0x0fu) + bias_u);
    pu[2u * t + 1u] += xv * (s1u * (float)(by & 0xf0u) + bias_u);
  }
}
#pragma clang loop unroll(full)
for (uint j = 0; j < 8u; ++j) {
  pg[j] = simd_sum(pg[j]);
}
#pragma clang loop unroll(full)
for (uint j = 0; j < 8u; ++j) {
  pu[j] = simd_sum(pu[j]);
}
if (lane < 8u) {
  float gv = (float)((bfloat)pg[lane]);
  float uv = (float)((bfloat)pu[lane]);
  bfloat e1 = (bfloat)metal::precise::exp(metal::abs(gv));
  bfloat d = (bfloat)(1.0f + (float)e1);
  bfloat yy = (bfloat)(1.0f / (float)d);
  bfloat ss = (gv < 0.0f) ? yy : (bfloat)(1.0f - (float)yy);
  bfloat pp = (bfloat)(gv * (float)ss);
  h_small[pair * 384u + b * 32u + sg * 8u + lane] = (bfloat)((float)pp * uv);
}
