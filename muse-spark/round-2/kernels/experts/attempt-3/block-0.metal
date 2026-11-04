// Kernel GU (builder-1-c, hyp-1-experts-exact hardening attempt-3):
// gate+up+silu for 4 distinct 32-output blocks per 128-thread
// threadgroup (4 simdgroups, one block per simdgroup), writing
// hidden[pair, blk*32 + lane]. Grid rows*1536 threads = rows*12
// groups (3 groups per pair x 4 blocks = 12 blocks/pair); at 1 row
// 12 threadgroups instead of attempt-1's 4 fused groups, without
// attempt-2's 1-simdgroup-per-group collapse. Size branch on rows
// (from inds_shape[0]): below 60 the qvm arithmetic (qvm_block +
// full-bf16 silu, per-block strided-lane + simd_sum order identical
// to attempt-1/attempt-4); at 60 and above attempt-1's scalar
// qmm_accum (bfloat16-rounded dequant + BK=32 serial order). Both
// keep the four bfloat16 rounding points and positional order.
uint tg = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint simd = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint rows = (uint)inds_shape[0];
bool prefill = rows >= 60u;
uint pair = tg / 3u;
uint chunk = tg % 3u;
uint blk = chunk * 4u + simd;
uint r = pair >> 2;
uint e = inds[pair];
ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;

threadgroup bfloat tg_x[768];
threadgroup bfloat tg_gate[128];
threadgroup bfloat tg_up[128];

for (uint i = tid; i < 768u; i += 128u) {
  tg_x[i] = x[r * 768u + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

if (!prefill) {
  float p[32];
  qvm_block(gate_w + eW, gate_scales + eS, gate_biases + eS, tg_x, 768u,
            12u, 48u, blk, blk * 4u, lane, p);
  if (lane == 0u) {
    for (uint j = 0; j < 32u; ++j) {
      tg_gate[simd * 32u + j] = (bfloat)p[j];
    }
  }
  qvm_block(up_w + eW, up_scales + eS, up_biases + eS, tg_x, 768u, 12u,
            48u, blk, blk * 4u, lane, p);
  if (lane == 0u) {
    for (uint j = 0; j < 32u; ++j) {
      tg_up[simd * 32u + j] = (bfloat)p[j];
    }
  }
} else {
  float ag = qmm_accum(gate_w + eW, gate_scales + eS, gate_biases + eS,
                       tg_x, 768u, 12u, 48u, blk, blk * 4u, lane);
  tg_gate[simd * 32u + lane] = (bfloat)ag;
  float au = qmm_accum(up_w + eW, up_scales + eS, up_biases + eS, tg_x,
                       768u, 12u, 48u, blk, blk * 4u, lane);
  tg_up[simd * 32u + lane] = (bfloat)au;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

hidden[pair * 384u + blk * 32u + lane] =
    silu_mul_bf16(tg_gate[simd * 32u + lane], tg_up[simd * 32u + lane]);
