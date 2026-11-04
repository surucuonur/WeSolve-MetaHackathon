// Kernel GU (builder-1-a, hyp-1-experts-exact direction a): gate+up+silu
// for one (pair, 32-output block) per 32-thread threadgroup, writing
// hidden[pair, block*32 + lane]. Grid rows*4*12 groups: at 1 row 48
// threadgroups instead of the starting point's 4, so a chained 1-row
// call fills the GPU. Size branch on rows (from inds_shape[0]): below
// 60 the attempt-4 qvm arithmetic (qvm_block + full-bf16 silu); at 60
// and above the gather_qmm_rhs arithmetic (qmm_block). Both keep the
// per-block accumulation order bit-identical to their reference.
uint tg = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint rows = (uint)inds_shape[0];
uint pair = tg / 12u;
uint blk = tg % 12u;
uint r = pair >> 2;
uint e = inds[pair];
ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;

threadgroup bfloat tg_x[768];
threadgroup bfloat tg_gate[32];
threadgroup bfloat tg_up[32];
threadgroup float qXs[256];
threadgroup float qWs[1024];
threadgroup float qTmp[256];

if (rows < 60u) {
  for (uint i = lane; i < 768u; i += 32u) {
    tg_x[i] = x[r * 768u + i];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float p[32];
  qvm_block(gate_w + eW, gate_scales + eS, gate_biases + eS, tg_x, 768u,
            12u, 48u, blk, blk * 4u, lane, p);
  if (lane == 0u) {
    for (uint j = 0; j < 32u; ++j) {
      tg_gate[j] = (bfloat)p[j];
    }
  }
  qvm_block(up_w + eW, up_scales + eS, up_biases + eS, tg_x, 768u, 12u,
            48u, blk, blk * 4u, lane, p);
  if (lane == 0u) {
    for (uint j = 0; j < 32u; ++j) {
      tg_up[j] = (bfloat)p[j];
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  hidden[pair * 384u + blk * 32u + lane] =
      silu_mul_bf16(tg_gate[lane], tg_up[lane]);
} else {
  float og[32];
  float ou[32];
  qmm_block(gate_w + eW, gate_scales + eS, gate_biases + eS, x + r * 768u,
            768u, 12u, 48u, blk, lane, qXs, qWs, qTmp, og);
  qmm_block(up_w + eW, up_scales + eS, up_biases + eS, x + r * 768u, 768u,
            12u, 48u, blk, lane, qXs, qWs, qTmp, ou);
  hidden[pair * 384u + blk * 32u + lane] =
      silu_mul_bf16((bfloat)og[lane], (bfloat)ou[lane]);
}
