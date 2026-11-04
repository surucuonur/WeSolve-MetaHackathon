// Kernel Down (builder-1-a, hyp-1-experts-exact direction a): down
// product for one (pair, 32-output block) per 32-thread threadgroup,
// reading hidden from kernel GU. Grid rows*4*24 groups: 96
// threadgroups at 1 row. Same size branch as GU: qvm_block below 60
// rows, qmm_block at 60 and above, each bit-identical in per-block
// accumulation order to its reference kernel.
uint tg = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint rows = (uint)inds_shape[0];
uint pair = tg / 24u;
uint blk = tg % 24u;
uint e = inds[pair];
ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;

threadgroup bfloat tg_h[384];
threadgroup float qXs[256];
threadgroup float qWs[1024];
threadgroup float qTmp[256];

if (rows < 60u) {
  for (uint i = lane; i < 384u; i += 32u) {
    tg_h[i] = hidden[pair * 384u + i];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float p[32];
  qvm_block(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u,
            24u, 96u, blk, blk * 4u, lane, p);
  if (lane == 0u) {
    for (uint j = 0; j < 32u; ++j) {
      y[pair * 768u + blk * 32u + j] = (bfloat)p[j];
    }
  }
} else {
  float o[32];
  qmm_block(down_w + eW, down_scales + eS, down_biases + eS,
            hidden + pair * 384u, 384u, 24u, 96u, blk, lane, qXs, qWs,
            qTmp, o);
  y[pair * 768u + blk * 32u + lane] = (bfloat)o[lane];
}
