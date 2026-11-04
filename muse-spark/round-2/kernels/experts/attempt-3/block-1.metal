// Kernel Down (builder-1-c, hyp-1-experts-exact hardening attempt-3):
// down product for 4 distinct 32-output blocks per 128-thread
// threadgroup (4 simdgroups, one block per simdgroup), reading
// hidden from kernel GU. Grid rows*3072 threads = rows*24 groups
// (6 groups per pair x 4 blocks = 24 blocks/pair); at 1 row 24
// threadgroups. Same size branch as GU: qvm_block below 60 rows,
// attempt-1's scalar qmm_accum at 60 and above, each bit-identical
// in per-block accumulation order to its reference kernel.
uint tg = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint simd = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint rows = (uint)inds_shape[0];
bool prefill = rows >= 60u;
uint pair = tg / 6u;
uint chunk = tg % 6u;
uint blk = chunk * 4u + simd;
uint e = inds[pair];
ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;

threadgroup bfloat tg_h[384];

for (uint i = tid; i < 384u; i += 128u) {
  tg_h[i] = hidden[pair * 384u + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

if (!prefill) {
  float p[32];
  qvm_block(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u,
            24u, 96u, blk, blk * 4u, lane, p);
  if (lane == 0u) {
    for (uint j = 0; j < 32u; ++j) {
      y[pair * 768u + blk * 32u + j] = (bfloat)p[j];
    }
  }
} else {
  float acc = qmm_accum(down_w + eW, down_scales + eS, down_biases + eS,
                        tg_h, 384u, 24u, 96u, blk, blk * 4u, lane);
  y[pair * 768u + blk * 32u + lane] = (bfloat)acc;
}
