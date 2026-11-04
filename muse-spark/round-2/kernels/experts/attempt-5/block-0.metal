// hyp-1-experts-exact hardening, round-2 attempt-5 (builder-1-e).
// Builds on attempt-4: single fused per-pair kernel, 512 threads/pair,
// branched in-Metal on rows >= 60, decode gate/up interleaved
// (qvm_gate_up_pair). One main change (decode only):
//  - Parallel block stores: after each simd_sum every lane holds the
//    full reduced 32-vector, so lane j stores element j (one store
//    per lane, 32-way parallel) for gate, up and down, instead of
//    lane 0 serially storing all 32 elements per block.
//  - rows >= 60 (prefill): unchanged — already one output per lane.
// Same qvm orders, interleave, silu, prefill branch and grids.
// Grid: rows*4 pairs x 512 threads.
uint pair = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint simd = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint r = pair >> 2;
uint e = inds[pair];
bool prefill = x_shape[0] >= 60;

threadgroup bfloat tg_x[768];
threadgroup bfloat tg_gate[384];
threadgroup bfloat tg_up[384];
threadgroup bfloat tg_h[384];

for (uint i = tid; i < 768u; i += 512u) {
  tg_x[i] = x[r * 768u + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;
float p[32];

if (!prefill) {
  // gate+up interleaved: 12 blocks of 32 outputs, one wave of 16
  // simdgroups (4 idle). Simdgroup b computes block b of both gate
  // and up in a single K loop (qvm_gate_up_pair): for each strided
  // k both blocks' loads are issued before accumulating, keeping
  // each block's qvm_block add order and simd_sum identical.
  if (simd < 12u) {
    uint b = simd;
    float pu[32];
    qvm_gate_up_pair(gate_w + eW, gate_scales + eS, gate_biases + eS,
                     up_w + eW, up_scales + eS, up_biases + eS,
                     tg_x, 768u, 12u, 48u, b, b * 4u, lane, p, pu);
    tg_gate[b * 32u + lane] = (bfloat)p[lane];
    tg_up[b * 32u + lane] = (bfloat)pu[lane];
  }
} else {
  // gate: 12 blocks, one output per lane, contiguous BK=32 order.
  if (simd < 12u) {
    uint b = simd;
    float acc = qmm_accum(gate_w + eW, gate_scales + eS, gate_biases + eS, tg_x, 768u, 12u, 48u, b, b * 4u, lane);
    tg_gate[b * 32u + lane] = (bfloat)acc;
  }
  // up: same structure.
  if (simd < 12u) {
    uint b = simd;
    float acc = qmm_accum(up_w + eW, up_scales + eS, up_biases + eS, tg_x, 768u, 12u, 48u, b, b * 4u, lane);
    tg_up[b * 32u + lane] = (bfloat)acc;
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

// silu-mul with MLX's Sigmoid lowering computed ENTIRELY in bf16 (the
// compiled swiglu; identical in both branches, as in attempt-4).
for (uint t = tid; t < 384u; t += 512u) {
  float g = (float)tg_gate[t];
  bfloat e1 = (bfloat)metal::precise::exp(metal::abs(g));
  bfloat d = (bfloat)(1.0f + (float)e1);
  bfloat y = (bfloat)(1.0f / (float)d);
  bfloat s = (g < 0.0f) ? y : (bfloat)(1.0f - (float)y);
  bfloat p = (bfloat)(g * (float)s);
  tg_h[t] = (bfloat)((float)p * (float)tg_up[t]);
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

if (!prefill) {
  // down: 24 blocks of 32 outputs in 2 waves of 16 simdgroups.
  for (uint v = 0; v < 2u; ++v) {
    uint b = v * 16u + simd;
    if (b < 24u) {
      qvm_block(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u, 24u, 96u, b, b * 4u, lane, p);
      y[pair * 768u + b * 32u + lane] = (bfloat)p[lane];
    }
  }
} else {
  // down: 24 blocks in 2 waves, one output per lane.
  for (uint v = 0; v < 2u; ++v) {
    uint b = v * 16u + simd;
    if (b < 24u) {
      float acc = qmm_accum(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u, 24u, 96u, b, b * 4u, lane);
      y[pair * 768u + b * 32u + lane] = (bfloat)acc;
    }
  }
}
