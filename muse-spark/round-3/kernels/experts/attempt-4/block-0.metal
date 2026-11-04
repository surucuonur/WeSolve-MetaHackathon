// hyp-2-experts-prefill direction (c), round-3 attempt-3 (builder-2-c).
// Decode kernel: round-2 attempt-7's fused body kept for the decode path,
// with an early return above 60 rows (prefill runs in the grouped kernel,
// block-2/3) and the output renamed y -> y_decode for the select kernel.
// All decode lines below the return are byte-identical to attempt-7.
// Builds on attempt-5 (best: fused single kernel, 512 threads/pair,
// interleaved gate/up + parallel block stores, G2 pass all six,
// 92.1/162.0/534.8 us/layer) with attempt-6's down-interleave made
// to compile: one simdgroup computes two down blocks (2*simd,
// 2*simd+1) in one merged K loop (qvm_down_pair), per-block order
// + parallel stores kept. Attempt-6 failed G1 (512 > pipeline max
// 448) with three distinct 32-float thread arrays (p, gate/up's
// scoped pu, down's scoped q). This attempt shares ONE kernel-scope
// second accumulator q[32] between gate/up's pu and down's second
// block (disjoint phases, separated by barriers, each helper
// re-zeroes both accumulators on entry), restoring attempt-5's
// two-array footprint at unchanged 512 threads. All accumulation
// orders, silu, prefill branch and rounding points identical to
// attempt-5/6. Grid: rows*4 pairs x 512 threads.
uint pair = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint simd = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint r = pair >> 2;
uint e = inds[pair];
bool prefill = x_shape[0] >= 60;
if (prefill) return;

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
// Shared accumulators: p is block 0 of the current pair, q is the
// second accumulator. Gate/up uses (p,q) as (gate,up); down uses
// (p,q) as (block 2s, block 2s+1). The phases are separated by
// threadgroup barriers and each helper re-zeroes both arrays, so
// sharing one kernel-scope q (instead of a scoped pu plus a scoped
// q as in attempt-6) is a pure storage rename with identical bits.
float p[32];
float q[32];

if (!prefill) {
  // gate+up interleaved: 12 blocks of 32 outputs, one wave of 16
  // simdgroups (4 idle). Simdgroup b computes block b of both gate
  // and up in a single K loop (qvm_gate_up_pair): for each strided
  // k both blocks' loads are issued before accumulating, keeping
  // each block's qvm_block add order and simd_sum identical.
  if (simd < 12u) {
    uint b = simd;
    qvm_gate_up_pair(gate_w + eW, gate_scales + eS, gate_biases + eS,
                     up_w + eW, up_scales + eS, up_biases + eS,
                     tg_x, 768u, 12u, 48u, b, b * 4u, lane, p, q);
    tg_gate[b * 32u + lane] = (bfloat)p[lane];
    tg_up[b * 32u + lane] = (bfloat)q[lane];
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
  // down interleaved: 24 blocks as 12 pairs, one wave of 16
  // simdgroups (4 idle). Simdgroup s computes blocks 2s and 2s+1
  // in a single K loop (qvm_down_pair): for each strided k both
  // blocks' loads are issued before accumulating, keeping each
  // block's qvm_block add order and simd_sum identical. Stores
  // stay parallel: lane j stores element j of each block.
  if (simd < 12u) {
    uint b0 = 2u * simd;
    uint b1 = b0 + 1u;
    qvm_down_pair(down_w + eW, down_scales + eS, down_biases + eS,
                  tg_h, 384u, 24u, 96u, b0, b0 * 4u, b1, b1 * 4u, lane, p, q);
    y_decode[pair * 768u + b0 * 32u + lane] = (bfloat)p[lane];
    y_decode[pair * 768u + b1 * 32u + lane] = (bfloat)q[lane];
  }
} else {
  // down: 24 blocks in 2 waves, one output per lane.
  for (uint v = 0; v < 2u; ++v) {
    uint b = v * 16u + simd;
    if (b < 24u) {
      float acc = qmm_accum(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u, 24u, 96u, b, b * 4u, lane);
      y_decode[pair * 768u + b * 32u + lane] = (bfloat)acc;
    }
  }
}
