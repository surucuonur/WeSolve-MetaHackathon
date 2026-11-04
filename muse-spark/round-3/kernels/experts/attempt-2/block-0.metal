// hyp-2-experts-prefill, round-3 attempt-2 (builder-2-b).
// Builds on muse-spark/round-2/kernels/experts/attempt-7 (fused single
// kernel, 512 threads/pair, G2 pass all six). One main change: the
// rows>=60 prefill branch is vectorized per weight word — one thread
// computes a word's 8 outputs with 8 accumulators (qmm_accum8), each
// output keeping the scalar branch's contiguous BK=32 bfloat16-rounded
// order. The fused decode path (qvm branches, silu, dispatch,
// accumulators, grid) is byte-identical to attempt-7. Grid: rows*4
// pairs x 512 threads.
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
  // gate: 12 blocks as 48 word-units, packed 8 blocks per simdgroup
  // (block = simd*8 + (lane>>2), word = lane&3), one wave of 2
  // simdgroups (16 lanes idle in the second). Each thread runs
  // qmm_accum8 into p[0..7] (kernel-scope p reused, disjoint phase,
  // re-zeroed on entry — no new thread array, same 512-thread
  // pipeline budget as attempt-7), then stores its word's 8 outputs.
  if (simd < 2u) {
    uint b = simd * 8u + (lane >> 2u);
    if (b < 12u) {
      uint word = b * 4u + (lane & 3u);
      qmm_accum8(gate_w + eW, gate_scales + eS, gate_biases + eS, tg_x, 768u, 12u, 48u, b, word, p);
      for (uint j = 0u; j < 8u; ++j) {
        tg_gate[b * 32u + ((lane & 3u) * 8u) + j] = (bfloat)p[j];
      }
    }
  }
  // up: same structure.
  if (simd < 2u) {
    uint b = simd * 8u + (lane >> 2u);
    if (b < 12u) {
      uint word = b * 4u + (lane & 3u);
      qmm_accum8(up_w + eW, up_scales + eS, up_biases + eS, tg_x, 768u, 12u, 48u, b, word, p);
      for (uint j = 0u; j < 8u; ++j) {
        tg_up[b * 32u + ((lane & 3u) * 8u) + j] = (bfloat)p[j];
      }
    }
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
    y[pair * 768u + b0 * 32u + lane] = (bfloat)p[lane];
    y[pair * 768u + b1 * 32u + lane] = (bfloat)q[lane];
  }
} else {
  // down: 24 blocks as 96 word-units, one wave of 3 simdgroups, all
  // 96 threads active (block = simd*8 + (lane>>2), word = lane&3).
  if (simd < 3u) {
    uint b = simd * 8u + (lane >> 2u);
    uint word = b * 4u + (lane & 3u);
    qmm_accum8(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u, 24u, 96u, b, word, p);
    for (uint j = 0u; j < 8u; ++j) {
      y[pair * 768u + b * 32u + ((lane & 3u) * 8u) + j] = (bfloat)p[j];
    }
  }
}
