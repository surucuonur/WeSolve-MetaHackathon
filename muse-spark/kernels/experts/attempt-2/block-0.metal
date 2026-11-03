// hyp-1 direction (b), v2: one threadgroup per (row, expert-slot) pair.
// Fuses gate/up/down + silu-mul with intermediates in threadgroup memory,
// keeping per-pair weight reads (no cross-pair batching, no sort).
// Numerics replicate the reference exactly: each product uses qvm's
// strided-lane accumulation + simd_sum reduction, and silu uses MLX's
// Sigmoid lowering in fp32. Grid: rows*4 pairs x 128 threads.
uint pair = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint simd = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint r = pair >> 2;
uint e = inds[pair];

threadgroup bfloat tg_x[768];
threadgroup bfloat tg_gate[384];
threadgroup bfloat tg_up[384];
threadgroup bfloat tg_h[384];

for (uint i = tid; i < 768u; i += 128u) {
  tg_x[i] = x[r * 768u + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

ulong eW = (ulong)e * 36864ul;
ulong eS = (ulong)e * 9216ul;
float p[32];

// gate: 12 blocks of 32 outputs in 3 waves of 4 simdgroups.
for (uint v = 0; v < 3u; ++v) {
  uint b = v * 4u + simd;
  qvm_block(gate_w + eW, gate_scales + eS, gate_biases + eS, tg_x, 768u, 12u, 48u, b, b * 4u, lane, p);
  if (lane == 0) {
    for (uint j = 0; j < 32u; ++j) {
      tg_gate[b * 32u + j] = (bfloat)p[j];
    }
  }
}
// up: same structure.
for (uint v = 0; v < 3u; ++v) {
  uint b = v * 4u + simd;
  qvm_block(up_w + eW, up_scales + eS, up_biases + eS, tg_x, 768u, 12u, 48u, b, b * 4u, lane, p);
  if (lane == 0) {
    for (uint j = 0; j < 32u; ++j) {
      tg_up[b * 32u + j] = (bfloat)p[j];
    }
  }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

// silu-mul with MLX's Sigmoid lowering computed ENTIRELY in bf16: the
// compiled swiglu rounds after every op (proven by CPU search over all 64
// rounding placements: only full-bf16 matches the reference, bad=0).
for (uint t = tid; t < 384u; t += 128u) {
  float g = (float)tg_gate[t];
  bfloat e1 = (bfloat)metal::precise::exp(metal::abs(g));
  bfloat d = (bfloat)(1.0f + (float)e1);
  bfloat y = (bfloat)(1.0f / (float)d);
  bfloat s = (g < 0.0f) ? y : (bfloat)(1.0f - (float)y);
  bfloat p = (bfloat)(g * (float)s);
  tg_h[t] = (bfloat)((float)p * (float)tg_up[t]);
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);

// down: 24 blocks of 32 outputs in 6 waves.
for (uint v = 0; v < 6u; ++v) {
  uint b = v * 4u + simd;
  qvm_block(down_w + eW, down_scales + eS, down_biases + eS, tg_h, 384u, 24u, 96u, b, b * 4u, lane, p);
  if (lane == 0) {
    for (uint j = 0; j < 32u; ++j) {
      y[pair * 768u + b * 32u + j] = (bfloat)p[j];
    }
  }
}
