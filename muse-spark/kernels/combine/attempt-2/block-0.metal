// Fused combine, direction (b): one threadgroup (192 threads) per row.
// Cooperatively stage weights (4 f32) and shared (768 bf16, bit-exact to
// f32) into threadgroup memory with vectorized 8-byte loads, then weight,
// sum in fp32, round once to bf16, and bf16-add shared. Grid is
// (rows*192, 1, 1), threadgroup (192, 1, 1); thread t owns outputs t*4..t*4+3.
//
// Reference (slots.py:51-52): (y * weights[..., None]).sum(axis=-2)
//   .astype(bf16) + shared. y is [rows,4,768] bf16 row-contiguous, weights
//   [rows,4] f32, shared [rows,768] bf16, out [rows,768] bf16. Both bf16
//   rounding points are kept: bfloat(acc) is the astype, and the + shared
//   uses the same single-rounding bf16 add as MLX's binary_vv (as in
//   attempt-1). The 4-term fp32 sum order is free per the hyp-3 boundary.

uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x; // 0..191

threadgroup float wsh[4];
threadgroup float shsh[768];

if (tid < 4u) {
  wsh[tid] = weights[row * 4u + tid];
}

// Stage shared[row]: 768 bf16 = 192 uint64, one vectorized load per thread.
// Row base is row*1536 bytes, 8-byte aligned.
{
  const device uint64_t* sp = (const device uint64_t*)(shared + row * 768u);
  uint64_t q = sp[tid];
  uint base = tid * 4u;
  shsh[base + 0u] = as_type<float>((uint(q & 0xFFFFull)) << 16u);
  shsh[base + 1u] = as_type<float>((uint((q >> 16) & 0xFFFFull)) << 16u);
  shsh[base + 2u] = as_type<float>((uint((q >> 32) & 0xFFFFull)) << 16u);
  shsh[base + 3u] = as_type<float>((uint((q >> 48) & 0xFFFFull)) << 16u);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float w0 = wsh[0];
float w1 = wsh[1];
float w2 = wsh[2];
float w3 = wsh[3];

// y[row] is 4*768 bf16 = 768 uint64; expert j owns 192. Thread tid loads
// quad tid of each expert (4 consecutive bf16 = its 4 outputs), unpacks
// bit-exactly to fp32, weights and accumulates: per element the sum runs
// j = 0..3, the reference's natural order.
uint base = tid * 4u;
const device uint64_t* yp = (const device uint64_t*)(y + row * 3072u);
float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
for (uint j = 0u; j < 4u; j++) {
  uint64_t q = yp[j * 192u + tid];
  float wj = j == 0u ? w0 : (j == 1u ? w1 : (j == 2u ? w2 : w3));
  acc0 += as_type<float>((uint(q & 0xFFFFull)) << 16u) * wj;
  acc1 += as_type<float>((uint((q >> 16) & 0xFFFFull)) << 16u) * wj;
  acc2 += as_type<float>((uint((q >> 32) & 0xFFFFull)) << 16u) * wj;
  acc3 += as_type<float>((uint((q >> 48) & 0xFFFFull)) << 16u) * wj;
}

uint obase = row * 768u + base;
out[obase + 0u] = bfloat(acc0) + bfloat(shsh[base + 0u]);
out[obase + 1u] = bfloat(acc1) + bfloat(shsh[base + 1u]);
out[obase + 2u] = bfloat(acc2) + bfloat(shsh[base + 2u]);
out[obase + 3u] = bfloat(acc3) + bfloat(shsh[base + 3u]);
