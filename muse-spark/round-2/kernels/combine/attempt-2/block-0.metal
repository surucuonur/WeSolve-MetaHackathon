// Fused combine, exact (hyp-3-combine-exact, direction (b)): one
// threadgroup (192 threads) per row. Weights (4 f32) and shared (768 bf16,
// expanded exactly to f32) are staged in threadgroup memory with
// vectorized 8-byte loads, then each thread weights its 4 consecutive
// outputs. Grid is (rows*192, 1, 1), threadgroup (192, 1, 1).
//
// Reference (slots.py:51-52): (y * weights[..., None]).sum(axis=-2)
//   .astype(bf16) + shared, run eagerly as separate kernels, so each
//   product is rounded once to f32 and the fold ((p0+p1)+p2)+p3 rounds
//   after every add. A single-expression sum lets the compiler contract
//   mul+add pairs into FMAs (unrounded products) and breaks bit-exactness
//   (starting point: 0/2/2 differ at decode 1/8/32 rows). Here every
//   product and every partial sum is assigned in its own statement and
//   passed through a volatile register round-trip, which forces the f32
//   rounding to materialize; no FMA can form across it. Both bf16
//   roundings are kept: bfloat(acc) is the astype, then the bf16 add of
//   shared (the staged float is the exact f32 expansion of the bf16
//   value, so bfloat(shsh) recovers it bit for bit).

uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x; // 0..191

threadgroup float wsh[4];
threadgroup float shsh[768];

if (tid < 4u) {
  wsh[tid] = weights[row * 4u + tid];
}

// Stage shared[row]: 768 bf16 = 192 uint64, one vectorized load per thread.
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
// quad tid of each expert: its 4 outputs, unpacked bit-exactly to f32.
uint base = tid * 4u;
const device uint64_t* yp = (const device uint64_t*)(y + row * 3072u);
uint64_t q0 = yp[tid];
uint64_t q1 = yp[192u + tid];
uint64_t q2 = yp[384u + tid];
uint64_t q3 = yp[576u + tid];

float yv[4][4];
for (uint k = 0u; k < 4u; k++) {
  uint sh = k * 16u;
  yv[0][k] = as_type<float>((uint((q0 >> sh) & 0xFFFFull)) << 16u);
  yv[1][k] = as_type<float>((uint((q1 >> sh) & 0xFFFFull)) << 16u);
  yv[2][k] = as_type<float>((uint((q2 >> sh) & 0xFFFFull)) << 16u);
  yv[3][k] = as_type<float>((uint((q3 >> sh) & 0xFFFFull)) << 16u);
}

uint obase = row * 768u + base;
for (uint k = 0u; k < 4u; k++) {
  // Products in separate statements, each forced through a volatile
  // round-trip so it is rounded to f32 before any add can see it.
  float p0 = yv[0][k] * w0;
  { volatile float b = p0; p0 = b; }
  float p1 = yv[1][k] * w1;
  { volatile float b = p1; p1 = b; }
  float p2 = yv[2][k] * w2;
  { volatile float b = p2; p2 = b; }
  float p3 = yv[3][k] * w3;
  { volatile float b = p3; p3 = b; }
  // Left fold ((p0+p1)+p2)+p3, rounding after every add.
  float t = p0 + p1;
  { volatile float b = t; t = b; }
  t = t + p2;
  { volatile float b = t; t = b; }
  float acc = t + p3;
  out[obase + k] = bfloat(acc) + bfloat(shsh[base + k]);
}
