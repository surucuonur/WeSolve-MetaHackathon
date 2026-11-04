"""Probe: which barrier idiom suppresses FMA contraction in the staged
threadgroup-per-row combine kernel? Compares each variant bit-for-bit
against MLX's eager ref_combine on random + wide-scale data.
Scratch only; never graded.
"""
import sys
import mlx.core as mx

# Common staged skeleton (round-1 attempt-2 shape): 192 threads/row,
# weights+shared staged in threadgroup memory, one barrier, then each
# thread owns 4 consecutive outputs. ARITH is substituted per variant.
SKELETON = r"""
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;

threadgroup float wsh[4];
threadgroup float shsh[768];

if (tid < 4u) {
  wsh[tid] = weights[row * 4u + tid];
}
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
  float acc;
  ARITH
  out[obase + k] = bfloat(acc) + bfloat(shsh[base + k]);
}
"""

# (a) control: plain separate statements, no barrier (expect contraction)
ARITH_NAIVE = r"""
  float p0 = yv[0][k] * w0;
  float p1 = yv[1][k] * w1;
  float p2 = yv[2][k] * w2;
  float p3 = yv[3][k] * w3;
  float t = p0 + p1;
  t = t + p2;
  acc = t + p3;
"""

# (b) volatile register round-trip after every product and partial sum
ARITH_VOL = r"""
  float p0 = yv[0][k] * w0;
  { volatile float b = p0; p0 = b; }
  float p1 = yv[1][k] * w1;
  { volatile float b = p1; p1 = b; }
  float p2 = yv[2][k] * w2;
  { volatile float b = p2; p2 = b; }
  float p3 = yv[3][k] * w3;
  { volatile float b = p3; p3 = b; }
  float t = p0 + p1;
  { volatile float b = t; t = b; }
  t = t + p2;
  { volatile float b = t; t = b; }
  acc = t + p3;
"""

# (c) register array + asm compiler barrier (may not compile in MSL)
ARITH_ASM = r"""
  float p[4];
  p[0] = yv[0][k] * w0;
  p[1] = yv[1][k] * w1;
  p[2] = yv[2][k] * w2;
  p[3] = yv[3][k] * w3;
  asm volatile("" : "+r"(p[0]), "+r"(p[1]), "+r"(p[2]), "+r"(p[3]) :: "memory");
  float t = p[0] + p[1];
  asm volatile("" : "+r"(t) :: "memory");
  t = t + p[2];
  asm volatile("" : "+r"(t) :: "memory");
  acc = t + p[3];
"""

VARIANTS = {"naive": ARITH_NAIVE, "volatile": ARITH_VOL, "asm": ARITH_ASM}


def build(arith):
    src = SKELETON.replace("ARITH", arith)
    return mx.fast.metal_kernel(
        name="probe_combine",
        input_names=["y", "weights", "shared"],
        output_names=["out"],
        source=src,
    )


def run(fn, y, w, s):
    rows = y.shape[0]
    out = fn(inputs=[y, w, s], grid=(rows * 192, 1, 1), threadgroup=(192, 1, 1),
             output_shapes=[(rows, 768)], output_dtypes=[mx.bfloat16])[0]
    mx.eval(out)
    return out


def ref(y, w, s):
    r = (y * w[..., None]).sum(axis=-2).astype(mx.bfloat16) + s
    mx.eval(r)
    return r


def differs(a, b):
    return int((mx.view(a, mx.uint16) != mx.view(b, mx.uint16)).sum().item())


def main():
    only = sys.argv[1:] or list(VARIANTS)
    fns = {}
    for name in only:
        try:
            fns[name] = build(VARIANTS[name])
            print(f"[{name}] compiled OK", flush=True)
        except Exception as e:
            print(f"[{name}] COMPILE FAIL: {str(e)[:300]}", flush=True)
    if not fns:
        return
    for trial, (rows, scale, seed) in enumerate([
            (8, 1.0, 0), (32, 1.0, 1), (32, 30.0, 2), (32, 0.01, 3),
            (127, 3.0, 4), (32, 300.0, 5), (32, 1.0, 6), (32, 1.0, 7)]):
        mx.random.seed(seed)
        y = (mx.random.normal((rows, 4, 768)) * scale).astype(mx.bfloat16)
        w = mx.random.uniform(0.0, 1.0, (rows, 4)).astype(mx.float32)
        w = w / w.sum(axis=-1, keepdims=True) * 2.5
        s = (mx.random.normal((rows, 768)) * scale).astype(mx.bfloat16)
        mx.eval(y, w, s)
        r = ref(y, w, s)
        line = f"trial {trial} rows={rows} scale={scale}: ref done;"
        for name, fn in fns.items():
            try:
                o = run(fn, y, w, s)
                line += f" {name}={differs(o, r)}"
            except Exception as e:
                line += f" {name}=ERR:{str(e)[:80]}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
