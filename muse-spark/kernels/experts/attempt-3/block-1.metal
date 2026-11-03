// Kernel 2: grouped down product. One threadgroup per expert (E=60 groups of
// 384 threads); thread o computes output columns 2o,2o+1 for every pair
// routed to this expert, in unrolled chunks of up to 2 pairs with scalar
// accumulators, branchless inner loop, K unrolled x2 (see block-0). y[t] with
// t = r*4+j keeps the positional pairing with inds (Boundary neighbour B).
uint e = threadgroup_position_in_grid.x;
uint o = thread_position_in_threadgroup.x;
int rows = inds_shape[0];
uint P = (uint)rows * 4u;

threadgroup uint match_[128];
threadgroup uint plist[128];
threadgroup uint dW[3072];
threadgroup bfloat dS[768];
threadgroup bfloat dB[768];
threadgroup bfloat hT[64];
threadgroup uint nP;

if (o < P) {
    match_[o] = (inds[o] == e) ? 1u : 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (o == 0) {
    uint n = 0;
    for (uint t = 0; t < P; ++t) {
        if (match_[t] != 0u) {
            plist[n] = t;
            ++n;
        }
    }
    nP = n;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint Pe = nP;

uint eb = (uint)e * 384u;
uint o0 = o * 2u;
uint word = o0 >> 3u;
uint sh0 = (o0 & 7u) * 4u;
uint sh1 = ((o0 + 1u) & 7u) * 4u;
uint sg = o0 >> 5u;
for (uint c0 = 0; c0 < Pe; c0 += 2u) {
    uint n = min(2u, Pe - c0);
    uint t0 = plist[c0];
    uint t1 = (1u < n) ? plist[c0 + 1u] : t0;
    float e00 = 0.0f, o00 = 0.0f, e01 = 0.0f, o01 = 0.0f;
    float e10 = 0.0f, o10 = 0.0f, e11 = 0.0f, o11 = 0.0f;
    for (uint i0 = 0; i0 < 384u; i0 += 32u) {
        // Weight tile: 32 inputs x 96 words, 384 threads x 8 coalesced words.
        for (uint k = 0; k < 8u; ++k) {
            uint idx = o * 8u + k;
            uint ii = i0 + idx / 96u;
            uint w = idx % 96u;
            dW[idx] = down_w[(eb + ii) * 96u + w];
        }
        // Scales/biases: 32 inputs x 24 groups, 384 threads x 2 values.
        for (uint k = 0; k < 2u; ++k) {
            uint idx = o * 2u + k;
            uint ii = i0 + idx / 24u;
            uint sc = idx % 24u;
            uint si = (eb + ii) * 24u + sc;
            dS[idx] = down_scales[si];
            dB[idx] = down_biases[si];
        }
        // hidden tiles for this chunk's pairs (phantom slot repeats pair 0).
        if (o < 64u) {
            uint p = o / 32u;
            uint ii = i0 + (o % 32u);
            uint t = (p == 0u) ? t0 : t1;
            hT[o] = hidden[t * 384u + ii];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const uint* dWp = dW + word;
        threadgroup const bfloat* dSp = dS + sg;
        threadgroup const bfloat* dBp = dB + sg;
        for (uint i = 0; i < 32u; i += 2u) {
            float w0 = (float)dSp[0] * (float)((dWp[0] >> sh0) & 15u) + (float)dBp[0];
            float w1 = (float)dSp[0] * (float)((dWp[0] >> sh1) & 15u) + (float)dBp[0];
            float h0e = (float)hT[i];
            float h1e = (float)hT[32u + i];
            e00 += w0 * h0e; e01 += w1 * h0e;
            e10 += w0 * h1e; e11 += w1 * h1e;
            float w0o = (float)dSp[24] * (float)((dWp[96] >> sh0) & 15u) + (float)dBp[24];
            float w1o = (float)dSp[24] * (float)((dWp[96] >> sh1) & 15u) + (float)dBp[24];
            float h0o = (float)hT[i + 1u];
            float h1o = (float)hT[32u + i + 1u];
            o00 += w0o * h0o; o01 += w1o * h0o;
            o10 += w0o * h1o; o11 += w1o * h1o;
            dWp += 192u; dSp += 48u; dBp += 48u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[t0 * 768u + o0] = (bfloat)(e00 + o00);
    y[t0 * 768u + o0 + 1u] = (bfloat)(e01 + o01);
    if (1u < n) {
        y[t1 * 768u + o0] = (bfloat)(e10 + o10);
        y[t1 * 768u + o0 + 1u] = (bfloat)(e11 + o11);
    }
}
