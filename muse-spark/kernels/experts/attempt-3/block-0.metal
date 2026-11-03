// Kernel 1: grouped gate+up+activation. One threadgroup per expert (E=60 groups
// of 192 threads); thread o computes hidden columns 2o,2o+1 for every (row,
// expert) pair routed to this expert, in chunks of up to 2 pairs. Two
// consecutive outputs share one weight word and one scale/bias (8/32-wide
// groups never split an even pair), halving loads per output. Weight tiles
// load once per chunk into threadgroup memory. The 2 pair slots are unrolled
// with scalar accumulators and a branchless inner loop (short chunks are
// padded with pair 0's x and their stores skipped); K is unrolled x2 with
// even/odd accumulators for ILP. v3: accumulator arrays spill (~10x slow).
// v4: pair-outermost re-reads tiles per pair (bandwidth-bound). v5: chunk-8
// with per-iteration guards: guard/icache bloat eats the sharing gain.
// Rounding: f32 accumulation, bf16 round after each product, MLX's stable
// bfloat sigmoid bit-exact, bfloat muls (brief:90).
uint e = threadgroup_position_in_grid.x;
uint o = thread_position_in_threadgroup.x;
int rows = inds_shape[0];
uint P = (uint)rows * 4u;

threadgroup uint match_[128];
threadgroup uint plist[128];
threadgroup uint gW[1536];
threadgroup uint uW[1536];
threadgroup bfloat gS[384];
threadgroup bfloat gB[384];
threadgroup bfloat uS[384];
threadgroup bfloat uB[384];
threadgroup bfloat xT[64];
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

uint eb = (uint)e * 768u;
uint o0 = o * 2u;
uint word = o0 >> 3u;
uint sh0 = (o0 & 7u) * 4u;
uint sh1 = ((o0 + 1u) & 7u) * 4u;
uint sg = o0 >> 5u;
for (uint c0 = 0; c0 < Pe; c0 += 2u) {
    uint n = min(2u, Pe - c0);
    uint t0 = plist[c0];
    uint t1 = (1u < n) ? plist[c0 + 1u] : t0;
    float eG00 = 0.0f, oG00 = 0.0f, eU00 = 0.0f, oU00 = 0.0f;
    float eG01 = 0.0f, oG01 = 0.0f, eU01 = 0.0f, oU01 = 0.0f;
    float eG10 = 0.0f, oG10 = 0.0f, eU10 = 0.0f, oU10 = 0.0f;
    float eG11 = 0.0f, oG11 = 0.0f, eU11 = 0.0f, oU11 = 0.0f;
    for (uint i0 = 0; i0 < 768u; i0 += 32u) {
        // Weight tile: 32 inputs x 48 words, 192 threads x 8 coalesced words.
        for (uint k = 0; k < 8u; ++k) {
            uint idx = o * 8u + k;
            uint ii = i0 + idx / 48u;
            uint w = idx % 48u;
            uint gi = (eb + ii) * 48u + w;
            gW[idx] = gate_w[gi];
            uW[idx] = up_w[gi];
        }
        // Scales/biases: 32 inputs x 12 groups, 192 threads x 2 values.
        for (uint k = 0; k < 2u; ++k) {
            uint idx = o * 2u + k;
            uint ii = i0 + idx / 12u;
            uint sc = idx % 12u;
            uint si = (eb + ii) * 12u + sc;
            gS[idx] = gate_scales[si];
            gB[idx] = gate_biases[si];
            uS[idx] = up_scales[si];
            uB[idx] = up_biases[si];
        }
        // x tiles for this chunk's pairs (phantom slot repeats pair 0).
        if (o < 64u) {
            uint p = o / 32u;
            uint ii = i0 + (o % 32u);
            uint t = (p == 0u) ? t0 : t1;
            xT[o] = x[(t >> 2u) * 768u + ii];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const uint* gWp = gW + word;
        threadgroup const uint* uWp = uW + word;
        threadgroup const bfloat* gSp = gS + sg;
        threadgroup const bfloat* gBp = gB + sg;
        threadgroup const bfloat* uSp = uS + sg;
        threadgroup const bfloat* uBp = uB + sg;
        for (uint i = 0; i < 32u; i += 2u) {
            float gw0 = (float)gSp[0] * (float)((gWp[0] >> sh0) & 15u) + (float)gBp[0];
            float uw0 = (float)uSp[0] * (float)((uWp[0] >> sh0) & 15u) + (float)uBp[0];
            float gw1 = (float)gSp[0] * (float)((gWp[0] >> sh1) & 15u) + (float)gBp[0];
            float uw1 = (float)uSp[0] * (float)((uWp[0] >> sh1) & 15u) + (float)uBp[0];
            float x0e = (float)xT[i];
            float x1e = (float)xT[32u + i];
            eG00 += gw0 * x0e; eU00 += uw0 * x0e;
            eG01 += gw1 * x0e; eU01 += uw1 * x0e;
            eG10 += gw0 * x1e; eU10 += uw0 * x1e;
            eG11 += gw1 * x1e; eU11 += uw1 * x1e;
            float gw0o = (float)gSp[12] * (float)((gWp[48] >> sh0) & 15u) + (float)gBp[12];
            float uw0o = (float)uSp[12] * (float)((uWp[48] >> sh0) & 15u) + (float)uBp[12];
            float gw1o = (float)gSp[12] * (float)((gWp[48] >> sh1) & 15u) + (float)gBp[12];
            float uw1o = (float)uSp[12] * (float)((uWp[48] >> sh1) & 15u) + (float)uBp[12];
            float x0o = (float)xT[i + 1u];
            float x1o = (float)xT[32u + i + 1u];
            oG00 += gw0o * x0o; oU00 += uw0o * x0o;
            oG01 += gw1o * x0o; oU01 += uw1o * x0o;
            oG10 += gw0o * x1o; oU10 += uw0o * x1o;
            oG11 += gw1o * x1o; oU11 += uw1o * x1o;
            gWp += 96u; uWp += 96u;
            gSp += 24u; gBp += 24u; uSp += 24u; uBp += 24u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {
        bfloat gb = (bfloat)(eG00 + oG00);
        bfloat ub = (bfloat)(eU00 + oU00);
        bfloat yb = 1 / (1 + metal::precise::exp(metal::abs(gb)));
        bfloat sb = (gb < 0) ? yb : 1 - yb;
        hidden[t0 * 384u + o0] = (gb * sb) * ub;
    }
    {
        bfloat gb = (bfloat)(eG01 + oG01);
        bfloat ub = (bfloat)(eU01 + oU01);
        bfloat yb = 1 / (1 + metal::precise::exp(metal::abs(gb)));
        bfloat sb = (gb < 0) ? yb : 1 - yb;
        hidden[t0 * 384u + o0 + 1u] = (gb * sb) * ub;
    }
    if (1u < n) {
        bfloat gb = (bfloat)(eG10 + oG10);
        bfloat ub = (bfloat)(eU10 + oU10);
        bfloat yb = 1 / (1 + metal::precise::exp(metal::abs(gb)));
        bfloat sb = (gb < 0) ? yb : 1 - yb;
        hidden[t1 * 384u + o0] = (gb * sb) * ub;
        bfloat gb1 = (bfloat)(eG11 + oG11);
        bfloat ub1 = (bfloat)(eU11 + oU11);
        bfloat yb1 = 1 / (1 + metal::precise::exp(metal::abs(gb1)));
        bfloat sb1 = (gb1 < 0) ? yb1 : 1 - yb1;
        hidden[t1 * 384u + o0 + 1u] = (gb1 * sb1) * ub1;
    }
}
