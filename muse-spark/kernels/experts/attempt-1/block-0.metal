// Fused grouped experts kernel: direction (a), ONE Metal kernel, no sort/unsort.
//
// Threadgroup e (0..59, 192 threads) serves every (row, slot) pair routed to
// expert e, in chunks of 2 pairs. Gate/up/down plus silu-mul are fused with
// gate/up/hidden intermediates on-chip (no hidden DRAM round-trip).
//
// Numerics follow attempt-3's G2-proven forms exactly (same values, same
// order): per product, weight/scales/x tiles stream through threadgroup
// memory while each thread accumulates with even/odd-K split accumulators
// and plain s*(shifted nibble)+bias dequant, rounding (bfloat)(even+odd);
// silu is MLX's stable bfloat form verbatim. Gate/up match attempt-3 k1
// thread-for-thread (2 pairs x 2 outputs); down matches k2's per-thread
// shape (2 pairs x 2 outputs, 8 accs) over 2 output blocks, keeping
// occupancy parity. Valid for every row count 1..32 (grid rows-independent;
// rows from x_shape[0]; all indexing in bounds; phantom slots valid, stores
// guarded).
using namespace metal;

threadgroup uint plist[128];
threadgroup uint Wtile[3072];
threadgroup bfloat Stile[768];
threadgroup bfloat Btile[768];
threadgroup bfloat xT[64];
threadgroup bfloat hbuf[2 * 384];
threadgroup uint nP;

{
    uint e = threadgroup_position_in_grid.x;
    uint o = thread_position_in_threadgroup.x;
    int rows = x_shape[0];
    uint P = (uint)rows * 4u;

    // Pair scan: one thread collects the (row, slot) codes routed to expert e.
    if (o == 0) {
        uint n = 0;
        for (uint t = 0; t < P; ++t) {
            if (inds[t] == e) {
                plist[n++] = t;
            }
        }
        nP = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint Pe = nP;

    uint ebGU = e * 768u;
    uint ebD = e * 384u;

    for (uint c0 = 0; c0 < Pe; c0 += 2u) {
        uint n = min(2u, Pe - c0);
        uint t0 = plist[c0];
        uint t1 = (1u < n) ? plist[c0 + 1u] : t0;

        // Gate/up sweep: thread o covers hidden columns 2o,2o+1.
        uint o0 = o * 2u;
        uint word = o0 >> 3u;
        uint sh0 = (o0 & 7u) * 4u;
        uint sh1 = ((o0 + 1u) & 7u) * 4u;
        uint sg = o0 >> 5u;
        float eG00 = 0.0f, oG00 = 0.0f, eU00 = 0.0f, oU00 = 0.0f;
        float eG01 = 0.0f, oG01 = 0.0f, eU01 = 0.0f, oU01 = 0.0f;
        float eG10 = 0.0f, oG10 = 0.0f, eU10 = 0.0f, oU10 = 0.0f;
        float eG11 = 0.0f, oG11 = 0.0f, eU11 = 0.0f, oU11 = 0.0f;
        for (uint i0 = 0; i0 < 768u; i0 += 32u) {
            // Weight tiles: 32 inputs x 48 words, stride-1 across threads.
            for (uint k = 0; k < 8u; ++k) {
                uint idx = k * 192u + o;
                uint ii = i0 + idx / 48u;
                uint w = idx % 48u;
                uint gi = (ebGU + ii) * 48u + w;
                Wtile[idx] = gate_w[gi];
                Wtile[1536u + idx] = up_w[gi];
            }
            // Scales/biases: 32 inputs x 12 groups, stride-1.
            for (uint k = 0; k < 2u; ++k) {
                uint idx = k * 192u + o;
                uint ii = i0 + idx / 12u;
                uint sc = idx % 12u;
                uint si = (ebGU + ii) * 12u + sc;
                Stile[idx] = gate_scales[si];
                Btile[idx] = gate_biases[si];
                Stile[384u + idx] = up_scales[si];
                Btile[384u + idx] = up_biases[si];
            }
            // x tiles for this chunk's pairs (phantom slot repeats pair 0).
            if (o < 64u) {
                uint p = o / 32u;
                uint ii = i0 + (o % 32u);
                uint t = (p == 0u) ? t0 : t1;
                xT[o] = x[(t >> 2u) * 768u + ii];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup const uint* gWp = Wtile + word;
            threadgroup const uint* uWp = Wtile + 1536u + word;
            threadgroup const bfloat* gSp = Stile + sg;
            threadgroup const bfloat* gBp = Btile + sg;
            threadgroup const bfloat* uSp = Stile + 384u + sg;
            threadgroup const bfloat* uBp = Btile + 384u + sg;
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
        // silu-mul into on-chip hbuf (both slots always valid writes).
        {
            bfloat gb = (bfloat)(eG00 + oG00);
            bfloat ub = (bfloat)(eU00 + oU00);
            bfloat yb = 1 / (1 + metal::precise::exp(metal::abs(gb)));
            bfloat sb = (gb < 0) ? yb : 1 - yb;
            hbuf[o0] = (gb * sb) * ub;
        }
        {
            bfloat gb = (bfloat)(eG01 + oG01);
            bfloat ub = (bfloat)(eU01 + oU01);
            bfloat yb = 1 / (1 + metal::precise::exp(metal::abs(gb)));
            bfloat sb = (gb < 0) ? yb : 1 - yb;
            hbuf[o0 + 1u] = (gb * sb) * ub;
        }
        {
            bfloat gb = (bfloat)(eG10 + oG10);
            bfloat ub = (bfloat)(eU10 + oU10);
            bfloat yb = 1 / (1 + metal::precise::exp(metal::abs(gb)));
            bfloat sb = (gb < 0) ? yb : 1 - yb;
            hbuf[384u + o0] = (gb * sb) * ub;
            bfloat gb1 = (bfloat)(eG11 + oG11);
            bfloat ub1 = (bfloat)(eU11 + oU11);
            bfloat yb1 = 1 / (1 + metal::precise::exp(metal::abs(gb1)));
            bfloat sb1 = (gb1 < 0) ? yb1 : 1 - yb1;
            hbuf[384u + o0 + 1u] = (gb1 * sb1) * ub1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down sweep: 2 output blocks x 384 (2 outputs/thread, k2 shape).
        for (uint bb = 0; bb < 2u; ++bb) {
            uint d0 = bb * 384u + o * 2u;
            uint dword = d0 >> 3u;
            uint dsh0 = (d0 & 7u) * 4u;
            uint dsh1 = ((d0 + 1u) & 7u) * 4u;
            uint dsg = d0 >> 5u;
            float e00 = 0.0f, o00 = 0.0f, e01 = 0.0f, o01 = 0.0f;
            float e10 = 0.0f, o10 = 0.0f, e11 = 0.0f, o11 = 0.0f;
            for (uint i0 = 0; i0 < 384u; i0 += 32u) {
                // Weight tile: 32 inputs x 96 words, stride-1.
                for (uint k = 0; k < 16u; ++k) {
                    uint idx = k * 192u + o;
                    uint ii = i0 + idx / 96u;
                    uint w = idx % 96u;
                    Wtile[idx] = down_w[(ebD + ii) * 96u + w];
                }
                // Scales/biases: 32 inputs x 24 groups, stride-1.
                for (uint k = 0; k < 4u; ++k) {
                    uint idx = k * 192u + o;
                    uint ii = i0 + idx / 24u;
                    uint sc = idx % 24u;
                    uint si = (ebD + ii) * 24u + sc;
                    Stile[idx] = down_scales[si];
                    Btile[idx] = down_biases[si];
                }
                // Hidden tiles from on-chip hbuf (chunk slots 0/1).
                if (o < 64u) {
                    uint p = o / 32u;
                    uint ii = i0 + (o % 32u);
                    xT[o] = hbuf[p * 384u + ii];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup const uint* dWp = Wtile + dword;
                threadgroup const bfloat* dSp = Stile + dsg;
                threadgroup const bfloat* dBp = Btile + dsg;
                for (uint i = 0; i < 32u; i += 2u) {
                    float w0 = (float)dSp[0] * (float)((dWp[0] >> dsh0) & 15u) + (float)dBp[0];
                    float w1 = (float)dSp[0] * (float)((dWp[0] >> dsh1) & 15u) + (float)dBp[0];
                    float h0e = (float)xT[i];
                    float h1e = (float)xT[32u + i];
                    e00 += w0 * h0e; e01 += w1 * h0e;
                    e10 += w0 * h1e; e11 += w1 * h1e;
                    float w0o = (float)dSp[24] * (float)((dWp[96] >> dsh0) & 15u) + (float)dBp[24];
                    float w1o = (float)dSp[24] * (float)((dWp[96] >> dsh1) & 15u) + (float)dBp[24];
                    float h0o = (float)xT[i + 1u];
                    float h1o = (float)xT[32u + i + 1u];
                    o00 += w0o * h0o; o01 += w1o * h0o;
                    o10 += w0o * h1o; o11 += w1o * h1o;
                    dWp += 192u; dSp += 48u; dBp += 48u;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            y[t0 * 768u + d0] = (bfloat)(e00 + o00);
            y[t0 * 768u + d0 + 1u] = (bfloat)(e01 + o01);
            if (1u < n) {
                y[t1 * 768u + d0] = (bfloat)(e10 + o10);
                y[t1 * 768u + d0 + 1u] = (bfloat)(e11 + o11);
            }
        }
        // Next chunk overwrites tiles/hbuf; all reads above must be done.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
