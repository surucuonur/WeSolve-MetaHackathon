// Kernel 2 of the two-kernel routing split: top-4 selection + weight normalize.
// Grid: rows threads, one independent thread per row. Selection runs on the
// biased scores; weights use the unbiased scores of the selected experts,
// normalized with epsilon 1e-20 and scaled by 2.5, as in the reference.
uint r = thread_position_in_grid.x;
uint base = r * 60;

float top_v[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
uint top_i[4] = {0, 0, 0, 0};
for (uint e = 0; e < 60; ++e) {
    float v = biased[base + e];
    if (v > top_v[3]) {
        top_v[3] = v;
        top_i[3] = e;
        for (uint j = 3; j > 0 && top_v[j] > top_v[j - 1]; --j) {
            float tv = top_v[j];
            top_v[j] = top_v[j - 1];
            top_v[j - 1] = tv;
            uint ti = top_i[j];
            top_i[j] = top_i[j - 1];
            top_i[j - 1] = ti;
        }
    }
}

float w0 = scores[base + top_i[0]];
float w1 = scores[base + top_i[1]];
float w2 = scores[base + top_i[2]];
float w3 = scores[base + top_i[3]];
float denom = w0 + w1 + w2 + w3 + 1e-20f;
inds[r * 4 + 0] = top_i[0];
inds[r * 4 + 1] = top_i[1];
inds[r * 4 + 2] = top_i[2];
inds[r * 4 + 3] = top_i[3];
weights[r * 4 + 0] = w0 / denom * 2.5f;
weights[r * 4 + 1] = w1 / denom * 2.5f;
weights[r * 4 + 2] = w2 / denom * 2.5f;
weights[r * 4 + 3] = w3 / denom * 2.5f;
