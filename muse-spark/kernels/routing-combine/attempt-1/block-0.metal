// [harness] Written by Meta Muse Spark (muse-spark-1.3-contributor) via the Meta Model API.
// [harness] Call muse-spark/calls/0002-kernels-routing-combine-a1, response resp_6ae913db4181fa96e7db464c.
// [harness] --- Muse Spark output below, unmodified ---
uint row = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
threadgroup float xcache[768];
threadgroup float sc[60];
for (uint k = tid; k < 768; k += 64) {
    xcache[k] = static_cast<float>(x[row * 768 + k]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 60) {
    const device float* wptr = w_router + tid * 768;
    float acc = 0.0f;
    for (uint k = 0; k < 768; ++k) {
        acc += xcache[k] * wptr[k];
    }
    float sig = 1.0f / (1.0f + metal::exp(-acc));
    sc[tid] = sig;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint top_idx[4] = {0, 0, 0, 0};
    float top_b[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (uint e = 0; e < 60; ++e) {
        float b = sc[e] + bias[e];
        if (b > top_b[0]) {
            top_b[3] = top_b[2]; top_idx[3] = top_idx[2];
            top_b[2] = top_b[1]; top_idx[2] = top_idx[1];
            top_b[1] = top_b[0]; top_idx[1] = top_idx[0];
            top_b[0] = b; top_idx[0] = e;
        } else if (b > top_b[1]) {
            top_b[3] = top_b[2]; top_idx[3] = top_idx[2];
            top_b[2] = top_b[1]; top_idx[2] = top_idx[1];
            top_b[1] = b; top_idx[1] = e;
        } else if (b > top_b[2]) {
            top_b[3] = top_b[2]; top_idx[3] = top_idx[2];
            top_b[2] = b; top_idx[2] = e;
        } else if (b > top_b[3]) {
            top_b[3] = b; top_idx[3] = e;
        }
    }
    float s0 = sc[top_idx[0]];
    float s1 = sc[top_idx[1]];
    float s2 = sc[top_idx[2]];
    float s3 = sc[top_idx[3]];
    float sum = s0 + s1 + s2 + s3;
    float inv = 2.5f / (sum + 1e-20f);
    inds[row * 4 + 0] = top_idx[0];
    inds[row * 4 + 1] = top_idx[1];
    inds[row * 4 + 2] = top_idx[2];
    inds[row * 4 + 3] = top_idx[3];
    weights[row * 4 + 0] = s0 * inv;
    weights[row * 4 + 1] = s1 * inv;
    weights[row * 4 + 2] = s2 * inv;
    weights[row * 4 + 3] = s3 * inv;
}
