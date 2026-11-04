// Kernel 1 of the two-kernel routing split: dense router matmul + sigmoid + bias.
// Grid: rows*60 threads, one per (row, expert). Threadgroup: 480 = 8 rows x 60.
// The k=768 reduction is tiled in chunks of 64 through threadgroup memory so the
// 184 KB router weight is read once per 8 rows instead of once per row.
// Every threadgroup covers whole rows (grid and group sizes are multiples of 60),
// so no bounds checks are needed.
threadgroup float w_tile[60 * 64];
threadgroup float x_tile[8 * 64];

uint gid = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint r = gid / 60;
uint e = gid % 60;
uint lr = tid / 60;

float acc = 0.0f;
for (uint k0 = 0; k0 < 768; k0 += 64) {
    // The first local row stages this expert's weight chunk.
    if (lr == 0) {
        uint wbase = e * 768 + k0;
        for (uint j = 0; j < 64; ++j) {
            w_tile[e * 64 + j] = w_router[wbase + j];
        }
    }
    // Thread e==0 of each local row stages its row's x chunk (exact bf16->f32).
    if (e == 0) {
        uint xbase = r * 768 + k0;
        for (uint j = 0; j < 64; ++j) {
            x_tile[lr * 64 + j] = float(x[xbase + j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 64; ++j) {
        acc = fma(x_tile[lr * 64 + j], w_tile[e * 64 + j], acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float s = 1.0f / (1.0f + metal::exp(-acc));
scores[r * 60 + e] = s;
biased[r * 60 + e] = s + bias[e];
