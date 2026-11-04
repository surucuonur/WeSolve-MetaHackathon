// hyp-2-experts-prefill direction (e), round-3 attempt-5 (builder-2-e).
// K3 select_small: exact copy of the active path's output to y.
// rows == 1 ran the unfused K0/K1 path (y_small); every other row count
// ran the fused kernel (y_fused). One thread per bfloat16 element.
uint rows = y_fused_shape[0];
uint n = rows * 3072u;
uint gid = thread_position_in_grid.x;
if (gid >= n) { return; }
y[gid] = (rows == 1u) ? y_small[gid] : y_fused[gid];
