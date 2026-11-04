// hyp-2-experts-prefill direction (c), round-3 attempt-3 (builder-2-c).
// Select kernel: positional copy from the decode or prefill output,
// branched on the same rows >= 60 dispatch threshold. Exact bfloat copy.
uint idx = thread_position_in_grid.x;
uint rows = y_decode_shape[0];
uint total = rows * 3072u;
if (idx >= total) return;
bool prefill = rows >= 60u;
y[idx] = prefill ? y_prefill[idx] : y_decode[idx];
