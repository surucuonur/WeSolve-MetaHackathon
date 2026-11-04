// hyp-2-experts-prefill direction (c), round-3 attempt-3 (builder-2-c).
// Count kernel: counting sort of pairs by expert (E=60, B=rows*4 <= 8192).
// One threadgroup of 64 threads; threads 0-59 own one expert each.
// Outputs the sorted pair indices plus expert and row-tile prefix sums,
// computed once per call (MLX recomputes offsets per product).
uint tid = thread_position_in_threadgroup.x;
uint rows = inds_shape[0];
uint B = rows * 4u;

threadgroup uint tg_counts[60];
threadgroup uint tg_expert_off[61];

if (tid < 60u) {
  uint cnt = 0u;
  for (uint p = 0u; p < B; ++p) {
    if (inds[p] == tid) cnt++;
  }
  tg_counts[tid] = cnt;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (tid == 0u) {
  uint off = 0u;
  uint toff = 0u;
  for (uint e = 0u; e < 60u; ++e) {
    tg_expert_off[e] = off;
    expert_offsets[e] = off;
    uint cnt = tg_counts[e];
    off += cnt;
    uint te = (cnt + 15u) / 16u;
    tile_offsets[e] = toff;
    toff += te;
  }
  tg_expert_off[60] = off;
  expert_offsets[60] = off;
  tile_offsets[60] = toff;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (tid < 60u) {
  uint base = tg_expert_off[tid];
  uint rank = 0u;
  for (uint p = 0u; p < B; ++p) {
    if (inds[p] == tid) {
      sorted_pairs[base + rank] = p;
      rank++;
    }
  }
}
