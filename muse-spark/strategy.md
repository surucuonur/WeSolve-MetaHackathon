> [harness] Written by Meta Muse Spark (muse-spark-1.3-contributor) via the Meta Model API.
> [harness] Call muse-spark/calls/0001-strategy, response resp_6ae9131a4c3c30fa5ab34357.
> [harness] --- Muse Spark output below, unmodified ---
Routing is launch overhead, experts is bandwidth. At 1 row the step is 867 kernels with 8% occupancy and the MoE bookkeeping alone is 340 kernels; at 32 rows the routed experts are 52% of the step and read 2x bytes at 62 GB/s.

Fusing routing+combine removes ~320 launches/step and most CPU encode for that piece with tiny trivially parallel math, so high probability to beat `mx.compile` which cannot fuse `argpartition`. Fusing experts to read each expert once attacks the 465 MB/step of DRAM repeat reads, larger ceiling at 32 rows but much harder to beat tuned `gather_qmm`, so second.

```json
{
  "targets": [
    {
      "slot": "routing+combine",
      "why": "340 kernels/step (240 routing+100 combine). Knock-out 1.02+0.61=1.63ms of 4.75ms at rows=1, 1.68+0.33=2.01ms at rows=32. Effective 2-4 GB/s, occupancy-bound, plus 0.3ms CPU build and ~2ms encode share. Largest 1-row piece, helps all rows, low risk: tiny math, no quant.",
      "expected_saving_ms": {"rows_1": 1.3, "rows_32": 1.5},
      "design": "routing: 1 kernel/layer, grid rows*32. Each row: 60 dots of len 768 (x bf16->fp32 * w_router fp32) + sigmoid + bias, top-4 by registers insertion sort, normalize unbiased scores *2.5. Writes inds+weights. combine: 1 kernel/layer, grid rows*768. Each thread 1 output: out=shared_bf16 + sum_j weights[j]*y[j] in fp32 -> bf16. Single read of y/shared/weights, single write."
    },
    {
      "slot": "experts",
      "why": "14.67ms of 28.19ms at rows=32 (52%). Needs 443MB/layer-step distinct but reads 908MB from DRAM (1416MB per-pair, half cached). 62 GB/s vs 200 peak. Grouped read-once would save ~465MB DRAM + raise occupancy. Small at rows=1 (0.92ms) but required for large 32-row win.",
      "expected_saving_ms": {"rows_1": 0.6, "rows_32": 9.0},
      "design": "Group pairs by expert per layer (sort inds on CPU/GPU or per-expert threadgroups scanning inds). One threadgroup per (expert, output-tile 32): stream dequantized gate/up/down on the fly (scale*q+bias, q from 4b packed 8/word), reuse x and weights across tokens sharing expert via threadgroup memory, fp32 accumulate then bf16 round, silu(gate)*up in bf16 before down. 3 fused GEMMs, no per-pair reread."
    }
  ]
}
```