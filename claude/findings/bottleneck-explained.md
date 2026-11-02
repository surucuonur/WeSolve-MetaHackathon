# Where the bottleneck is, explained

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../PROVENANCE.md).*

This is a plain-English companion to [decode-bottlenecks.md](decode-bottlenecks.md) and [step-breakdown.md](step-breakdown.md). It pulls their results into one answer: what limits a decode step of MobileMoE-S on the M1 Pro, and where the waste is. Each number is worked through from its inputs, so every calculation can be checked by hand.

Every number comes from these runs:
- the piece-by-piece runs of 2026-11-01 ([tables](../../results/runs/2026-11-01-breakdown/tables.md));
- the GPU counter recordings of 2026-11-02 ([tables](../../results/runs/2026-11-02-gpu-counters/tables.md));
- the model's `config.json`;
- MLX's source code.

Each claim is marked **measured**, **from the code** or **estimate**, as in the other findings.

## The answer

The step is limited by **neither memory bandwidth nor arithmetic**. It is limited by the shape of the work. A decode step is 867–1,088 small GPU kernels run one after another. Each kernel pays a fixed start-up cost and fills only a small part of the GPU. Where most of the time goes depends on the setting:

- **One request:** the chain of tiny kernels, plus the CPU time needed to prepare them.
- **32 requests:** the **routed experts**, which take half the step. They read their weights once per token instead of once per expert, and even then move data at a third of peak bandwidth.
- **32 requests with 1,024-token prompts:** vLLM's **attention**, whose kernel re-reads the stored keys and values.

## Terms

[GLOSSARY.md](../GLOSSARY.md) defines the project's shared terms. These are the ones this page leans on:

| Term | Meaning |
|---|---|
| **Decode step** | One pass through the model's 20 layers that produces one new token for each request in the step. |
| **Lower bound** | The fastest the step could run: its bytes at peak bandwidth (200 GB/s) or its arithmetic at peak compute (5.2 TFLOP/s), whichever takes longer. |
| **Bandwidth-bound / compute-bound / overhead-bound** | A step near its bandwidth limit, near its compute limit, or far below both. |
| **Kernel** | One small program the GPU runs, such as "multiply this matrix" or "add these two arrays". A step is a chain of them. |
| **Fixed cost per kernel** | Time each kernel pays whatever its size: setting it up, filling the GPU's cores with threads, and draining them at the end. Here it is about 4–7 µs per kernel. |
| **Threadgroup** | A team of GPU threads that works on one tile of a kernel's output. A kernel with little data launches few teams. |
| **Occupancy** | The share of the GPU's thread slots in use. Low occupancy means most of the chip is idle even while the GPU counts as busy. |
| **GPU busy %** | The share of time the GPU had any kernel running. It says nothing about how full the GPU was. |
| **ALU** | The GPU's arithmetic units. |
| **Limiter %** | How close a hardware unit is to being the bottleneck. 100% means it is the limit. |
| **Cache** | Small, fast memory on the chip that keeps recently read data. A second read of the same data may come from the cache instead of main memory. |
| **Expert, routed, shared** | Each mixture-of-experts layer has 60 small sub-networks called experts. Each token uses the 4 the router picks (the routed experts) plus 1 that every token uses (the shared expert). |
| **Routing, combine** | Choosing each token's 4 experts and their weights; then weighting the experts' outputs and adding them up. This is bookkeeping, not the model's main math. |
| **(Token, expert) pair** | One token using one expert. 32 tokens × 4 experts make 128 pairs per layer. |
| **KV cache** | The stored keys and values of earlier tokens, which attention reads in every step. |
| **Knock-out** | Replacing one piece of the step with a stand-in that returns the right shape without GPU work. The time and bytes the step saves are that piece's cost inside the real step. |
| **Split-K** | MLX's way of running a small matrix product on more threads: the long input dimension (K) is cut into chunks, each chunk's partial result is written to memory, and a second kernel adds them up. |

## 1. What limits a decode step

To produce a token, the GPU must read the weights the step uses and do the arithmetic on them. At one request, a step reads 179 MB and does 0.56 billion operations (GFLOP):

```
bandwidth limit:  179 MB   ÷ 200 GB/s      = 0.89 ms
compute limit:    0.56 GFLOP ÷ 5.2 TFLOP/s = 0.11 ms
lower bound (the slower of the two)        = 0.89 ms
```

Measured step times (2026-11-01, unrecorded runs, median of 3 repeats):

| Engine | Requests | Prompt | Step | Lower bound | Times the bound |
|---|---:|---:|---:|---:|---:|
| plain MLX | 1 | 128 | 4.75 ms | 0.89 ms | 5.3 |
| plain MLX | 32 | 128 | 28.19 ms | 3.71 ms | 7.6 |
| vLLM | 1 | 128 | 5.64 ms | 0.89 ms | 6.3 |
| vLLM | 32 | 128 | 30.47 ms | 3.50 ms | 8.7 |
| vLLM | 32 | 1,024 | 39.71 ms | 6.38 ms | 6.2 |

Every step takes 5–9 times its lower bound, so every step is **overhead-bound** (measured).

## 2. The GPU is busy but mostly empty

The GPU's own counters, recorded with Xcode during steady decode steps (measured):

| Counter | 1 request | 8–32 requests | What a bottleneck would show |
|---|---:|---:|---:|
| GPU busy | 95–96% | 92–98% | — |
| Memory bandwidth used | 17–20% of peak | 24–28% of peak | near 100% |
| Arithmetic units used | 14–17% | 33–45% | near 100% |
| Thread slots filled (occupancy) | 8–9% | 21–27% | high |
| Closest unit to its limit | 11–13% | 47–56% | near 100% |

The GPU always has a kernel running, but no part of it is close to full. This is what many small kernels look like: the GPU is never idle, yet each kernel is too small to use the whole chip.

## 3. Too many small kernels

**How many kernels a step runs.** Counted from the step's graph (measured):

| | 1 request | 32 requests |
|---|---:|---:|
| plain MLX | 867 | 987 |
| vLLM | 968 | 1,088 |

The 867 kernels of plain MLX's one-request step break down as:
- **241 (28%) do the model's main math:** quantized projections, expert products, attention and the router's product.
- **The rest are small steps around that math:** adds, norms, number-format conversions, index ranges, and routing's selection and weighting.

**Why a small kernel can't reach peak bandwidth.** Take the q projection of one layer at one request:

```
weights to read:   7.37 MB for 20 layers ÷ 20 = 0.37 MB per layer
time at 200 GB/s:  0.37 MB ÷ 200 GB/s        = 1.8 µs
fixed cost:                                    about 4–7 µs
```

The kernel spends longer starting and stopping than moving its data. It also launches few threads. A GPU only reaches full bandwidth with many memory reads in flight at once, and a few threads can't keep enough of them going. The counters show the low occupancy; that it is what holds bandwidth down is an estimate.

**The same hardware at very different speeds.** What each piece costs at one request, by knock-out, against the bytes it must read (measured, plain MLX):

| Piece | Kernels | MB to read | Time it costs | Effective speed |
|---|---:|---:|---:|---:|
| Output layer | 1 | 61.6 | 0.39 ms | 158 GB/s (79% of peak) |
| Shared expert | 80 | 44.2 | 0.51 ms | 86 GB/s |
| Routed experts | 140 | 44.2 | 0.92 ms | 48 GB/s |
| q, k and v projections | 60 | 12.3 | 0.39 ms | 32 GB/s |
| Routing | 240 | 3.7 | 1.02 ms | 4 GB/s |

The output layer is one large kernel and nearly reaches peak. Routing is 240 tiny kernels that read almost nothing, yet it costs more time than the output layer. Routing (1.02 ms) and combine (0.61 ms) together cost 1.6 ms, more than the expert products themselves (0.92 ms).

**At one request, the CPU is near its limit too.** Before the GPU can run the kernels, the CPU must describe each one to it; this is called encoding. In the baseline, vllm-metal's step-submit call used 4.7 ms of CPU time in a 5.5 ms step (measured). An infinitely fast GPU would still leave the step near 5 ms. Fewer kernels shorten both the GPU's chain and the CPU's encoding.

## 4. With 32 requests, the routed experts take half the step

What each piece costs at 32 requests with 128-token prompts, by knock-out (measured):

| Piece | plain MLX | vLLM |
|---|---:|---:|
| Whole step | 28.19 ms | 30.47 ms |
| Routed experts | 14.67 ms (52%) | 14.85 ms (49%) |
| Attention, all of it | 5.60 ms | 7.48 ms |
| Shared expert | 2.90 ms | 3.13 ms |
| Output layer | 2.50 ms | 5.08 ms |
| Routing | 1.68 ms | 1.53 ms |

The routed experts lose time in two ways: they read about twice the data they need, and they move it slowly.

### Worked example: how big is one expert?

From `config.json` (from the code):
- 60 experts per layer;
- 4 experts per token;
- 20 layers;
- hidden size 768, expert size 384.

One expert is three matrices: gate (768 → 384), up (768 → 384) and down (384 → 768). Each holds 768 × 384 = 294,912 weights. In the int4 model, each weight takes half a byte. Each group of 32 weights also stores a bfloat16 scale and offset, 2 bytes each:

```
weights:           294,912 × 0.5 byte          = 147,456 bytes
scales, offsets:   294,912 ÷ 32 groups × 4 B   =  36,864 bytes
one matrix                                     = 184,320 bytes
one expert, 3 matrices                         = 552,960 bytes ≈ 0.553 MB
```

The converted model agrees: its expert weights total 663.55 MB, and 663.55 MB ÷ (60 experts × 20 layers) = 0.553 MB.

### Worked example: what the experts need, and what they read

**One request:** 4 experts × 20 layers × 0.553 MB = **44 MB** needed. The GPU counter shows 193 − 142 = **51 MB** read (the full step's reads minus the reads with the experts knocked out). Each expert is read once.

**32 requests:** the 32 tokens choose experts independently and often share them. The distinct experts each layer used in the counted step (measured, 2026-11-01):

```
layer:     1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
distinct: 43  50  45  52  46  43  41  39  27  38  31  39  40  38  36  43  39  37  36  39
total: 802 (layer, expert) combinations
```

Layer 1, for example, has 32 × 4 = 128 pairs falling on 43 experts, so each of those experts is wanted by about 3 tokens.

There are two ways to read the weights:

```
each distinct expert once:     802 × 0.553 MB                       =   443 MB
once per (token, expert) pair: 32 × 4 × 20 = 2,560 pairs × 0.553 MB = 1,416 MB
```

The second is what MLX does: below 60 rows per step it picks a kernel that handles each pair on its own (from the code: `quantized.cpp`, MLX 0.32.4). The GPU counter shows what reached memory (measured):

```
step reads, all pieces running:       1,557 MB
step reads, routed experts knocked out: 649 MB
routed experts read:                    908 MB
```

The pair-by-pair kernel would read 1,416 − 443 = 973 MB more than needed. Only 908 − 443 = **465 MB** of that reached memory; the cache served the rest, about half. Those 465 MB are 30% of everything the step reads (465 ÷ 1,557).

### Worked example: the experts are also slow per byte

```
908 MB ÷ 14.67 ms                        = 62 GB/s (31% of peak)
908 MB at 200 GB/s                       = 4.5 ms
443 MB at 200 GB/s (each expert once)    = 2.2 ms
measured time                            = 14.67 ms
```

The repeat reads are real, but most of the 14.67 ms is still kernel efficiency, not bytes (estimate, from the numbers above).

## 5. Attention re-reads the stored keys and values

### Worked example: what attention needs

Per token and layer, the cache holds a key and a value for each of 4 stored heads: 64 numbers each, in bfloat16 (from the code). During the 32-request recording, the 32 conversations held 7,792 tokens in all, about 243 each: 128 from the prompt plus those generated so far (measured).

```
per token, per layer:  2 (key, value) × 4 heads × 64 × 2 bytes = 1,024 bytes
needed:                7,792 tokens × 1,024 bytes × 20 layers  =   160 MB
```

### Why it reads more

The model has 12 query heads but only 4 stored heads, so each stored head is shared by 3 query heads. A kernel built for this sharing reads each stored head once. A simpler kernel lets each query head read it on its own (from the code):

```
needed:                          160 MB
each query head reads its own:   3 × 160 = 479 MB
plain MLX, measured:             265 MB   (1.7 × what it needs)
```

### vLLM's attention

vLLM uses vllm-metal's attention kernel instead. That kernel has no version for this head layout (12 query heads, 4 stored heads, head size 64), so it falls back to reading the stored heads once per query head (from the code; see [decode-bottlenecks.md](decode-bottlenecks.md#6-at-long-contexts-attention-reads-each-keyvalue-head-three-times)). vllm-metal's attention path also runs about 1 ms of Python per step and makes 60–120 extra copies of q, k and v (measured).

With 1,024-token prompts at 32 requests, vLLM's step is 6.5 ms longer than plain MLX's (39.71 vs 33.20 ms, measured). The counter recordings did not separate vLLM's attention bytes.

## 6. Split-K writes about 170 MB per step from 14 requests up

### The clue

A decode step writes little to memory: the new keys and values, and the results passed from one kernel to the next. Bytes written per step (measured):

```
requests:          1      8       32
MB written:        3     20–21   251–266
```

From 8 to 32 requests the writes grow about 12-fold, although the requests grow only 4-fold. No piece of the model explains that.

### What split-K does

MLX divides a matrix product's output into tiles of 32 × 32, one threadgroup each. At 32 requests, the q projection (32 rows × 768 → 768) has 1 × 24 = 24 tiles. That is too few teams to fill a 16-core GPU. Split-K also cuts the 768 inputs (K) into chunks, one team per chunk: 12 chunks give 24 × 12 = 288 teams. Each chunk then holds only a partial result. All of them are written to memory as float32 (4 bytes per number), and a second kernel reads them back and adds them up.

### When MLX uses it, and how much it writes

From the code (`quantized.cpp`, MLX 0.32.4):

- **When:** `get_qmv_batch_limit` returns 14 for the M1 Pro (`applegpu_g13s`) at these sizes. With 14 or more rows, MLX uses split-K. Below 14 rows it uses a vector kernel with no partial results. So 8 requests write little and 32 requests write a lot.
- **How many chunks:** `split_k = 512 ÷ tiles`, capped at K ÷ 32, then lowered until K divides evenly by `split_k × 32`.
- **Partial-result buffer:** `split_k × rows × outputs × 4 bytes`.

At 32 rows:

| Product | K → N | Tiles | 512 ÷ tiles → chunks | Buffer per layer | × 20 layers |
|---|---|---:|---|---:|---:|
| q_proj | 768 → 768 | 24 | 21 → 12 | 12 × 32 × 768 × 4 = 1.18 MB | 23.6 MB |
| k_proj | 768 → 256 | 8 | 64 → 24 | 24 × 32 × 256 × 4 = 0.79 MB | 15.7 MB |
| v_proj | 768 → 256 | 8 | 64 → 24 | 0.79 MB | 15.7 MB |
| o_proj | 768 → 768 | 24 | 21 → 12 | 1.18 MB | 23.6 MB |
| shared gate_proj | 768 → 1,536 | 48 | 10 → 8 | 8 × 32 × 1,536 × 4 = 1.57 MB | 31.5 MB |
| shared up_proj | 768 → 1,536 | 48 | 10 → 8 | 1.57 MB | 31.5 MB |
| shared down_proj | 1,536 → 768 | 24 | 21 → 16 | 16 × 32 × 768 × 4 = 1.57 MB | 31.5 MB |
| **Total** | | | | | **173 MB** |

How the q projection gets 12 chunks: 512 ÷ 24 = 21. But 768 isn't a multiple of 21 × 32 = 672, so the count drops one at a time until it is: 12 × 32 = 384, and 768 = 2 × 384.

Three pieces don't use split-K:
- the output layer, which already has 128,256 ÷ 32 ≈ 4,008 tiles, so 512 ÷ 4,008 rounds down to 1 chunk;
- the routed experts, which use a different, batched kernel;
- the router, which isn't quantized.

### Checking the calculation against the counters

**By knock-out**, the writes saved when a piece is removed (measured):

| Piece | Worked out | Measured |
|---|---:|---:|
| q, k and v projections | 55 MB | 57 MB |
| Shared expert | 94 MB | 102 MB |

The measured values are a little higher because removing a piece also removes its ordinary outputs.

**By whole-step totals** (estimate): 8 requests write about 20 MB without split-K. Four times as many requests would write about 80 MB of ordinary output. Adding the 173 MB of partial results gives about 253 MB, against the 251–266 MB measured.

### Is it a loss?

Not shown. The partial results cost 173 MB of writes, their read-back, and 7 extra adding kernels per layer, 140 per step. In return, these products run on hundreds of teams instead of 24–48. Each layer's buffer is only 0.8–1.6 MB, so the read-back may come mostly from the cache. Comparing step times with split-K reduced or turned off would answer it. That needs a patched MLX, because I know of no setting that turns split-K off.

## 7. The 32-request step, roughly accounted

Plain MLX, 128-token prompts:

```
step reads (measured):                  1,557 MB
model minimum:                            731 MB
extra:                                    826 MB
  routed experts, read per pair:          465 MB   measured
  attention, re-reading:                  105 MB   measured
  not attributed:                        ~255 MB   estimate
```

The unattributed part likely includes the split-K partial results read back, and the results passed between kernels, which the minimum doesn't count. Knock-outs of small pieces at 32 requests can't separate it: removing a piece changes what the rest of the step finds in the cache, so those differences are unreliable. For example, removing the shared expert raised the step's reads by 51 MB.

## 8. The inefficiencies, against the fix list

Fix numbers refer to [decode-bottlenecks.md](decode-bottlenecks.md#what-to-do-in-priority-order), updated in [step-breakdown.md](step-breakdown.md#6-what-this-changes-in-the-fix-list).

| Inefficiency | Where | Measured cost | Fix | Expected effect |
|---|---|---|---|---|
| 867–1,088 small kernels, 8–27% of thread slots filled | whole model | every step at 5–9 times its lower bound | 1 (compile the mixture-of-experts block), 2 (fuse the projections) | Compiling the block cut it by 29% at one request (measured, whole 20-layer stack compiled as one function) |
| Routing and combine cost more than the expert math | 340 kernels per step | 1.6 ms of a 4.75 ms step at one request | 1 | part of the above |
| CPU encoding almost as long as the GPU's step | vllm-metal's step submit | 4.7 of 5.5 ms at one request | 3, and fewer kernels from 1 and 2 | needed once 1 and 2 shorten the GPU's step |
| Routed experts read per (token, expert) pair, and slowly | MLX's kernel choice below 60 rows | 14.7 ms of a 28.2 ms step; 465 MB extra per step | 4 (read each expert once) | 25–36% faster on equal data (measured). Real routing has fewer pairs per expert, so likely less. |
| vLLM's attention re-reads the stored heads 3 times, plus Python | vllm-metal's attention | +6.5 ms against plain MLX at 1,024-token prompts | 5 (a kernel for the 12/4 head layout) | up to −6.4 ms (estimate) |
| Split-K partial results | MLX's quantized products from 14 rows | 173 MB written per step | a lead, not yet a fix | untested |

Together, fixes 1–3 are estimated to take the one-request step from 5.5 to about 3.5–4 ms. Fixes 4–5 are estimated to take 32 requests with 1,024-token prompts from 40 to about 28–30 ms. Both are estimates from separate tests whose gains may overlap.

None of this makes the step bandwidth-bound. Fewer, larger kernels (fixes 1 and 4) stay the main lever.

## What is not the bottleneck

All measured:

- **vLLM's scheduler:** 0.06 ms per step at one request, 0.30 ms at 32.
- **The rest of vLLM's runner:** 0.5–1.6 ms per step.
- **Small operations repeated in every step,** such as format conversions and index ranges. Removing 140 of them saved 1.2%, because they run alongside the main chain.
- **The layout of Meta's expert weights:** within 4% for the grouped kernel.

## What remains uncertain

- **Why small kernels move data slowly.** That too few threads limit the reads in flight is the standard explanation. The counters show the low occupancy, not the mechanism.
- **Split-K's effect on step time** is not measured.
- **The ~255 MB of unattributed reads** at 32 requests.
- **How the fixes combine.** The estimates come from separate tests.
- **Real text.** All prompts were random tokens. Real text may route to a different number of distinct experts per step, which changes the experts' bytes.
- **What the read counter counts.** It is taken as reads from main memory, which the 512 MB calibration supports, but Apple doesn't document it.

The limits of each measurement are listed in [step-breakdown.md](step-breakdown.md#limits).
