# Inside a decode step, piece by piece

[Where a decode step's time goes](decode-bottlenecks.md) found that a decode step is hundreds of small GPU kernels, and that vLLM adds little over plain MLX. It timed parts of the step on their own. This document measures every piece inside the real step, in vLLM and in plain MLX side by side:

- **inside vllm-metal's model runner:** the scheduler, building the batch, calling the model, encoding the GPU work, sampling and waiting;
- **inside the model:** each projection, norm, RoPE, cache write and attention kernel, the router, the routed experts, the combine, the shared expert and the output layer.

For each piece it reports the GPU kernels it launches, the CPU time it takes to build, and what the step saves without it. It then lists the work that is repeated in every step for nothing, and measures what removing it saves.

Everything was measured on 2026-11-01, and the GPU counters on 2026-11-02, on the baseline's machine, software and int4 model ([method](../benchmark/method.md)). Each claim is marked **measured** (timed here), **from the code** (read in the source) or **estimate** (reasoned from those two). The scripts are in [bench/micro/](../../bench/micro/): `breakdown.py`, `breakdown_tables.py`, `repeat_fixes.py`, `lm_head_shapes.py` and `gpu_counters.py`. The raw data is in [results/runs/2026-11-01-breakdown/](../../results/runs/2026-11-01-breakdown/) and [results/runs/2026-11-02-gpu-counters/](../../results/runs/2026-11-02-gpu-counters/), and every number below is in their `tables.md` files ([breakdown](../../results/runs/2026-11-01-breakdown/tables.md), [counters](../../results/runs/2026-11-02-gpu-counters/tables.md)).

## Summary

- **The scheduler and the rest of vLLM's runner are small.**
  - vLLM's scheduler takes 0.06 ms per step with one request and 0.30 ms with 32.
  - Everything the runner does outside the model takes 0.5 ms with one request and 1.6 ms with 32.
  - The CPU time goes to the model: recording its graph (2.4–3.1 ms) and encoding it into GPU commands (4.1–6.5 ms).
- **vLLM's step is 7–20% longer than plain MLX's, and the difference is in attention.**
  - On the same harness and model, vLLM is 19% slower with one request, 7–8% slower with 8 and 32, and 20% (6.5 ms) slower with 1,024-token prompts at 32 requests. The earlier "5–10%" compared different harnesses, on short prompts only.
  - vllm-metal's attention path runs about 1 ms of Python per step outside the kernels, against 0.23 ms in plain MLX.
  - It adds 60 copies of q, k and v per step, and 120 from 2 rows up.
  - Its attention kernel falls further behind as the context grows.
- **With one request, plain MLX's step is its chain of kernels.**
  - Most pieces cost 4–7 µs per kernel where they run.
  - The mixture-of-experts block's bookkeeping, routing and combine, costs 1.6 ms. That is more than its expert products, at 0.9 ms.
  - The output layer reads its 61.6 MB at about 158 GB/s, close to peak.
- **With 32 requests, the routed experts take half the step:** 14.7 of 28.4 ms in plain MLX and 14.9 of 30.3 ms in vLLM. Attention comes next, at 5.6 and 7.5 ms.
- **The GPU is busy 92–98% of the time but moves only 17–28% of its peak memory bandwidth** (measured with the GPU's own counters).
  - With one request each weight is read once: 193 MB per step against the 180 MB the model needs. Only the output layer comes close to peak, at about 160 GB/s. The other pieces' kernels read too little each to get past their fixed cost.
  - From 8 requests up, the step reads 1.7–2.2 times what the model needs. At 32 requests the routed experts read 908 MB from memory against the 443 MB they need: about half their repeat reads per (token, expert) pair reach memory, the rest come from cache.
  - At 32 requests MLX's split quantized products also write about 170 MB of partial sums per step.
  - While busy, the GPU's arithmetic units are 14–45% used and most of its thread slots are empty.
- **The repeated work is real but cheap.**
  - Three changes that leave the outputs bit-identical remove 140 of vLLM's 967 kernel operations per step.
  - They save 0.17 ms per step at 8 requests (1.2%). Where a gain stood clear of the run-to-run noise, it was at most 1.5% of the step.
  - Kernels that wait on nothing (an index range, a cast of a stored parameter) run alongside the chain, so removing them saves little.

## How the step was cut into pieces

`bench/micro/breakdown.py` runs the int4 model in two engines, each in its own process:

- **Plain MLX:** the mlx_lm port with its own key/value cache. Like vllm-metal, it builds each step while the GPU runs the previous one.
- **vLLM:** vLLM 0.32.0 inside the measuring process, with vllm-metal's model runner, paged attention and sampler, prefix caching off and a 1 GiB key/value cache.

Both get the sweep's workload:
- random 128-token prompts at 1, 8 and 32 decode rows;
- 1,024-token prompts at 32 rows;
- 256 tokens generated greedily for each.

Only steady decode steps count: every row is decoding and none is joining or leaving. The first 5 of each run are dropped, which leaves 248 per run.

Wrappers installed at runtime give each piece a named **span**. No file in vLLM, vllm-metal or the port is edited. The mixture-of-experts block's `__call__` (`mobilemoe.py:217-232`) is replaced at runtime by a copy split into router, top-4 selection, weights and combine. Before measuring, the harness checks that the copy gives bit-identical outputs to the port's at 1, 8 and 32 rows.

```
vLLM engine step                                   plain MLX loop
  schedule            scheduler picks the requests
  execute_model       runner builds the batch
    start_forward     inputs and attention metadata
      target_forward  the model records its graph      target_forward
      submit          mx.async_eval: encode it         submit
  sampling, sampler   next tokens (argmax)             sampling
  wait_gpu            the previous step's tokens       wait_gpu
  update_from_output, process_outputs

the model, both engines
  embed
  20 x layer:  input_norm -> attention -> add -> post_norm -> moe -> add
    attention: q_proj, k_proj, v_proj -> q_norm, k_norm -> rope
               -> kv_write -> attn_kernel -> o_proj
               (vLLM also: qkv_prep, metadata, contiguous)
    moe:       router -> topk -> weights -> experts (gate, up, down products)
               -> combine, adding the shared expert (gate, up, down products)
  final_norm -> lm_head
```

Each run uses one mode:

| Mode | What changes | What it measures |
|---|---|---|
| bare | only the step boundaries are hooked | the step as served |
| off | every wrapper installed but idle | the reference for knock-outs at 32 rows |
| cpu | every span timed | each piece's CPU time |
| sync | each span's inputs and outputs evaluated on the spot | each piece's kernel operations, counted on one step |
| knock-out | one piece skipped: each call returns the output it gave the first time, tied to its current inputs so those are still computed | what the step saves without the piece |
| control | the knock-out's bookkeeping, with the piece still running | the reference for knock-outs at 1 row |

Every mode and setting ran 3 times, in a different shuffled order each time. Knock-outs ran at 1 and 32 rows with 128-token prompts.

**Terms**

- **Span:** one named piece of the step, timed from entry to exit. Its **self time** leaves out the spans inside it.
- **Recording the graph:** the model's Python code records MLX operations. No GPU work happens yet.
- **Encoding:** `mx.async_eval` turns the recorded operations into GPU commands on the CPU, and hands them to the GPU.
- **Kernel operation:** an operation in the graph that launches a GPU kernel. Operations that only change how an array is viewed launch none and are not counted; these are reshape, transpose, slice and squeeze.
- **Saved:** the step period with the piece minus the step period without it, so it includes the piece's CPU time, its GPU time, or both, depending on which side sets the pace.

## 1. vLLM against plain MLX, on the same harness

**Measured.** The median of 3 repeats, in bare mode:

| Decode rows | Prompt tokens | Plain MLX | vLLM | vLLM's extra | vLLM served in the sweep |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 4.75 ms | 5.64 ms | +0.89 ms (+19%) | 5.49 ms |
| 8 | 128 | 13.37 ms | 14.30 ms | +0.93 ms (+7%) | 14.51 ms |
| 32 | 128 | 28.19 ms | 30.47 ms | +2.28 ms (+8%) | 30.04 ms |
| 32 | 1,024 | 33.20 ms | 39.71 ms | +6.51 ms (+20%) | 39.99 ms |

- **The harness measures what vLLM serves.** In-process vLLM runs within 3% of the served sweep.
- **With one request**, vLLM's least disturbed repeat ran at 5.41 ms (+14%) ([Limits](#limits)).
- **This replaces finding 1 of [decode-bottlenecks.md](decode-bottlenecks.md#1-vllms-engine-adds-only-510).** That finding compared a separate plain-MLX script with the sweep, and only on 128-token prompts.
- **Longer prompts widen the gap.** Going from 128- to 1,024-token prompts at 32 rows adds 5.0 ms to plain MLX's step and 9.2 ms to vLLM's. Only attention reads the longer context, so the extra 4.2 ms is attention (estimate).

## 2. Inside vllm-metal's model runner

**Measured** in cpu mode: the mean time per steady step, in ms, over 3 repeats.

| Part of the step | 1 row | 8 rows | 32 rows | 32 rows, 1,024-token prompts |
|---|---:|---:|---:|---:|
| scheduler: pick the requests (`schedule`) | 0.06 | 0.12 | 0.30 | 0.48 |
| runner: build the batch, inputs and attention metadata | 0.13 | 0.22 | 0.44 | 0.77 |
| model: record the graph ([section 3](#3-inside-the-model)) | 2.41 | 2.45 | 3.12 | 5.00 |
| `mx.async_eval`: encode the GPU work | 4.10 | 4.70 | 6.46 | 7.16 |
| `mx.async_eval`: wait for the GPU | 0.41 | 6.58 | 19.39 | 26.90 |
| sampling: next tokens and their bookkeeping | 0.10 | 0.13 | 0.20 | 0.25 |
| wait for the previous step's tokens | 0.02 | 0.03 | 0.04 | 0.06 |
| scheduler and output processing after the step | 0.05 | 0.16 | 0.37 | 0.75 |
| everything else | 0.15 | 0.21 | 0.20 | 0.29 |
| **step** | **7.44** | **14.59** | **30.53** | **41.66** |

What it shows:

- **The scheduler is never the problem.** All of the runner's work outside the model is 0.5 ms per step with one request, 0.9 ms with 8, 1.6 ms with 32, and 2.6 ms with 32 long prompts.
- **With 8 or more requests, the GPU sets the pace.** The CPU spends 6.6 of 14.6 ms with 8 rows, and 19.4 of 30.5 ms with 32, waiting inside `mx.async_eval` for the GPU to catch up. The runner's own work happens in that time and costs nothing.
- **With one request, the CPU sets the pace.**
  - The CPU waits only 0.41 ms per step for the GPU, and 0.06 ms in the least disturbed repeat.
  - Plain MLX waits 0.67 ms, so its step is set by the GPU, with CPU to spare.
  - In vLLM, every microsecond of CPU work, the runner's 0.5 ms included, lengthens the step.
- **Encoding is the largest CPU cost.** It takes 4.1 ms with one request, against 2.6 ms in plain MLX ([section 4](#4-where-vllm-differs-from-plain-mlx)).

With one request, the instrumentation itself lengthens vLLM's step (7.44 ms here against 5.64 ms bare), so the figures in that column are upper bounds. With 8 and 32 requests it changes nothing: the medians are 14.31 against 14.30 ms, and 30.32 against 30.47 ms.

## 3. Inside the model

**Measured.** For each piece:
- its kernel operations per step, counted on one step;
- the CPU time to record it, in cpu mode, per step;
- what the step saves when it is knocked out.

Pieces marked with a dash are parts of the piece above them.

### With one request

| Piece | Kernels, plain / vLLM | CPU µs, plain / vLLM | Saved ms, plain | Saved ms, vLLM |
|---|---:|---:|---:|---:|
| attention, all of it | 220 / 320 | 558 / 1,340 | 1.21 | 2.0–2.6 |
| – q, k and v projections | 60 / 60 | 52 / 71 | 0.39 | noise |
| – q and k norms | 40 / 40 | 42 / 60 | 0.28 | noise |
| – RoPE | 40 / 40 | 60 / 255 | 0.07 | 0.4 |
| – copies of q, k and v | 0 / 60 | 0 / 25 | | noise |
| – cache write | 40 / 20 | 134 / 31 | | noise |
| – attention kernel | 20 / 20 | 21 / 30 | 0.41 | 0.8–1.0 |
| – o projection | 20 / 20 | 17 / 30 | 0.26 | noise |
| input and post-attention norms | 40 / 40 | 26 / 43 | 0.27 | noise |
| mixture of experts, all of it | 560 / 560 | 688 / 902 | 2.94 | 3.1–3.3 |
| – routing: router, top-4, weights | 240 / 240 | 173 / 255 | 1.02 | 1.1–1.3 |
| – routed experts | 140 / 140 | 220 / 293 | 0.92 | 0.5–0.6 |
| – combine | 100 / 100 | 75 / 91 | 0.61 | noise |
| – shared expert | 80 / 80 | 134 / 163 | 0.51 | 0.7–0.9 |
| output layer | 1 / 1 | 1 / 1 | 0.39 | noise |

**How the vLLM column was read.** vLLM's one-request runs were disturbed by paging ([Limits](#limits)), so its savings are given as the range between two estimates: the median of the repeats and the fastest repeat. "Noise" marks pieces where an estimate is zero or below, or where the two differ by more than 0.3 ms and by more than a third of the larger.

What it shows, for plain MLX:

- **Most pieces cost 4–7 µs per kernel.**
  - Attention: 1.21 ms for 220 kernels, 5.5 µs each.
  - Routing: 1.02 ms for 240, 4.3 µs each.
  - Routed experts: 0.92 ms for 140, 6.6 µs each.
  - Combine: 0.61 ms for 100, 6.1 µs each.

  That matches the fixed cost of about 7 µs per dependent kernel measured in [decode-bottlenecks finding 2](decode-bottlenecks.md#2-with-one-request-the-step-is-about-870-kernels-in-a-chain). There are exceptions:
  - The attention kernel (20 µs each) and the o projection (13 µs) cost more.
  - RoPE's 40 kernels cost only 0.07 ms in all, because the rotations of q and of k don't wait for each other.
- **The mixture-of-experts block's bookkeeping costs more than its arithmetic.**
  - Routing and combine are 340 kernels and 1.63 ms, a third of the step.
  - The four routed experts' products cost 0.92 ms; the shared expert's 0.51 ms.
  - Fusing the bookkeeping ([decode-bottlenecks fix 1](decode-bottlenecks.md#what-to-do-in-priority-order)) targets the largest piece.
- **The output layer runs near peak bandwidth where it sits.**
  - It is one kernel taking 0.39 ms to read 61.6 MB: about 158 GB/s, against 200 GB/s peak.
  - Timed on its own, [decode-bottlenecks section 7](decode-bottlenecks.md#7-smaller-observations) measured 0.93 ms (66 GB/s) and suspected low GPU clocks. This confirms that suspicion.

### With 32 requests

| Piece | Kernels, plain / vLLM | CPU µs, plain / vLLM | Saved ms, plain / vLLM | Share of the step, plain / vLLM |
|---|---:|---:|---:|---:|
| attention, all of it | 220 / 320 | 1,062 / 1,716 | 5.60 / 7.48 | 20% / 25% |
| – q, k and v projections | 60 / 60 | 95 / 80 | 1.67 / 2.06 | 6% / 7% |
| – q and k norms | 40 / 40 | 80 / 69 | 0.25 / 0.05 | 1% / 0% |
| – RoPE | 40 / 40 | 115 / 262 | 0.11 / 0.44 | 0% / 1% |
| – copies of q, k and v | 0 / 60 | 0 / 35 | – / 0.24 | – / 1% |
| – cache write | 40 / 20 | 264 / 30 | – / 0.59 | – / 2% |
| – attention kernel | 20 / 20 | 42 / 48 | 3.12 / 4.33 | 11% / 14% |
| – o projection | 20 / 20 | 33 / 30 | 0.75 / 1.10 | 3% / 4% |
| input and post-attention norms | 40 / 40 | 51 / 51 | 0.34 / 0.37 | 1% / 1% |
| mixture of experts, all of it | 680 / 680 | 1,404 / 1,201 | 17.09 / 20.35 | 60% / 67% |
| – routing | 240 / 240 | 338 / 348 | 1.68 / 1.53 | 6% / 5% |
| – routed experts | 260 / 260 | 537 / 445 | 14.67 / 14.85 | 52% / 49% |
| – combine | 100 / 100 | 135 / 109 | 0.33 / 0.69 | 1% / 2% |
| – shared expert | 80 / 80 | 243 / 188 | 2.90 / 3.13 | 10% / 10% |
| output layer | 1 / 1 | 2 / 2 | 2.50 / 5.08 | 9% / 17% |

The share is the saving over the step with every wrapper idle (28.40 ms in plain MLX, 30.31 ms in vLLM). With 32 rows both engines' repeats agree within 2.5%.

What it shows:

- **The routed experts take half the step,** 14.7 ms in plain MLX and 14.9 ms in vLLM.
  - The step needs 40 distinct experts per layer, 443 MB, which would take 2.2 ms at peak bandwidth.
  - The per-pair kernel reads each expert once per (token, expert) pair instead ([decode-bottlenecks finding 5](decode-bottlenecks.md#5-with-many-requests-experts-are-read-once-per-token-expert-pair)).
- **Attention is next:** 5.6 ms in plain MLX and 7.5 ms in vLLM. Its kernel is 3.1 and 4.3 ms of that.
- **The routing bookkeeping stops mattering**, at 1.5–1.7 ms. Its kernels don't grow with the number of rows, while the experts' work does.
- **Savings don't add up across pieces.** Removing a large piece changes how the rest of the step overlaps. Each knock-out also replays the same outputs, which changes the tokens generated and so the experts that later steps use. Read each figure as that piece's cost where it runs, not as a share of a whole.
- **vLLM's output layer saves twice as much as plain MLX's (5.1 against 2.5 ms), and the kernel is not the reason.**
  - Timed on its own, the output layer takes 2.35 ms for 32 rows in either engine's layout, with identical results (`lm_head_shapes.py`). vllm-metal hands the model one packed sequence (1 × 32 rows), plain MLX 32 sequences of 1.
  - The rest of vLLM's 5.1 ms comes from knocking it out inside vLLM's step, not from the layer itself. It is not explained here.
  - vLLM's top-level savings also add up to 33.3 ms, more than its 30.3 ms step; plain MLX's add up to 25.5 ms, less than its step, as overlapping pieces should. So at least one of vLLM's 32-row knock-outs removes more than its own piece's work.

### Memory bandwidth and compute, piece by piece

**Derived from the measurements above**: the bytes the model must read, over measured times. The [next subsection](#memory-traffic-counted-by-the-gpu) checks them against the traffic the GPU counted. For each piece:

- **MB to read:** the weights it uses in one step, summed over the 20 layers. Only the experts the step's tokens picked count, each once. The attention kernel also reads the key/value cache up to each row's position (bfloat16, 1 KB per token per layer).
- **GFLOP:** its multiply-adds, counted as 2 operations each.
- **Achieved:** those bytes and operations over what the step saves without the piece, so the rate at which the piece does its work where it runs. Peaks are 200 GB/s and 5.2 TFLOP/s.

Activations are not counted, and a piece that reads the same weights twice is counted once, so the bandwidths are lower bounds on the real traffic. vLLM at one request is left out: its step is set by the CPU, so its savings are CPU time.

**With one request (plain MLX):**

| Piece | MB to read | ms at peak bandwidth | Saved ms | GB/s | Share of peak bandwidth |
|---|---:|---:|---:|---:|---:|
| attention, all of it | 24.9 | 0.12 | 1.21 | 21 | 10% |
| – q, k and v projections | 12.3 | 0.06 | 0.39 | 32 | 16% |
| – attention kernel | 5.3 | 0.03 | 0.41 | 13 | 6% |
| – o projection | 7.4 | 0.04 | 0.26 | 29 | 14% |
| mixture of experts, all of it | 92.2 | 0.46 | 2.94 | 31 | 16% |
| – routing | 3.7 | 0.02 | 1.02 | 4 | 2% |
| – routed experts | 44.2 | 0.22 | 0.92 | 48 | 24% |
| – shared expert | 44.2 | 0.22 | 0.51 | 86 | 43% |
| output layer | 61.6 | 0.31 | 0.39 | 158 | 79% |
| whole step | 178.7 | 0.89 | 4.75 | 38 | 19% |

**With 32 requests** (MB and GFLOP from plain MLX's step; vLLM's own are within 10%, with 36 distinct experts per layer against 40):

| Piece | MB to read | GFLOP | Saved ms, plain / vLLM | GB/s, plain / vLLM | Share of peak compute, plain / vLLM |
|---|---:|---:|---:|---:|---:|
| attention, all of it | 188.4 | 2.52 | 5.60 / 7.48 | 34 / 25 | 9% / 6% |
| – q, k and v projections | 12.3 | 1.26 | 1.67 / 2.06 | 7 / 6 | 14% / 12% |
| – attention kernel | 168.8 | 0.51 | 3.12 / 4.33 | 54 / 39 | 3% / 2% |
| – o projection | 7.4 | 0.75 | 0.75 / 1.10 | 10 / 7 | 19% / 13% |
| mixture of experts, all of it | 491.4 | 9.12 | 17.09 / 20.35 | 29 / 22 | 10% / 9% |
| – routing | 3.7 | 0.06 | 1.68 / 1.53 | 2 / 2 | 1% / 1% |
| – routed experts | 443.5 | 4.53 | 14.67 / 14.85 | 30 / 27 | 6% / 6% |
| – shared expert | 44.2 | 4.53 | 2.90 / 3.13 | 15 / 14 | 30% / 28% |
| output layer | 61.6 | 6.30 | 2.50 / 5.08 | 25 / 12 | 48% / 24% |
| whole step | 741.4 | 17.94 | 28.19 / 30.47 | 26 / 23 | 12% / 11% |

**The whole step at every setting**, over its bare period. The lower bound is the step at peak bandwidth or peak compute, whichever is slower:

| Requests × prompt tokens | MB to read, plain / vLLM | GB/s, plain / vLLM | Share of peak bandwidth, plain / vLLM | Times the lower bound, plain / vLLM |
|---|---:|---:|---:|---:|
| 1 × 128 | 179 / 179 | 38 / 32 | 19% / 16% | 5.3 / 6.3 |
| 8 × 128 | 393 / 389 | 29 / 27 | 15% / 14% | 6.8 / 7.4 |
| 32 × 128 | 741 / 701 | 26 / 23 | 13% / 12% | 7.6 / 8.7 |
| 32 × 1,024 | 1,290 / 1,275 | 39 / 32 | 19% / 16% | 5.1 / 6.2 |

What it shows:

- **The step uses 12–19% of peak bandwidth in either engine.** That matches the served sweep's 12–17% ([baseline](../benchmark/findings.md)), whose byte counts (179, 720 and 1,248 MB) agree with these within 4%.
- **With one request, a piece gets closer to peak the more bytes each of its kernels reads.**
  - The output layer is one kernel reading 61.6 MB: 158 GB/s.
  - The shared expert reads 0.74 MB in each of its three kernels per layer: 86 GB/s.
  - The projections read 0.12–0.37 MB per kernel: about 30 GB/s.
  - Routing reads 3.7 MB across 240 kernels: 4 GB/s.

  At 5–7 µs per kernel, a kernel that reads 0.37 MB cannot move more than 50–75 GB/s, whatever the memory system allows. With one request the limit is the number of kernels, not bandwidth.
- **With 32 requests, nothing reaches either limit, for two different reasons.**
  - **The routed experts read the same weights again and again.** On the 443.5 MB they need, they reach 30 GB/s. But the per-pair kernel reads an expert once per (token, expert) pair: 32 × 4 pairs per layer, 1,416 MB per step. Counted that way they move 97 GB/s, half of peak, and two thirds of those bytes are repeats (estimate). A grouped kernel that reads each expert once targets this ([decode-bottlenecks fix 4](decode-bottlenecks.md#what-to-do-in-priority-order)).
  - **The attention kernel reads the cache at 54 GB/s in plain MLX and 39 GB/s in vLLM.** vllm-metal's kernel reads each key/value head once per query head, 3 times ([decode-bottlenecks finding 6](decode-bottlenecks.md#6-at-long-contexts-attention-reads-each-keyvalue-head-three-times)). Counted that way, it moves about 117 GB/s (estimate).
  - **The dense layers lean towards compute.** The output layer reaches 48% of peak compute in plain MLX (6.3 GFLOP in 2.50 ms), the shared expert 30%, the projections 14–19%. vLLM's output layer shows half that only because of the unexplained 5.08 ms above.

The next subsection checks these figures against the memory traffic the GPU itself counted.

### Memory traffic counted by the GPU

**Measured on 2026-11-02 with the GPU's own counters**, through Xcode's `xctrace` and its Metal GPU Counters instrument. The script is [`bench/micro/gpu_counters.py`](../../bench/micro/gpu_counters.py); the data and every number below are in [results/runs/2026-11-02-gpu-counters/](../../results/runs/2026-11-02-gpu-counters/tables.md). For each setting it:

1. decodes 100 steady steps;
2. pauses, starts the recording and records 0.3 s of idle GPU;
3. lets about one second of steady steps run;
4. pauses again and stops the recording.

The GPU sends a sample about every 20 µs while it has work and none while it is idle. Each sample holds 31 counters, among them:
- **read and write bandwidth:** traffic between the GPU and memory;
- **ALU utilization:** how busy the arithmetic units are;
- **occupancy:** how full the GPU's slots for running threads are;
- **limiters:** how close each unit came to its throughput limit.

The counters cover the whole GPU, not one program.

Terms used below:
- **Read bandwidth is a percentage of the 200 GB/s peak.** Each run first reads a 512 MB buffer over and over. That is larger than any cache, so every byte must come from memory. The bandwidth counter's average, times 2.002, gives the bytes actually read: 2.002 and 2.001 in the two calibration runs.
- **MB read per step** is the average read bandwidth over the window (counting idle gaps as zero) times the step period. It counts traffic that reached memory; reads served by the GPU's caches are not in it (estimate, from the experts below).
- **GPU busy** is the share of the window in which the GPU sent samples.

**The whole step:**

| Engine | Requests × prompt tokens | MB read per step (counter) | MB the model must read | Ratio | GB/s at the unrecorded step time (share of peak) | MB written per step | GPU busy | ALU utilization while busy | Occupancy while busy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 1 × 128 | 193 | 180 | 1.07 | 41 (20%) | 3 | 96% | 17% | 9% |
| plain | 8 × 128 | 685 | 393 | 1.74 | 51 (26%) | 21 | 98% | 45% | 23% |
| plain | 32 × 128 | 1,561 | 731 | 2.14 | 55 (28%) | 266 | 92% | 37% | 26% |
| plain | 32 × 1,024 | 2,297 | 1,275 | 1.80 | not used | 380 | 74% | 38% | 27% |
| vLLM | 1 × 128 | 193 | 180 | 1.07 | 34 (17%) | 3 | 95% | 14% | 8% |
| vLLM | 8 × 128 | 684 | 389 | 1.76 | 48 (24%) | 20 | 96% | 42% | 21% |
| vLLM | 32 × 128 | 1,511 | 692 | 2.18 | 50 (25%) | 251 | 97% | 36% | 24% |
| vLLM | 32 × 1,024 | 2,250 | 1,268 | 1.77 | 57 (28%) | 253 | 97% | 33% | 26% |

The GB/s use the step periods measured without recording (section 1), because recording slows the step a little ([Limits](#limits)). Plain MLX at 32 × 1,024 is left out: recording slowed its steps from 33 to 57 ms, with the GPU idle a quarter of the window.

**Each piece (plain MLX):** the bytes the step stops reading or writing when the piece is knocked out, against the recorded step with every wrapper idle. The time saved is the unrecorded one from the tables above.

| Requests | Piece | MB read saved (counter) | MB it must read (model) | MB written saved (counter) | Saved ms | GB/s while it runs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | q, k and v projections | 14 | 12 | 0 | 0.39 | 35 |
| 1 | attention kernel | 11 | 7 | 0 | 0.41 | 26 |
| 1 | routing | 4 | 4 | 0 | 1.02 | 4 |
| 1 | routed experts | 51 | 44 | 1 | 0.92 | 55 |
| 1 | shared expert | 45 | 44 | 0 | 0.51 | 88 |
| 1 | output layer | 65 | 62 | 0 | 0.39 | 167 |
| 32 | attention kernel | 265 | 160 | 2 | 3.12 | 85 |
| 32 | routed experts | 908 | 443 (1,416 once per pair) | 26 | 14.67 | 62 |
| 32 | q, k and v projections | −11 | 12 | 57 | 1.67 | |
| 32 | shared expert | −51 | 44 | 102 | 2.90 | |

What the counters show:

- **The GPU is never idle, yet uses only a fifth to a quarter of its memory bandwidth (measured).**
  - The GPU is busy 92–98% of every window, in both engines.
  - It moves 17–28% of peak bandwidth, counting every byte it really reads.
  - While busy, its arithmetic units are 14–17% used with one request and 33–45% from 8 requests up.
  - Occupancy is 8–9% and 21–27%: most of the GPU's thread slots are empty.
  - No counter is near its limit. The highest are the buffer-read and last-level-cache limiters, at 53–56% with 32 requests.
  - This is what a chain of small dependent kernels looks like from the hardware side: each kernel launches too few threads to fill the GPU, and the next one waits for it.
- **With one request, the derived figures above hold (measured).**
  - Both engines read 193 MB per step, 7% more than the 180 MB of weights and cache the model needs. The difference is activations and small reads.
  - Every piece reads within 1–7 MB of what the model says, so each weight is read once.
  - The output layer's 65 MB over its 0.39 ms is 167 GB/s, against the derived 158.
- **From 8 requests up, the step reads 1.7–2.2 times what the model needs (measured).** It is the bandwidth spent on reading again, not the bytes needed, that grows with the requests.
- **The routed experts' repeat reads come half from memory, half from cache.**
  - **Measured:** at 32 requests they read 908 MB from memory, against the 443 MB they need and the 1,416 MB they would read once per (token, expert) pair. That is 62 GB/s over their 14.67 ms, against the derived 30 (on the bytes needed) and 97 (counting every pair).
  - **Estimate:** about half of the 973 MB of repeat reads reached memory (465 MB); the GPU's caches served the rest.
  - A grouped kernel that reads each expert once targets those 465 MB per step, 30% of the step's reads.
- **Plain MLX's attention kernel reads 265 MB at 32 requests, 1.7 times the cache it needs (measured).**
  - It moves 85 GB/s over its 3.12 ms.
  - **Estimate:** the extra reads are consistent with key/value heads being read for more than one query head, partly from cache. This is plain MLX's own kernel, not vllm-metal's, whose triple reads ([decode-bottlenecks finding 6](decode-bottlenecks.md#6-at-long-contexts-attention-reads-each-keyvalue-head-three-times)) were not knocked out here.
- **With 32 requests, the step writes 251–266 MB, and most of it is partial sums (measured and from the code).**
  - **From the code:** for a quantized product with a few rows, MLX splits the long inner dimension into `split_k` chunks. Each chunk writes its own float32 copy of the output, of `split_k × rows × outputs` numbers. A second kernel then reads them back and adds them up. MLX picks `split_k` to give the GPU about 512 groups of threads to run (`mlx/backend/metal/quantized.cpp:1166-1226`, chosen at `:1909-1915`). On this GPU (`applegpu_g13s`), MLX switches to this kernel at 14 rows for these layer sizes (`quantized.cpp:89-125`). That is why the step writes 20–21 MB at 8 requests and 251–266 MB at 32.
  - **From the code, worked out:** for MobileMoE-S at 32 rows this gives `split_k` = 12, 24 and 24 for the q, k and v projections, 8, 8 and 16 for the shared expert's three products, and 12 for the o projection. The temporaries come to 55 MB, 94 MB and 24 MB per step: 173 MB.
  - **Measured:** knocking out the q, k and v projections cuts the writes by 57 MB, and the shared expert by 102 MB. The output layer has 4,000 tiles of outputs, so it is not split, and it cuts them by only 7 MB, about the size of its logits.
  - **Not measured:** whether the partial sums are read back from memory or from cache. Nor has a smaller `split_k` been tried, which would trade these bytes for fewer threads running at once.
- **Not resolved: the small pieces at 32 requests.**
  - Knocking out the q, k and v projections or the shared expert raises the step's reads, by 11 and 51 MB, although each needs 12 and 44 MB of weights.
  - Two recordings of the same step agree to 0.5% (1,561 and 1,555 MB), so this is not counter noise.
  - **Estimate:** removing a piece changes what stays in the GPU's caches for the rest of the step. Changes below about 60 MB are therefore not attributed to one piece.
  - The output layer's knock-out (197 MB) is left out too: its recording ran slower than the step with the layer still in, with the GPU idle 19% of the time.

## 4. Where vLLM differs from plain MLX

**Kernels (measured, one counted step).** vLLM's step has 968 kernel operations with one request, against plain MLX's 867, and 1,088 against 987 with 32. The whole difference is in attention:

| Operation | Per step | Where | Launches a kernel? |
|---|---:|---|---|
| `Gather` | +60 | `queries[0]`, `keys[0]` and `values[0]` in vllm-metal's `sdpa_forward` (`attention/impls/sdpa.py:722-724`) | yes. In Python, `x[0]` becomes a gather, which copies, not a view (MLX `python/src/indexing.cpp:105-110,164`) |
| `Contiguous` | +60 | `mx.contiguous(...)` on the same lines | only from 2 rows up. With one row the transposed array is already laid out in order, so MLX shares its memory (`mlx/backend/gpu/primitives.cpp:50-62`, `mlx/backend/common/utils.h:204-206`) |
| `ReshapeAndCache` in place of 2 `SliceUpdate` | −20 | the cache write | yes |
| `PagedAttention` in place of `ScaledDotProductAttention` | 0 | the attention kernel | yes |
| `AsType` | +1 | the runner's inputs | yes |

vLLM's graph also has 140–220 more view operations per step (706 against 566 with one row): squeezes, transposes and cache views. They launch no kernel but cost CPU to record and encode.

**CPU (measured, cpu mode).** With one request, vLLM spends 2–3 ms more CPU per step than plain MLX:

- **0.5 ms in the runner** ([section 2](#2-inside-vllm-metals-model-runner)).
- **0.7–1.0 ms more recording the model's graph.** Almost all of it is attention: vllm-metal's attention path takes 1,340 µs per step against plain MLX's 558:
  - `sdpa_forward` and the wrapper's own code: 564 µs, against 233 µs for plain MLX's whole attention code;
  - `prepare_sdpa_qkv`, around the projections and norms: 250 µs;
  - the RoPE helper, around the RoPE kernels: 161 µs;
  - the metadata lookup and the copies: 49 µs.

  vLLM's cache write is cheaper: 31 µs against plain MLX's 134 µs.
- **1.1–1.5 ms more encoding**: 4.10 against 2.61 ms, and 3.68 ms in vLLM's least disturbed repeat.
  - vLLM's graph has 101 more kernel operations and 140 more views to encode.
  - vllm-metal's own two operations cost no more to encode than MLX's. Run piece by piece, its attention kernel took 0.62 ms per step to encode against 0.60 ms, and its cache write 0.58 against 0.76 ms.
  - Part of the gap may be paging, which counts as CPU time. Plain MLX's one-request runs saw no swapping; vLLM's always did.

Plain MLX's step with one request has about 0.7 ms of CPU to spare. vLLM's extra CPU uses that up and then lengthens the step, which is why one request shows the largest relative gap.

**GPU (measured, knock-outs at 32 rows, where the GPU sets the pace).**
- vLLM's attention costs 7.5 ms per step in place, against plain MLX's 5.6 ms.
- Its kernel accounts for 4.3 ms against 3.1 ms, and the 60 copies in front of it for 0.24 ms.
- With 1,024-token prompts the step gap reaches 6.5 ms ([section 1](#1-vllm-against-plain-mlx-on-the-same-harness)). This is consistent with vllm-metal's decode kernel reading each key/value head three times ([decode-bottlenecks finding 6](decode-bottlenecks.md#6-at-long-contexts-attention-reads-each-keyvalue-head-three-times)), where plain MLX runs MLX's own attention kernel (estimate).

## 5. Work repeated in every step

**From the code.** Five pieces of work are redone in every step although their result is already known or already in the right form:

| Repeated work | Where | Per step | Exact fix |
|---|---|---:|---|
| Row indices for every expert product. `gather_qmm` builds `reshape(arange(n))` when no `lhs_indices` are given | MLX `ops.cpp:79-108` (`indices_or_default`), called at `ops.cpp:5789`. The port passes only `rhs_indices` (`mobilemoe.py:172-184`) | 60 `Arange` | Below 16 rows, pass the same indices, built once per shape. From 16 rows, MLX only uses its sorted kernels when no indices are given (`ops.cpp:5824-5825`, `quantized.cpp:2012`, `:2233`), so the fix belongs in MLX |
| `expert_bias` converted from bfloat16 to float32 in every layer | Meta's checkpoint stores it in bfloat16 and the router weight in float32. `convert_qat.py:81` copies it unchanged, and the router adds the two (`mobilemoe.py:223`) | 20 `AsType` | Store it in float32 at conversion. Converting bfloat16 to float32 is exact |
| q, k and v copied with `[0]`, then copied again into the kernel's layout | vllm-metal `attention/impls/sdpa.py:722-724` | 60 `Gather`, plus 60 copies from 2 rows up | `squeeze(0)` instead of `[0]`. It gives a view of the same values, already in the kernel's layout, so the `Contiguous` copies stop copying too |
| The experts' outputs converted to float32 and back around the weighting | `mobilemoe.py:231`: bfloat16 outputs times float32 weights | 40 `AsType` | none that is exact on its own. Compiling the block fuses them ([decode-bottlenecks fix 1](decode-bottlenecks.md#what-to-do-in-priority-order)) |
| RoPE positions rebuilt in every layer | vllm-metal `attention/impls/varlen_rope_compat.py:221-228`: a Python check over the rows and a new `mx.array(offsets)` per layer, the same for all 20 layers | no kernels, 161–203 µs of CPU in the RoPE helper | build them once per step |

**Measured: what the three exact fixes save.**
- `repeat_fixes.py` applies each fix at runtime and times the step: 3 repeats, in shuffled order, with Docker's VM quit.
- Before timing, it checks that each change gives bit-identical outputs to the unchanged model, and counts each change's kernel operations on one step.
  - **Plain MLX:** the logits after 9 decode steps.
  - **vLLM:** 24 greedy tokens for 1 and for 32 requests.
- Every change passed.
- Its operation counts are one lower than `breakdown.py`'s (967 against 968 for vLLM with one request, 986 against 987 for plain MLX with 32): the two scripts draw the edge of the step one operation apart.

| Change | Kernel operations removed, 1 / 32 rows | Plain MLX, ms per step at 1 / 8 / 32 rows | vLLM, ms per step at 1 / 8 / 32 rows |
|---|---:|---:|---:|
| `expert_bias` in float32 | 20 / 20 | −0.04 / −0.03 / +0.10 | −0.05 / −0.09 / −0.00 |
| row indices passed in, below 16 rows | 60 / 0 | −0.02 / −0.04 / +0.04 | −0.26 / +0.04 / +0.13 |
| q, k and v as views (vLLM) | 60 / 60, plus 60 copies that no longer copy | | −0.59 / −0.21 / −0.08 |
| all of them | 80 / 20 in plain MLX, 140 / 80 in vLLM | −0.06 / −0.07 / +0.05 | −0.35 / −0.17 / −0.33 |

The repeats' spread around the unchanged step is the yardstick:

| Unchanged step | Plain MLX | vLLM |
|---|---|---|
| 1 row | 4.68–4.73 ms | 5.36–6.20 ms |
| 8 rows | within 0.06 ms | within 0.03 ms |
| 32 rows | within 0.01 ms | 29.63–30.20 ms |

So:
- **What clearly beats the spread:**
  - in plain MLX, 0.06–0.07 ms at 1 and 8 rows (1.3% and 0.5%);
  - in vLLM at 8 rows, 0.21 ms (1.5%) from the q, k and v views, and 0.17 ms with everything.
- **Within vLLM's spread:** at 1 and 32 rows the changes look like gains of 0.1–0.6 ms but are inside that spread.
- **Drift, not effect:** in plain MLX at 32 rows, every change came out 0.04–0.10 ms (under 0.4%) slower.
  - The row-index change leaves 32 rows untouched: the sorted path keeps MLX's default, with the same 986 operations. Yet it too came out 0.04 ms slower.
  - Each variant ran at a different time while the machine paged, so drift between variants can exceed the unchanged step's own spread. About 0.1 ms at 32 rows is the floor for any claim.

**Why so little (from the code, and an estimate).**
- **From the code:** MLX's Metal backend dispatches kernels concurrently. It puts a barrier only before a kernel that reads what an earlier kernel wrote (`mlx/backend/metal/device.cpp:371`, `:409-421`, `:606`).
- **Estimate:** an index range and the cast of a stored bias read nothing that the step computes. So they can run alongside the chain, and removing them saves little GPU time; what they cost is mainly encoding.
- The q, k and v copies are on the chain, since the attention kernel waits for them, which is why removing them shows.

The kernel count is therefore a rough guide. What sets the step is the chain of kernels that wait for each other, plus the CPU time to record and encode all of them.

## 6. What this changes in the fix list

Compared with [decode-bottlenecks.md](decode-bottlenecks.md#what-to-do-in-priority-order):

- **Fix 1, compiling the mixture-of-experts block, stays first.** With one request, routing and combine are the largest piece, 1.6 ms of a 4.75 ms step. They are chains of tiny dependent kernels, which is what compiling fuses.
- **Fix 3, cutting vllm-metal's CPU per step, matters more than estimated for one request.**
  - vLLM's one-request step is set by the CPU, and its attention path alone runs about 1 ms of Python per step.
  - Caching each layer's attributes, building the RoPE positions once per step, and `squeeze(0)` are all small Python changes in vllm-metal.
- **The three exact fixes in section 5 are worth making as clean-ups, not as speed-ups.**
  - They save 0.2 ms per step at most.
  - The q, k and v view change is the one with a measured gain, and it is a three-line change in vllm-metal.
  - Storing `expert_bias` in float32 belongs in `bench/convert_qat.py`. The row indices need an MLX change to help from 16 rows up.
- **Fixes 4 and 5 are confirmed where they run.**
  - The routed experts are half the step at 32 rows (14.7–14.9 ms).
  - The GPU's counters put fix 4's target at 465 MB per step of repeat expert reads that reach memory, 30% of the step's reads at 32 rows. That is smaller than the 973 MB of repeat reads, because the caches serve the rest.
  - vLLM's attention kernel costs 4.3 ms at 32 rows with 128-token prompts.
  - The engines' gap reaches 6.5 ms with 1,024-token prompts.
- **A new lead, not yet a fix: MLX's split quantized products.**
  - From 14 rows, the q, k, v and o projections and the shared expert write about 173 MB of float32 partial sums per step, which a second kernel adds up.
  - Measured for the q, k and v projections and the shared expert; worked out from the code for the o projection.
  - A smaller split, or partial sums in bfloat16, would cut that traffic, but would change how many threads run at once and, for bfloat16, the rounding. Neither has been tried.
- **None of this makes the step bandwidth-bound.** The GPU is busy 92–98% of the time at 17–28% of peak bandwidth, with most thread slots empty. Fewer, larger kernels (fixes 1 and 4) stay the main lever.

## Limits

- **Memory.** The machine paged heavily throughout.
  - Docker's VM was quit for the runs, but other programs held most of the 16 GB, among them a 2 GB `llama-server`.
  - Swap in use stayed between 24.7 and 31.1 GB.
  - During the runs the machine swapped out 8,622 pages a second (about 140 MB/s) on average while plain MLX ran, and 18,118 (about 300 MB/s) while vLLM ran.
  - vllm-metal's startup memory check refused the 1 GiB key/value cache twice, because forcing 512 MiB resident pushed other programs' memory out ([console log](../../results/runs/2026-11-01-breakdown/console-vllm-probe-refused.txt)). So vLLM ran with that check off (`VLLM_METAL_KV_COMMIT_PROBE=0`); the per-run swap counts recorded the paging instead.
- **vLLM's one-request runs are noisy.**
  - Its repeats spread by 17–47%, against 0.2–3.3% for plain MLX at every setting and 1.1–2.6% for vLLM at 8 and 32 rows.
  - In the worst repeats, single steps stalled for up to 124 ms.
  - At one request vLLM's step is set by the CPU, so any CPU disturbance shows. Plain MLX had CPU to spare.
  - Hence the ranges and "noise" entries in section 3. `tables.md` also gives every knock-out from the fastest repeat.
- **The sync mode's times are not used.**
  - Run piece by piece, the GPU idles between pieces, and each piece starts cold.
  - The encoding and GPU times sum to 5–8 times the real step: plain MLX's GPU sum is 23.4 ms against a 4.72 ms step with one row. Only the kernel counts are used; the times are in `tables.md` for completeness.
- **Kernel counts are graph operations.** A `Contiguous` copy whose input is already laid out in order shares its memory and launches nothing; that is the case for vLLM's 60 at one row (from the code).
- **Knock-outs measure a piece where it runs.**
  - A faster version of the piece gains less than its full saving.
  - Removing a large piece changes the overlap of the rest, and the replayed outputs change the tokens generated. Savings at 32 rows are therefore not additive (section 3).
- **No per-kernel GPU trace.** The GPU side comes from step periods, knock-outs and whole-GPU counters, not from a timeline of each kernel. The GPU clock was not recorded, so whether the GPU ran at full clock is not known.
- **The GPU counters ([Memory traffic counted by the GPU](#memory-traffic-counted-by-the-gpu)):**
  - **Recording changes the step.** Recording only the counters cost plain MLX 3–9% of its step time at 1 and 32 requests in an interleaved check. Recorded vLLM steps were 1–6% longer than unrecorded ones.
    - Plain MLX at 32 requests with 1,024-token prompts went from 33 to 57–102 ms per step while recording, with the GPU idle for long stretches. The cause was not found, and its times are not used.
    - Xcode's full Metal System Trace template, used first, cost up to 20% and was dropped.
  - **One recording of about one second per setting.**
    - Bytes per step repeat within 0.5% (two recordings of the same step).
    - Step times at 32 requests vary by about ±10% between recordings, so the GB/s use the unrecorded step periods.
  - **The counters cover the whole GPU.**
    - In clean runs, other programs added under 0.2 GB/s (under 1 MB per step).
    - In four runs another program used the GPU just before the window: three plain MLX knock-outs with one request, and one with 32. That work did not carry on into the window, so nothing is subtracted for it.
  - **What "read bandwidth" counts.** Apple does not document whether reads served by the GPU's caches are counted. The calibration shows the units, and the experts' 908 MB against 1,416 MB read per pair suggests cache hits are not counted (estimate).
  - **Knock-outs and caches.** Removing a piece changes what the rest of the step finds in cache, so changes below about 60 MB at 32 requests are not attributed to a piece.
- **Prompts are random tokens**, as in the sweep. The two engines draw them from different generators, so their routing differs slightly: at 32 rows, 40.1 distinct experts per layer in plain MLX against 36.4 in vLLM, in the counted step.

## How this was measured

The same machine and software as the baseline ([method](../benchmark/method.md)): vLLM 0.32.0, vllm-metal 0.32.0.dev20261028121433 (commit e177b67), mlx 0.32.4 and macOS 15.7.4, with `models/MobileMoE-S-QAT-mlx-bfloat16`.

The runs took place on 2026-11-01 between 01:37 EDT and 01:24 EST (Docker's VM quit for 48 minutes in all). The order was plain MLX (14 min), vLLM (20 min), then `repeat_fixes.py`. `lm_head_shapes.py` ran afterwards, with Docker running again.

A first attempt on 2026-10-31 ran out of memory. Plain MLX scored all 32 × 1,024 prompt positions at once, 8.4 GB of logits. The harness now runs prompts the way mlx_lm generate does: in chunks of at most 2,048 tokens, scoring only the last position. A one-off check with the float32 model found the same next-token logits as a single pass, to within 0.0001.

To reproduce, run from the repo root (about 45 minutes):

```bash
R=results/runs/<date>-breakdown
.venv/bin/python bench/micro/breakdown.py plain $R
VLLM_METAL_KV_COMMIT_PROBE=0 .venv/bin/python bench/micro/breakdown.py vllm $R
.venv/bin/python bench/micro/repeat_fixes.py plain $R
VLLM_METAL_KV_COMMIT_PROBE=0 .venv/bin/python bench/micro/repeat_fixes.py vllm $R
.venv/bin/python bench/micro/lm_head_shapes.py > $R/lm_head_shapes.txt
.venv/bin/python bench/micro/breakdown_tables.py $R > $R/tables.md
```

Drop `VLLM_METAL_KV_COMMIT_PROBE=0` on a machine with memory to spare.

| File in [results/runs/2026-11-01-breakdown/](../../results/runs/2026-11-01-breakdown/) | What it holds |
|---|---|
| `runs-plain.jsonl`, `runs-vllm.jsonl` | one record per run: every steady step's period, each span's time per step, the swap-outs before and after; in sync runs, each span's operations and the distinct experts |
| `fixes-plain.jsonl`, `fixes-vllm.jsonl` | `repeat_fixes.py`: each change's operation count and output check, and every timed run |
| `console-*.txt` | the console output of each script, including the refused vLLM start |
| `machine-start.txt`, `machine-between.txt`, `machine-end.txt` | memory, swap and the largest processes before, between and after the runs |
| `lm_head_shapes.txt` | the output layer on both row layouts |
| `tables.md` | every table, generated by `breakdown_tables.py` |

**The GPU counters** ran on 2026-11-02 between 02:10 and 02:16 EST, after the user's other jobs had finished. Docker's VM was running (about 330 MB resident), Chrome was open, and swap was 12.8–13.9 GB in use. No run swapped out pages except plain MLX at 32 × 1,024, during its prompt. They need full Xcode (26.3 here) with its license accepted. To reproduce (about 10 minutes of recording, then about 20 minutes to export the traces):

```bash
R=results/runs/<date>-gpu-counters
.venv/bin/python bench/micro/gpu_counters.py record plain $R
VLLM_METAL_KV_COMMIT_PROBE=0 .venv/bin/python bench/micro/gpu_counters.py record vllm $R
.venv/bin/python bench/micro/gpu_counters.py tables $R > $R/tables.md
```

The `tables` step reads the 2026-11-01 breakdown for the unrecorded step periods and the model's byte counts. `--only MODE:ROWS:PROMPT` records one setting again; the newest recording of a setting is the one used.

| File in [results/runs/2026-11-02-gpu-counters/](../../results/runs/2026-11-02-gpu-counters/) | What it holds |
|---|---|
| `counters.jsonl` | one record per recording: its trace, steps and window length, the mean context, the swap-outs before and after |
| `counters-summary.json` | the same, with every counter's mean while busy and time average over the window and the lead-in, and the busy shares |
| `tables.md` | every table, generated by `gpu_counters.py tables` |

The traces themselves (1.4 GB) are in `results/traces/2026-11-02-gpu-counters/`, which is not committed.
