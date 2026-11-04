# Inefficiency: the experts attempt is slower than the reference when the layers are chained, at 1 row

- **Owner:** explorer-g3
- **Status:** superseded by ineff-baseline-experts-2
- **Version:** 2
- **Updated:** 2026-11-04T03:18:59Z
- **Next:** none — the same claim is carried kernel-side by `ineff-baseline-experts-2`; the G3-side arithmetic below is cited there by note

## Slot

`experts`. Kernel side owned by territory `baseline-experts` (starting point `muse-spark/kernels/experts/attempt-4`); this entry carries the measurement side from territory `g3`, so the lead can settle it as one claim with that territory's entry when it lands.

## What happens

Attempt-4 is one fused kernel per layer with grid `rows*1024` threads in threadgroups of 256 — `rows*4` threadgroups, one per (row, expert-slot) pair (`muse-spark/kernels/experts/attempt-4/spec.json`). Timed the old grader's way, with the 20 layers' calls overlapping, it beats the reference (61.0 vs 86.0 µs/layer at 1 row). Timed chained — each call's `x` is the previous call's output, as in a decode step — it is slower than the reference at 1 row and faster at 8 and 32 rows. In the model, where layers are chained by construction, it slows the 1-request decode step by ~1.2 ms. Claude's estimate from the code: 4 threadgroups at 1 row cannot fill the 16-core GPU, so one call's latency is long even when many calls together go fast ([muse-round-1.md:123](../../../claude/findings/muse-round-1.md)). That mechanism is an estimate; the timings below are measured.

## Cost

Chained vs independent piece timing, experts reference / attempt-4, µs per layer (`bench/micro/chain_timing.py`, `results/runs/2026-11-04-muse-round-1/console-chain.txt`; [territory-g3.md](territory-g3.md) piece 3):

| Timing | 1 row | 8 rows | 32 rows |
|---|---|---|---|
| independent: reference / attempt | 86.0 / 61.0 | 219.1 / 165.1 | 767.3 / 535.0 |
| chained: reference / attempt | 80.3 / **132.9** | 239.6 / **175.9** | 797.6 / **587.6** |
| chained gap × 20 layers | **+1.052 ms/step** | −1.274 ms/step | −4.200 ms/step |

End to end, attempt-4 alone against the unchanged model (G3, median of 3 shuffled repeats): **+1.162 ms (+21.6%) at 1 request**, −0.817 at 8, −4.154 at 32 in vLLM (`console-vllm.txt`); +1.228 / −0.832 / −3.661 ms in plain MLX (`console-plain.txt`). The chained gap at 1 row (+1.052 ms/step) reproduces the end-to-end slowdown to within 0.11 ms (vLLM) / 0.18 ms (plain). Against the brief's G3 rule the attempt also loses to its compiled reference at 1 request (compiled-experts +0.066 ms, vLLM). Bytes per pair are unchanged by row count (553 KB per pair, `muse-spark/kernels/experts/attempt-4/notes.md`): the 1-row cost is not bytes.

## Kind

Overhead-bound (`claude/GLOSSARY.md`): at 1 row the chained call leaves most of the GPU idle — few threadgroups, latency exposed — while reading the same weight bytes as the faster overlapped timing. At 8 and 32 rows the same kernel is bandwidth/efficiency-limited in the normal way and wins.

## Spans

`g3` (this measurement), `baseline-experts` (the kernel), `grader` (the round-1 timing that hid it; the new chained timing in `check_candidate.py` is the fixed instrument).

## Predicted saving

Removing the chained 1-row penalty — bringing the chained call to the reference's 80.3 µs/layer — saves (132.9 − 80.3) µs × 20 layers = **1.05 ms per decode step at 1 request**, with nothing claimed at 8 or 32 rows, where the attempt already saves 1.27 / 4.20 ms/step chained and the fix must not give that back. Measured against G3 instead of the piece timing, the same fix is worth up to 1.16 ms (vLLM) / 1.23 ms (plain) at 1 request: the amount by which the attempt currently slows the step it was meant to speed up.

## Verdict

