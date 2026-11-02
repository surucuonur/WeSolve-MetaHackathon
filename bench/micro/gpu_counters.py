"""Investigation: the GPU's own counters during steady MobileMoE-S decode steps.

Apple's Metal GPU Counters instrument (Xcode's xctrace) samples the whole GPU
about every 20 µs while it has work, and not at all while it is idle: memory
read and write bandwidth (as % of the 200 GB/s peak, checked below), ALU and
cache use, occupancy. Each recording here:

  1. decodes until a steady step (GATE_STEP steady steps in), then pauses so
     the GPU goes idle;
  2. starts xctrace, waits IDLE_S (the idle baseline), then lets about
     WINDOW_S of steady decode steps run;
  3. pauses again and stops xctrace.

The counters in that window, less the idle level before it, belong to the
decode steps; the step times come from the engine's own clock. Counters are
device-wide, so other programs' GPU work is in both the window and the
baseline.

  record   one engine, every setting; plain also runs knock-outs (as in
           breakdown.py) to see what each piece reads. A first recording
           reads a buffer of known size, to check the counters' units.
  tables   export each trace's counters, summarize, and print markdown

Run from the repo root:
  .venv/bin/python bench/micro/gpu_counters.py record plain results/runs/<date>-gpu-counters
  VLLM_METAL_KV_COMMIT_PROBE=0 .venv/bin/python bench/micro/gpu_counters.py record vllm results/runs/<date>-gpu-counters
  .venv/bin/python bench/micro/gpu_counters.py tables results/runs/<date>-gpu-counters

Recording costs plain MLX 3-9% of its step time, except at 32 rows with
1,024-token prompts, where steps took about 3x longer with the GPU idle half
the time (cause not found); vLLM's steps stay within 1-6%.

Traces (about 50 MB a second) go to results/traces/<folder name>/, which is
not committed; per-recording summaries go to counters.jsonl.
"""

import argparse
import json
import math
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import breakdown as bd  # noqa: E402  (also sets vLLM's environment)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

GATE_STEP = 100  # steady steps decoded before recording; the context then
# matches breakdown.py's mean over 256 tokens
WINDOW_S = 1.0  # steady decoding recorded
IDLE_S = 0.3  # idle GPU recorded before and after the window
KNOCKOUTS = ("attn_kernel", "qkv_proj", "routing", "experts", "shared_expert")
KNOCKOUTS += ("lm_head",)
STEP_MS = {(1, 128): 5, (8, 128): 14, (32, 128): 30, (32, 1024): 40}  # for vLLM's
# token budget: enough steps to fill the window
CALIBRATION_BYTES = 512 << 20
XCTRACE = ["xcrun", "xctrace"]


class Recording:
    """Pauses the engine at a steady step, records ~WINDOW_S of steps."""

    def __init__(self, trace):
        self.trace = trace
        self.steady = 0
        self.proc = None
        self.times = []  # engine clock at each recorded step's start
        self.done = False

    def step(self, steady):
        if self.done:
            return
        if self.proc is None:
            self.steady += steady
            if self.steady == GATE_STEP:
                self._start()
            return
        self.times.append(time.perf_counter())
        if self.times[-1] - self.times[0] >= WINDOW_S:
            self._stop()

    def _start(self):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        # The counters alone: the full Metal System Trace template also logs
        # every command buffer, which slowed 32-row decode with 1,024-token
        # prompts from 33 to 92 ms per step and made it page.
        cmd = XCTRACE + ["record", "--instrument", "Metal GPU Counters"]
        cmd += ["--all-processes"]
        cmd += ["--output", str(self.trace)]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        lines = []
        for line in self.proc.stdout:
            lines.append(line)
            if "Ctrl-C to stop" in line:
                break
        else:
            raise RuntimeError("xctrace did not start:\n" + "".join(lines))
        time.sleep(IDLE_S)
        self.times.append(time.perf_counter())

    def _stop(self):
        time.sleep(IDLE_S)  # the last step finishes, then the GPU idles
        self.proc.send_signal(signal.SIGINT)
        out = self.proc.communicate(timeout=600)[0]
        if not self.trace.exists():
            raise RuntimeError("xctrace saved nothing:\n" + out)
        self.done = True

    def summary(self):
        return {
            "trace": self.trace.name,
            "steps": len(self.times) - 1,
            "window_s": self.times[-1] - self.times[0],
        }


def calibrate(trace):
    """Read a buffer of known size over and over: the expected bytes read."""
    x = mx.ones((CALIBRATION_BYTES // 4,), dtype=mx.float32)
    mx.eval(x)
    rec = Recording(trace)
    rec.steady = GATE_STEP - 1
    rec.step(True)
    while not rec.done:
        mx.eval(x.sum())
        rec.step(True)
    return dict(rec.summary(), bytes_per_step=CALIBRATION_BYTES)


def decode_plain(model, rows, prompt, rec):
    mx.clear_cache()
    mx.random.seed(rows * 10000 + prompt)
    cache = make_prompt_cache(model)
    y = bd.prefill(model, mx.random.randint(0, 100000, (rows, prompt)), cache)
    mx.eval(y)
    ctx = []
    while not rec.done:
        bd.REC.begin_step(rows=rows, prefill=0)  # knock-outs replay inside steps
        rec.step(True)
        if rec.proc is not None and not rec.done:
            ctx.append(rows * cache[0].offset)
        ny = bd.greedy(model(y, cache=cache))
        mx.async_eval(ny)
        y.tolist()
        y = ny
    bd.REC.end_step()
    return statistics.mean(ctx)


def install_vllm_gate(state):
    from vllm_metal.v1.model_runner import MetalModelRunner

    original = MetalModelRunner._start_paged_forward

    def start_forward(self, batch, prefill_reqs, decode_reqs, scheduler_output):
        rec = state.get("rec")
        if rec is not None:
            steady = not prefill_reqs and len(decode_reqs) == state["rows"]
            rec.step(steady)
            if rec.proc is not None and not rec.done:
                ctx = sum(len(s.token_ids) for _, s in decode_reqs)
                state["ctx"].append(ctx)
        return original(self, batch, prefill_reqs, decode_reqs, scheduler_output)

    MetalModelRunner._start_paged_forward = start_forward


def record(args):
    out = args.out
    traces = Path("results/traces") / out.name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "counters.jsonl"

    def write(entry):
        with log.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    stamp = time.strftime("%H%M%S")
    write(
        dict(
            calibrate(traces / f"{args.engine}-{stamp}-calibration.trace"),
            kind="calibration",
            engine=args.engine,
        )
    )

    plan = [("bare", r, p) for r, p in bd.GRID]
    if args.quick:
        plan = [("bare", 1, 128)]
    if args.engine == "plain":
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(bd.MODEL))
        mx.eval(model.parameters())
        bd.build_roles(model)
        if not args.quick:
            plan += [(m, r, 128) for r in (1, 32) for m in ("off", *KNOCKOUTS)]
    else:
        from vllm import LLM, SamplingParams

        state = {}
        install_vllm_gate(state)
        llm = LLM(
            model=bd.MODEL,
            trust_remote_code=True,
            max_model_len=2048,
            max_num_seqs=32,
            max_num_batched_tokens=2048,
            kv_cache_memory_bytes=1 << 30,  # 32 rows x ~1,200 tokens need ~750 MB
            enable_prefix_caching=False,
            generation_config="vllm",
        )
        llm.generate(
            [{"prompt_token_ids": list(range(1000, 1016))}],
            SamplingParams(max_tokens=4, temperature=0, ignore_eos=True),
            use_tqdm=False,
        )

    if args.only:
        only = {(m, int(r), int(p)) for m, r, p in (o.split(":") for o in args.only)}
        plan = [x for x in plan if x in only]
    for mode, rows, prompt in plan:
        if mode != "bare" and not bd.PIECE_PATCHES:
            bd.install_model_hooks("plain")
        bd.REC.reset(rows)
        bd.REC.mode = "off"
        bd.REC.knockout = bd.KNOCKOUTS[mode] if mode in bd.KNOCKOUTS else set()
        name = f"{args.engine}-{stamp}-{mode}-r{rows}-p{prompt}.trace"
        rec = Recording(traces / name)
        before = bd.machine()
        if args.engine == "plain":
            ctx = decode_plain(model, rows, prompt, rec)
        else:
            state.update(rec=rec, rows=rows, ctx=[])
            rng = __import__("random").Random(rows * 10000 + prompt)
            prompts = [
                {
                    "prompt_token_ids": [
                        rng.randrange(1000, 100000) for _ in range(prompt)
                    ]
                }
                for _ in range(rows)
            ]
            extra = math.ceil(1.6 * WINDOW_S * 1000 / STEP_MS[(rows, prompt)])
            params = SamplingParams(
                max_tokens=GATE_STEP + extra + 10, temperature=0, ignore_eos=True
            )
            llm.generate(prompts, params, use_tqdm=False)
            if not rec.done:
                raise RuntimeError("the run ended before the window closed")
            ctx = statistics.mean(state["ctx"])
            state["rec"] = None
        entry = dict(
            rec.summary(),
            kind="run",
            engine=args.engine,
            mode=mode,
            rows=rows,
            prompt=prompt,
            ctx_mean=ctx,
            machine_before=before,
            machine_after=bd.machine(),
        )
        write(entry)
        print(
            f"{args.engine} {mode:>13} rows {rows:2d} prompt {prompt:4d}: "
            f"{entry['window_s'] / entry['steps'] * 1e3:6.2f} ms/step "
            f"({entry['steps']} steps)",
            flush=True,
        )
        if mode != "bare" and mode == plan[-1][0]:
            bd.unpatch(bd.PIECE_PATCHES)


# Summaries ----------------------------------------------------------------


def export(trace, schema, dest):
    xp = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    cmd = XCTRACE + ["export", "--input", str(trace), "--xpath", xp]
    subprocess.run(cmd + ["--output", str(dest)], check=True, capture_output=True)


def xml_rows(path):
    """Rows of an xctrace export, as lists of cell texts (id/ref resolved)."""
    ids = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "row":
            continue
        vals = []
        for cell in el:
            if "ref" in cell.attrib:
                vals.append(ids[cell.attrib["ref"]])
            else:
                v = cell.text if cell.text is not None else cell.get("fmt")
                if "id" in cell.attrib:
                    ids[cell.attrib["id"]] = v
                vals.append(v)
        el.clear()
        yield vals


def counter_series(trace):
    """{counter name: (times in s, values)}, cached next to the trace."""
    cache = trace.with_suffix(".npz")
    if not cache.exists():
        with tempfile.TemporaryDirectory() as tmp:
            info, values = Path(tmp) / "info.xml", Path(tmp) / "values.xml"
            export(trace, "gpu-counter-info", info)
            names = {int(r[1]): r[2] for r in xml_rows(info)}
            export(trace, "gpu-counter-value", values)
            data = defaultdict(lambda: ([], []))
            for r in xml_rows(values):
                ts, vs = data[int(r[1])]
                ts.append(int(r[0]))
                vs.append(float(r[2]))
        arrays = {}
        for cid, (ts, vs) in data.items():
            key = names[cid].replace("/", "_")
            arrays[f"t:{key}"] = np.array(ts) / 1e9
            arrays[f"v:{key}"] = np.array(vs)
        np.savez_compressed(cache, **arrays)
    z = np.load(cache)
    return {k[2:]: (z[k], z["v:" + k[2:]]) for k in z.files if k.startswith("t:")}


READ, WRITE = "GPU Read Bandwidth", "GPU Write Bandwidth"
GAP_S = 100e-6  # the GPU sends a sample every ~20 µs while busy, none while idle


def busy(ts, start, end):
    """The fraction of [start, end) the GPU was sending samples."""
    inside = (ts >= start) & (ts < end)
    dt = np.minimum(np.diff(ts, prepend=ts[0]), GAP_S)
    return float(dt[inside].sum() / (end - start))


def window_means(series, entry):
    """Each counter over the recorded steps and over the idle lead-in.

    Per counter: (mean while the GPU was busy in the window, the window's time
    average, the lead-in's time average). A time average is the mean while
    busy times the busy fraction, since an idle GPU sends no samples. Samples
    come in pairs ~9 and ~30 µs apart with different values, so each sample
    counts once rather than by the gap before it.

    The window starts where read bandwidth first rises above the idle level
    and lasts as long as the engine's clock says the steps took.
    """
    t, v = series[READ]
    smooth = np.convolve(v, np.ones(50) / 50, mode="same")  # about 1 ms
    idle_level = np.percentile(smooth, 5)
    rise = np.argmax(smooth > idle_level + max(5.0, 0.2 * (smooth.max() - idle_level)))
    start = t[rise] - 0.0005
    end = start + entry["window_s"]
    lead_in = (t[0], start - 0.02)
    out = {"busy": (busy(t, start, end), busy(t, *lead_in))}
    for name, (ts, vs) in series.items():
        inside = (ts >= start) & (ts < end)
        idle = (ts >= lead_in[0]) & (ts < lead_in[1])
        mean = float(vs[inside].mean())
        idle_mean = float(vs[idle].mean()) if idle.any() else 0.0
        out[name] = (
            mean,
            mean * busy(ts, start, end),
            idle_mean * busy(ts, *lead_in),
        )
    return out


def summarize(folder):
    traces = Path("results/traces") / folder.name
    # A setting recorded again (after a run that paged) replaces the earlier one.
    latest = {}
    for line in (folder / "counters.jsonl").open():
        e = json.loads(line)
        key = (e["kind"], e["engine"], e.get("mode"), e.get("rows"), e.get("prompt"))
        if e["kind"] == "calibration":
            key += (e["trace"],)
        latest[key] = e
    out = []
    for e in latest.values():
        trace = traces / e["trace"]
        if not trace.exists():
            continue
        means = window_means(counter_series(trace), e)
        out.append(dict(e, counters=means))
        print(f"summarized {e['trace']}", file=sys.stderr, flush=True)
    (folder / "counters-summary.json").write_text(json.dumps(out, indent=1))
    return out


def gbs(entry, name, scale):
    """The window's time-averaged rate, in GB/s.

    The idle lead-in is not subtracted: when idle it is under 0.2 GB/s (under
    1 MB a step), and when another program used the GPU there, that work did
    not carry on into the window (the pieces' bytes then match the model only
    without subtracting it). Such runs are flagged in the tables instead.
    """
    return entry["counters"][name][1] * scale


def tables(folder):
    import breakdown_tables as bt

    runs, headers = bt.load(Path("results/runs/2026-11-01-breakdown"))
    w = headers["plain"]["weight_bytes"]
    entries = summarize(folder)
    cal = [e for e in entries if e["kind"] == "calibration"]
    # The counter's unit: what one window's mean is per GB/s actually read.
    ratios = [
        e["bytes_per_step"] * e["steps"] / e["window_s"] / 1e9 / gbs(e, READ, 1.0)
        for e in cal
    ]
    scale = statistics.median(ratios)
    lines = [
        "# GPU counters during steady decode steps",
        f"Generated by `bench/micro/gpu_counters.py tables {folder}`.",
        "",
        "Calibration: reading a 512 MB buffer over and over, the read counter's "
        f"time average times {scale:.3f} gives the GB/s actually read "
        f"(each recording: {', '.join(f'{r:.3f}' for r in ratios)}).",
    ]
    pct = [
        "ALU Utilization",
        "F16 Utilization",
        "F32 Utilization",
        "Buffer Read Limiter",
        "Buffer Load Utilization",
        "GPU Last Level Cache Utilization",
        "GPU Last Level Cache Limiter",
        "Compute Occupancy",
        "MMU Utilization",
        "MMU TLB Miss Rate",
    ]
    runs_ = [e for e in entries if e["kind"] == "run"]
    rows = []
    for e in runs_:
        if e["mode"] != "bare":
            continue
        per_step = e["window_s"] / e["steps"]
        read = gbs(e, READ, scale)
        write = gbs(e, WRITE, scale)
        sync = runs.get((e["engine"], "sync", e["rows"], e["prompt"]))
        need, _ = bt.demands(w, e["rows"], e["ctx_mean"], sync[0]["distinct_experts"])
        unrecorded = bt.period(runs, (e["engine"], "bare", e["rows"], e["prompt"]))
        rows.append(
            [
                e["engine"],
                e["rows"],
                e["prompt"],
                f"{per_step * 1e3:.2f}",
                f"{unrecorded:.2f}",
                f"{e['counters']['busy'][0] * 100:.0f}",
                f"{read:.0f} ({read / 2:.0f}%)",
                f"{read * per_step * 1e3:.0f}",
                f"{read * per_step / unrecorded * 1e3:.0f}",
                f"{write * per_step * 1e3:.0f}",
                f"{need['step'][0] / 1e6:.0f}",
                *(f"{e['counters'][c][0]:.0f}" for c in pct),
            ]
        )
    lines += [
        "",
        "## Whole step (bare)",
        "Bandwidth is the window's time average; "
        "the percentages are means while the GPU was busy.",
        bt.table(
            [
                "Engine",
                "Rows",
                "Prompt",
                "ms/step recorded",
                "ms/step unrecorded",
                "GPU busy %",
                "Read GB/s (% of 200)",
                "MB read per step",
                "GB/s at the unrecorded step time",
                "MB written per step",
                "MB the model must read",
                *[f"{c} %" for c in pct],
            ],
            rows,
        ),
    ]
    pieces = []
    for r in (1, 32):
        ref = next((e for e in runs_ if e["mode"] == "off" and e["rows"] == r), None)
        if ref is None:
            continue
        ref_step = ref["window_s"] / ref["steps"]
        ref_mb = gbs(ref, READ, scale) * ref_step * 1e3
        ref_wr = gbs(ref, WRITE, scale) * ref_step * 1e3
        sync = runs[("plain", "sync", r, 128)][0]
        need, pairs = bt.demands(w, r, ref["ctx_mean"], sync["distinct_experts"])
        reread = {"experts": pairs, "attn_kernel": 3 * need["attn_kernel"][0]}
        for k in KNOCKOUTS:
            e = next((x for x in runs_ if x["mode"] == k and x["rows"] == r), None)
            if e is None:
                continue
            step = e["window_s"] / e["steps"]
            mb = gbs(e, READ, scale) * step * 1e3
            wr = gbs(e, WRITE, scale) * step * 1e3
            saved = bt.gain(runs, "plain", k, r)
            pieces.append(
                [
                    r,
                    k,
                    f"{(ref_step - step) * 1e3:.2f}",
                    bt.fmt(saved),
                    f"{ref_mb - mb:.0f}",
                    f"{ref_wr - wr:.0f}",
                    f"{need[k][0] / 1e6:.0f}",
                    f"{reread[k] / 1e6:.0f}" if k in reread else "",
                    f"{(ref_mb - mb) / saved:.0f}" if saved and saved > 0.05 else "",
                ]
            )
    lines += [
        "",
        "## Each piece (plain MLX): what the step no longer reads without it",
        "Bytes: the recorded step with every wrapper idle (`off`) less the "
        "recorded step without the piece. One 1-second recording per setting "
        "is too short for step times at 32 rows (about ±10%), so the time "
        "saved and the GB/s use the unrecorded savings, medians of 3 repeats "
        "in results/runs/2026-11-01-breakdown.",
        bt.table(
            [
                "Rows",
                "Knocked out",
                "Step ms saved (recorded)",
                "Step ms saved (unrecorded)",
                "MB read per step saved (counter)",
                "MB written per step saved (counter)",
                "MB it must read (model)",
                "MB if re-read per (token, expert) pair or per query head (model)",
                "GB/s while it runs",
            ],
            pieces,
        ),
    ]
    disturbed = [
        f"{e['engine']} {e.get('mode', 'calibration')} r{e.get('rows', '')}: "
        f"busy {e['counters']['busy'][1] * 100:.0f}%, "
        f"{e['counters'][READ][2] * scale:.1f} GB/s"
        for e in entries
        if e["counters"]["busy"][1] > 0.1
    ]
    lines += [
        "",
        "Lead-ins where another program used the GPU (busy over 10%; the "
        "window's bandwidth is not corrected for it): "
        + ("; ".join(disturbed) or "none"),
    ]
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("engine", choices=("plain", "vllm"))
    r.add_argument("out", type=Path)
    r.add_argument("--quick", action="store_true", help="calibration and one setting")
    r.add_argument(
        "--only", action="append", help="record only MODE:ROWS:PROMPT (repeatable)"
    )
    t = sub.add_parser("tables")
    t.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "record":
        record(args)
    else:
        tables(args.out)


if __name__ == "__main__":
    main()
