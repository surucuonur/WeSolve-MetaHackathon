"""Run Muse Spark's two stages under claude/muse-spark-brief.md.

  capture    record every layer's slot inputs in a real decode step (plain MLX)
  strategy   stage 1: Muse Spark reads the findings and chooses its targets
  kernels    stage 2: Muse Spark writes kernels for a slot, up to 5 attempts,
             each checked by check_candidate.py in its own process

  .venv/bin/python bench/muse/run.py capture
  .venv/bin/python bench/muse/run.py strategy
  .venv/bin/python bench/muse/run.py kernels routing+combine

Muse Spark's answers are saved verbatim (client.py); the harness only
extracts its code blocks and sends back what check_candidate.py measured.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402

ROOT = client.ROOT
MUSE = ROOT / "muse-spark"
ATTACH = [  # the only files sent; nothing else is read
    "claude/findings/bottleneck-explained.md",
    "claude/findings/step-breakdown.md",
    "results/runs/2026-11-01-breakdown/tables.md",
    "results/runs/2026-11-02-gpu-counters/tables.md",
    "third_party/mlx-lm/mlx_lm/models/mobilemoe.py",
    "third_party/mlx-lm/mlx_lm/models/switch_layers.py",
]
ATTEMPTS = 5
TIME_LIMIT = 180  # seconds per check_candidate run
SLOTS = ("routing", "combine", "routing+combine", "experts")


def first_message():
    import mlx.core as mx

    parts = [client.BRIEF.read_text(), "# Attached files"]
    for rel in ATTACH:
        if ".env" in Path(rel).name:
            raise SystemExit(f"refusing to send {rel}")
        lang = "python" if rel.endswith(".py") else "markdown"
        parts.append(f"## {rel}\n\n```{lang}\n{(ROOT / rel).read_text()}\n```")
    doc = mx.fast.metal_kernel.__doc__
    parts.append(
        f"## mx.fast.metal_kernel documentation (MLX 0.32.4)\n\n```text\n{doc}\n```"
    )
    parts.append(
        "# Stage 1\n\nChoose your targets now, in the Stage 1 answer format of the brief."
    )
    return {"role": "user", "content": "\n\n".join(parts)}


def strategy():
    if (MUSE / "strategy.md").exists():
        raise SystemExit("muse-spark/strategy.md exists; stage 1 runs once")
    text, call_dir = client.call("strategy", [first_message()])
    client.write_verbatim(MUSE / "strategy.md", text, call_dir, ">")
    targets = json.loads(client.blocks(text, "json")[-1])["targets"]
    print("targets:", [t["slot"] for t in targets])


def feedback(result):
    if not result.get("g1"):
        return f"G1 failed: the attempt did not compile or run.\n\n```text\n{result.get('error')}\n```"
    lines = ["G1 passed: it compiles and runs at 1, 8 and 32 rows.", "", "Correctness:"]
    for key, s in result["correctness"].items():
        lines.append(f"- {key}: {json.dumps(s)}")
    if not result["g2"]:
        return "\n".join(lines + ["", "G2 failed. Fix the errors above."])
    lines += [
        "",
        "G2 passed on all 20 layers. Time per layer call, µs (GPU waited each call):",
    ]
    for key, t in result["timing_us_per_layer"].items():
        lines.append(f"- {key}: {json.dumps(t)}")
    verdict = "beats" if result["beats_compiled"] else "does not beat"
    return "\n".join(
        lines + ["", f"It {verdict} the compiled reference at every row count."]
    )


def parse(text):
    """Muse Spark's Metal blocks and launch spec, or the reason they can't be read."""
    sources = client.blocks(text, "metal")
    specs = client.blocks(text, "json")
    if not sources or not specs:
        return (
            sources,
            None,
            "The answer needs ```metal blocks and one ```json launch spec.",
        )
    try:
        json.loads(specs[-1])
    except json.JSONDecodeError as e:
        return sources, None, f"The launch spec is not valid JSON: {e}"
    return sources, specs[-1], None


def kernels(slot):
    strat = client.read_verbatim(MUSE / "strategy.md")
    request = (
        f"# Stage 2: slot `{slot}`\n\nWrite the kernels for `{slot}` now, in the Stage 2 "
        f"answer format of the brief. This is attempt 1 of {ATTEMPTS}."
    )
    messages = [first_message(), {"role": "assistant", "content": strat}]
    messages.append({"role": "user", "content": request})
    best = None
    for n in range(1, ATTEMPTS + 1):
        name = slot.replace("+", "-")
        text, call_dir = client.call(f"kernels-{name}-a{n}", messages)
        attempt = MUSE / "kernels" / name / f"attempt-{n}"
        attempt.mkdir(parents=True, exist_ok=True)
        sources, spec, problem = parse(text)
        for i, src in enumerate(sources):
            client.write_verbatim(attempt / f"block-{i}.metal", src, call_dir, "//")
        if spec is not None:
            (attempt / "spec.json").write_text(spec)
        if problem:
            result = {"slot": slot, "g1": False, "error": problem}
        else:
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "check_candidate.py"),
                        str(attempt),
                        slot,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=TIME_LIMIT,
                )
                result = json.loads((attempt / "result.json").read_text())
                if proc.returncode:
                    result.setdefault("error", proc.stderr[-4000:])
            except subprocess.TimeoutExpired:
                result = {
                    "slot": slot,
                    "g1": False,
                    "error": f"timed out after {TIME_LIMIT} s",
                }
            except FileNotFoundError:
                result = {"slot": slot, "g1": False, "error": proc.stderr[-4000:]}
        result["call"] = call_dir.name
        (attempt / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        print(
            f"attempt {n}: g1 {result.get('g1')} g2 {result.get('g2')} beats {result.get('beats_compiled')}"
        )
        if result.get("g2"):
            us = sum(t["candidate"] for t in result["timing_us_per_layer"].values())
            if best is None or us < best[1]:
                best = (n, us)
            if result["beats_compiled"]:
                break
        if n < ATTEMPTS:
            note = f"\n\nThis is attempt {n + 1} of {ATTEMPTS}."
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": feedback(result) + note},
            ]
    summary = {"slot": slot, "attempts": n, "best_attempt": best and best[0]}
    (MUSE / "kernels" / slot.replace("+", "-") / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(summary)


def capture():
    sys.path.insert(0, str(ROOT / "bench" / "micro"))
    import breakdown as bd
    import mlx.core as mx
    import slots
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(ROOT / bd.MODEL))
    mx.eval(model.parameters())
    slots.check_identity(model)
    print("split MoE with reference slots: bit-identical to the port's")
    slots.save_inputs(model)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture")
    sub.add_parser("strategy")
    k = sub.add_parser("kernels")
    k.add_argument("slot", choices=SLOTS)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture()
    elif args.cmd == "strategy":
        strategy()
    else:
        kernels(args.slot)


if __name__ == "__main__":
    main()
