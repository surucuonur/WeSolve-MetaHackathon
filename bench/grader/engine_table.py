"""Tables of an engine comparison run by time_candidates.py, one row per version and setting.

From <run>/muse-<engine>.jsonl it writes, next to it:

  versions.csv     per version and setting (requests x prompt tokens): the
                   median decode step and its range over the repeats, its
                   change against the unchanged engine, the median time until
                   every prompt was processed and its ratio to the unchanged
                   engine's, and output tokens per second over the whole request
  correctness.csv  per version: whether its greedy tokens equal the unchanged
                   engine's at 1 and 32 requests

  .venv/bin/python bench/grader/engine_table.py results/runs/<run> [--engine vllm] [--tokens 256]

LABELS names the versions of the 2026-11-04 comparison; a variant not in it
keeps its own name.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

LABELS = {  # variant -> (label, short label, round, submitted)
    "base": ("Unchanged engine", "Unchanged", "none", "yes"),
    "compiled-3": ("MLX references, compiled", "Compiled references", "none", "yes"),
    "compiled-routing": ("Routing reference, compiled", "Compiled routing", "none", "yes"),
    "round1": ("Round 1: all three kernels", "Round 1", "1", "yes"),
    "round2-submitted": ("Round 2 submission: routing + combine", "R2 submission", "2", "yes"),
    "muse-routing": ("Round 2: routing only", "R2 routing", "2", "yes"),
    "round2-routing+experts-a7": ("Round 2: routing + experts attempt-7", "R2 routing + experts a7", "2", "no"),
    "round2-all-three": ("Round 2: routing + combine + experts attempt-7", "R2 all three", "2", "no"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--tokens", type=int, default=256, help="output tokens per request")
    args = ap.parse_args()
    records = [json.loads(line) for line in (args.run / f"muse-{args.engine}.jsonl").open()]
    runs = [r for r in records if r["kind"] == "run" and r["engine"] == args.engine]
    checks = {(r["variant"], r["rows"]): r for r in records if r["kind"] == "check"}
    seen = list(dict.fromkeys(r["variant"] for r in runs))
    variants = [v for v in LABELS if v in seen] + [v for v in seen if v not in LABELS]
    settings = sorted({(r["rows"], r["prompt"]) for r in runs})

    def runs_of(variant, rows, prompt):
        return [r for r in runs if (r["variant"], r["rows"], r["prompt"]) == (variant, rows, prompt)]

    out = []
    for rows, prompt in settings:
        base = runs_of("base", rows, prompt)
        base_step = statistics.median(r["period_median_ms"] for r in base)
        base_wall = statistics.median(r["wall_ms"] for r in base)
        base_prompt = statistics.median(r["prompt_ms"] for r in base)
        for v in variants:
            rs = runs_of(v, rows, prompt)
            steps = [r["period_median_ms"] for r in rs]
            step, wall = statistics.median(steps), statistics.median(r["wall_ms"] for r in rs)
            prompts = [r["prompt_ms"] for r in rs if r.get("prompt_ms") is not None]
            label, short, rnd, submitted = LABELS.get(v, (v, v, "", ""))
            out.append({
                "version": v,
                "label": label,
                "short": short,
                "round": rnd,
                "submitted": submitted,
                "setting": f"{rows} x {prompt}",
                "requests": rows,
                "prompt_tokens": prompt,
                "step_ms": round(step, 3),
                "step_min_ms": round(min(steps), 3),
                "step_max_ms": round(max(steps), 3),
                "step_change_ms": round(step - base_step, 3),
                "step_change_pct": round((step / base_step - 1) * 100, 2),
                "prompt_ms": round(statistics.median(prompts), 1) if prompts else "",
                "prompt_ratio": round(statistics.median(prompts) / base_prompt, 2) if prompts else "",
                "wall_ms": round(wall, 1),
                "tokens_per_s": round(rows * args.tokens / wall * 1e3, 1),
                "tokens_per_s_change_pct": round((base_wall / wall - 1) * 100, 2),
                "repeats": len(rs),
            })
    with (args.run / "versions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    with (args.run / "correctness.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "label", "same_tokens_1_request", "same_tokens_32_requests"])
        for v in variants:
            same = ["yes" if v == "base" or checks.get((v, n), {}).get("exact") else "no" for n in (1, 32)]
            w.writerow([v, LABELS.get(v, (v,))[0], *same])
    print(f"wrote {args.run / 'versions.csv'} ({len(out)} rows) and correctness.csv")


if __name__ == "__main__":
    main()
