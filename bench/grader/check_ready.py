"""The ready check: the Muse agent's in-round test of whole requests.

An attempt is `ready` only when this check says it improves whole requests
(claude/adr/0013-measure-what-you-claim.md). It runs G3 as Claude runs it after
the round, time_candidates.py vllm with the four G3 workloads, 5 repeats and
256 tokens, with every submitted slot in one call so that a combination is
measured as one. Nothing about that protocol can be changed from the command
line. Then it audits the log instead of trusting it:

  1. the log holds exactly one ready-check marker, with the protocol above;
  2. every timed run is vLLM, at a G3 workload, with 256 tokens and every
     measurement present, and each variant has repeats 0-4 at each workload;
  3. the variants are exactly base, compiled-<slot> and muse-<slot> for each
     slot, plus compiled-all and muse-all with more than one slot;
  4. every variant keeps the unchanged model's greedy tokens at 1 and 32 requests;
  5. g3-vllm.json equals the G3 verdict recomputed from the log.

It writes ready-verdict.json: each muse-* variant's verdict against base and
its compiled counterpart, and each variant's median, range and count per
workload and measurement. Exit codes: 0 every muse-* variant passes; 1 a
variant fails, a verdict that stands; 2 the run stopped or broke the protocol,
so there is no verdict and the check may run again in a new folder.

  .venv/bin/python bench/grader/check_ready.py <new out folder> SLOT=FOLDER [SLOT=FOLDER ...]
  .venv/bin/python bench/grader/check_ready.py --audit <out folder>

--audit re-checks a finished folder without the GPU and says whether its
ready-verdict.json agrees; it writes nothing. The swarm's checker uses it, and
so does Claude after the round.
"""

import argparse
import contextlib
import io
import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "micro"))
import time_candidates as tc  # noqa: E402  (through breakdown, sets the vLLM environment)

MARKER = "ready-check"
SETTINGS = [(1, 128), (8, 128), (32, 128), (32, 1024)]
REPEATS = 5
TOKENS = 256
PROTOCOL = {
    "engine": "vllm",
    "settings": [f"{r}x{p}" for r, p in SETTINGS],
    "repeats": REPEATS,
    "tokens": TOKENS,
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_protocol():
    """Stop before any GPU work if time_candidates.py's G3 is no longer this protocol."""
    g3 = [tuple(int(n) for n in s.split("x")) for s in tc.G3_SETTINGS.split(",")]
    if g3 != SETTINGS or tc.G3_REPEATS != REPEATS:
        sys.exit(
            f"time_candidates.py's G3 is {tc.G3_SETTINGS} with {tc.G3_REPEATS} repeats; "
            "change check_ready.py and ADR 0013 with it"
        )


def run(out, attempts):
    """Run the pinned G3 into *out*; the error that stopped it, or None."""
    check_protocol()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: every ready check gets a new folder")
    out.mkdir(parents=True, exist_ok=True)
    marker = {"kind": MARKER, **PROTOCOL, "attempts": {s: str(f) for s, f in attempts}, "started": now()}
    (out / "muse-vllm.jsonl").write_text(json.dumps(marker) + "\n")
    argv = [
        "time_candidates.py", "vllm", str(out), *[f"{s}={f}" for s, f in attempts],
        "--settings", ",".join(PROTOCOL["settings"]),
        "--repeats", str(REPEATS),
        "--tokens", str(TOKENS),
    ]
    saved, sys.argv = sys.argv, argv
    try:
        tc.main()
    except (Exception, SystemExit) as e:  # recorded as "the run stopped"; never a verdict
        if not (isinstance(e, SystemExit) and e.code in (0, None)):
            return f"{type(e).__name__}: {e}"
    finally:
        sys.argv = saved
    return None


def expected_variants(slots):
    names = ["base"] + [f"{p}-{s}" for s in slots for p in ("compiled", "muse")]
    return names + (["compiled-all", "muse-all"] if len(slots) > 1 else [])


def audit(out):
    """The verdict for the check in *out*, and its exit code."""
    log = out / "muse-vllm.jsonl"
    records = [json.loads(line) for line in log.open()] if log.exists() else []
    markers = [r for r in records if r.get("kind") == MARKER]
    runs = [r for r in records if r.get("kind") == "run"]
    checks = [r for r in records if r.get("kind") == "check"]
    marker = markers[0] if len(markers) == 1 else {}
    problems = [] if marker else [f"want one {MARKER} marker in {log}, found {len(markers)}"]
    problems += [f"marker {k} is {marker.get(k)!r}, want {v!r}" for k, v in PROTOCOL.items() if marker and marker.get(k) != v]
    variants = expected_variants(list(marker.get("attempts") or {}))
    verdict = {"checked_at": now(), "out": str(out), "protocol": {**PROTOCOL, "attempts": marker.get("attempts")}}

    timed = sorted({r.get("variant") for r in runs})
    if timed != sorted(variants):
        problems.append(f"variants timed {timed}, want {sorted(variants)}")
    for r in runs:
        bad = [f"engine {r.get('engine')!r}"] if r.get("engine") != "vllm" else []
        bad += [f"tokens {r.get('tokens')!r}"] if r.get("tokens") != TOKENS else []
        bad += [f"workload {r.get('rows')}x{r.get('prompt')}"] if (r.get("rows"), r.get("prompt")) not in SETTINGS else []
        bad += [f"no {k}" for k in tc.MEASURES.values() if r.get(k) is None]
        if bad:
            problems.append(f"run {r.get('variant')} {r.get('rows')}x{r.get('prompt')} rep {r.get('rep')}: {', '.join(bad)}")
    for v in variants:
        for rows, prompt in SETTINGS:
            reps = sorted(r.get("rep") for r in runs if (r.get("variant"), r.get("rows"), r.get("prompt")) == (v, rows, prompt))
            if reps != list(range(REPEATS)):
                problems.append(f"{v} {rows}x{prompt}: repeats {reps}, want 0 to {REPEATS - 1}")
    exact = {}
    for v in variants[1:]:
        for rows in (1, 32):
            c = [r for r in checks if (r.get("variant"), r.get("rows")) == (v, rows)]
            exact.setdefault(v, {})[str(rows)] = bool(c) and all(r.get("exact") is True for r in c)
            if not c:
                problems.append(f"no token check for {v} at {rows} requests")
            elif v.startswith("compiled-") and not exact[v][str(rows)]:
                problems.append(f"{v} changes the tokens at {rows} requests, so it is no bar")
    verdict["same_tokens"] = exact

    if not problems:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tc.g3(log, variants, SETTINGS, Path(tmp) / "g3.json")
            g3 = json.loads((Path(tmp) / "g3.json").read_text())
        written = out / "g3-vllm.json"
        if not written.exists() or json.loads(written.read_text()) != g3:
            problems.append("g3-vllm.json differs from the G3 verdict recomputed from the log")
    if problems:
        return verdict | {"pass": False, "judged": False, "problems": problems}, 2

    judged = {}
    for v in [n for n in variants if n.startswith("muse-")]:
        same = all(exact[v].values())
        bars = {bar: g3[v][bar] for bar in g3[v] if bar != "g3"}
        judged[v] = {
            "pass": g3[v]["g3"] and same,
            "same_tokens": same,
            "vs": {bar: b["pass"] for bar, b in bars.items()},
            "phases": {bar: b["phases"] for bar, b in bars.items()},
        }
    medians = {
        v: {
            f"{rows}x{prompt}": {
                m: {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}
                for m, k in tc.MEASURES.items()
                for xs in [[r[k] for r in runs if (r["variant"], r["rows"], r["prompt"]) == (v, rows, prompt)]]
            }
            for rows, prompt in SETTINGS
        }
        for v in variants
    }
    passed = all(j["pass"] for j in judged.values())
    return verdict | {"pass": passed, "judged": True, "problems": [], "variants": judged, "medians": medians}, 0 if passed else 1


def report(verdict, code):
    for v, j in (verdict.get("variants") or {}).items():
        vs = ", ".join(f"vs {bar} {'pass' if p else 'fail'}" for bar, p in j["vs"].items())
        print(f"ready {v}: {'pass' if j['pass'] else 'fail'} ({vs}; same tokens {j['same_tokens']})")
    for p in verdict.get("problems", [])[:20]:
        print(f"problem: {p}")
    print(f"ready check: {['PASS', 'FAIL', 'NO VERDICT'][code]} (exit {code})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", action="store_true", help="re-check a finished folder; writes nothing")
    ap.add_argument("out", type=Path)
    ap.add_argument("attempts", nargs="*", type=tc.attempt_arg, metavar="SLOT=FOLDER")
    args = ap.parse_args()
    if args.audit == bool(args.attempts):
        ap.error("give SLOT=FOLDER attempts to run a check, or --audit and a folder alone")

    if args.audit:
        verdict, code = audit(args.out)
        report(verdict, code)
        saved = args.out / "ready-verdict.json"
        old = json.loads(saved.read_text()) if saved.exists() else None
        keys = ("pass", "judged", "variants", "medians", "same_tokens")
        agrees = old is not None and all(old.get(k) == verdict.get(k) for k in keys)
        print(f"ready-verdict.json {'agrees' if agrees else 'does NOT agree'} with this audit")
        sys.exit(code if agrees else 2)

    stopped = run(args.out, args.attempts)
    verdict, code = audit(args.out)
    if stopped:
        verdict |= {"pass": False, "judged": False, "problems": [f"the run stopped: {stopped}", *verdict["problems"]]}
        verdict.pop("variants", None)
        verdict.pop("medians", None)
        code = 2
    (args.out / "ready-verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    report(verdict, code)
    sys.exit(code)


if __name__ == "__main__":
    main()
