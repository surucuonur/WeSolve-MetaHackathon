"""Check that every Muse Spark file is its API response, unmodified.

For each call in muse-spark/calls/: the answer's SHA-256 matches meta.json.
strategy.md and each attempt's Metal blocks and launch spec are re-extracted
from the call's response.json and compared byte for byte. Each response id
must appear in a commit's Muse-Spark-Response trailer. Prints the total cost.

  .venv/bin/python bench/muse/verify.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402

MUSE = client.ROOT / "muse-spark"


def main():
    problems, total = [], 0.0
    calls = sorted(MUSE.glob("calls/[0-9]*"))
    answers = {}
    for c in calls:
        meta = json.loads((c / "meta.json").read_text())
        text = client.answer_text(c / "response.json")
        answers[c.name] = text
        total += meta.get("cost_usd") or 0
        if client.sha256(text) != meta["answer_sha256"]:
            problems.append(f"{c.name}: answer hash differs from meta.json")
    strategy = MUSE / "strategy.md"
    if strategy.exists():
        first = next((t for n, t in answers.items() if n.endswith("-strategy")), None)
        if client.read_verbatim(strategy) != first:
            problems.append("strategy.md differs from its call's answer")
    for attempt in sorted(MUSE.glob("kernels/*/attempt-*")):
        call = json.loads((attempt / "result.json").read_text())["call"]
        text = answers[call]
        for i, src in enumerate(client.blocks(text, "metal")):
            if client.read_verbatim(attempt / f"block-{i}.metal") != src:
                problems.append(f"{attempt}/block-{i}.metal differs from {call}")
        specs = client.blocks(text, "json")
        if specs and (attempt / "spec.json").read_text() != specs[-1]:
            problems.append(f"{attempt}/spec.json differs from {call}")
    log = subprocess.run(
        ["git", "log", "--format=%(trailers:key=Muse-Spark-Response,valueonly)"],
        capture_output=True,
        text=True,
        cwd=client.ROOT,
    ).stdout.split()
    for c in calls:
        rid = json.loads((c / "meta.json").read_text())["response_id"]
        if rid not in log:
            problems.append(f"{c.name}: response {rid} is in no commit trailer (yet)")
    print(f"{len(calls)} calls, ${total:.4f} in total")
    print(
        "\n".join(problems)
        if problems
        else "all Muse Spark files match their responses"
    )
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
