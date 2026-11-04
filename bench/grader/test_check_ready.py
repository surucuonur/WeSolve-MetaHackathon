"""Check that check_ready.py's audit catches each way of gaming it, using one real check.

Copies a finished ready-check folder once per case, changes one thing in the
copy, and audits it. The unchanged copy must keep its verdict; every change
must end in exit 2 (no verdict) or, for changed tokens, exit 1 (fail). Uses no
GPU.

  .venv/bin/python bench/grader/test_check_ready.py <finished ready-check folder>
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_ready as cr  # noqa: E402


def load(folder):
    return [json.loads(line) for line in (folder / "muse-vllm.jsonl").open()]


def save(folder, records):
    (folder / "muse-vllm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def edit_runs(change):
    def apply(folder):
        save(folder, [change(r) if r.get("kind") == "run" else r for r in load(folder)])
    return apply


def drop(keep):
    def apply(folder):
        save(folder, [r for r in load(folder) if r.get("kind") != "run" or keep(r)])
    return apply


def edit_marker(change):
    def apply(folder):
        save(folder, [change(r) if r.get("kind") == cr.MARKER else r for r in load(folder)])
    return apply


def forge_g3(folder):
    g3 = json.loads((folder / "g3-vllm.json").read_text())
    for v in g3.values():
        v["g3"] = True
        for bar in v.values():
            if isinstance(bar, dict):
                bar["pass"] = True
    (folder / "g3-vllm.json").write_text(json.dumps(g3, indent=1) + "\n")


def extra_variant(folder):
    records = load(folder)
    save(folder, records + [r | {"variant": "myname"} for r in records if r.get("kind") == "run" and r["variant"] == "base"])


def changed_tokens(folder):
    muse = next(r["variant"] for r in load(folder) if r.get("kind") == "check" and r["variant"].startswith("muse-"))
    save(folder, [r | {"exact": False} if r.get("kind") == "check" and r["variant"] == muse else r for r in load(folder)])


def one_prompt_missing(folder):
    records = load(folder)
    first = next(i for i, r in enumerate(records) if r.get("kind") == "run")
    records[first] = records[first] | {"prompt_ms": None}
    save(folder, records)


def main():
    source = Path(sys.argv[1])
    original, code = cr.audit(source)
    slots = list(original["protocol"]["attempts"] or {})
    cases = [
        ("unchanged", lambda f: None, code),
        ("three repeats", drop(lambda r: r["rep"] < 3), 2),
        ("one workload", drop(lambda r: (r["rows"], r["prompt"]) == (32, 128)), 2),
        ("plain engine", edit_runs(lambda r: r | {"engine": "plain"}), 2),
        ("9 tokens", edit_runs(lambda r: r | {"tokens": 9}), 2),
        ("marker says 9 tokens", edit_marker(lambda r: r | {"tokens": 9}), 2),
        ("no marker", lambda f: save(f, [r for r in load(f) if r.get("kind") != cr.MARKER]), 2),
        ("two markers", lambda f: save(f, [r for r in load(f) if r.get("kind") == cr.MARKER] + load(f)), 2),
        ("extra variant", extra_variant, 2),
        ("forged g3-vllm.json", forge_g3, 2),
        ("a prompt time missing", one_prompt_missing, 2),
        ("changed tokens", changed_tokens, 1),
    ]
    if len(slots) > 1:
        cases.append(("combination not timed", drop(lambda r: not r["variant"].endswith("-all")), 2))
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, change, want in cases:
            folder = Path(tmp) / name.replace(" ", "-")
            shutil.copytree(source, folder)
            change(folder)
            verdict, got = cr.audit(folder)
            ok = got == want
            failures += not ok
            why = (verdict["problems"] or [""])[0][:90]
            print(f"{'ok  ' if ok else 'FAIL'} {name}: exit {got}, want {want} {why}")
        folder = Path(tmp) / "forged-ready-verdict"
        shutil.copytree(source, folder)
        saved = json.loads((folder / "ready-verdict.json").read_text())
        (folder / "ready-verdict.json").write_text(json.dumps(saved | {"pass": not saved["pass"]}))
        got = subprocess.run([sys.executable, str(HERE / "check_ready.py"), "--audit", str(folder)], capture_output=True, text=True)
        ok = got.returncode == 2 and "does NOT agree" in got.stdout
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} forged ready-verdict.json: --audit exit {got.returncode}, want 2")
    print(f"{len(cases) + 1 - failures} of {len(cases) + 1} cases as expected")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
