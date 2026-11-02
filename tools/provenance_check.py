"""Audit who made every commit, and that Muse Spark's and Claude's work never mix.

Checks, over the history of the given branches (default: HEAD, plus
muse-agent when it exists):

- every commit is authored and committed by surucuonur;
- no commit names both Muse Spark and Claude as co-authors;
- from the tag agent-era on: Muse Spark commits change only muse-spark/, and
  every other commit that isn't a merge of muse-agent leaves muse-spark/ alone;
- every transcript in muse-spark/transcripts/ was written by Meta models only
  (no claude-* model in any assistant turn);
- no tracked file contains an API key value from .env (when .env is present).

Prints a table and exits non-zero on any violation. The git hooks enforce the
same rules at commit time; this catches anything committed around them.

  .venv/bin/python tools/provenance_check.py [branch ...]
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as pv  # noqa: E402

SEP, END = "\x1f", "\x1e"


def commits(refs):
    fmt = SEP.join(["%H", "%an", "%ae", "%cn", "%ce", "%P", "%s", "%B"]) + END
    out = pv.git("log", f"--format={fmt}", *refs)
    for rec in filter(str.strip, out.split(END)):
        h, an, ae, cn, ce, parents, subject, body = rec.strip("\n").split(SEP)
        yield {
            "hash": h,
            "author": (an, ae),
            "committer": (cn, ce),
            "parents": parents.split(),
            "subject": subject,
            "kind": pv.kind(body),
        }


def changed(commit):
    if len(commit["parents"]) > 1:
        return []
    out = pv.git(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "--root",
        "-z",
        commit["hash"],
    )
    return [p for p in out.split("\0") if p]


def is_muse_merge(commit, agent_tips):
    return len(commit["parents"]) > 1 and any(
        p in agent_tips for p in commit["parents"][1:]
    )


def check_transcripts(problems):
    models = Counter()
    for path in sorted(Path("muse-spark/transcripts").glob("*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            entry = json.loads(line)
            if entry.get("type") != "assistant":
                continue
            model = (entry.get("message") or {}).get("model") or ""
            models[model] += 1
            if model.startswith("claude") or not model:
                problems.append(f"{path}:{n}: assistant turn by model {model!r}")
    return models


def check_secrets(problems):
    values = pv.secrets()
    if not values:
        return "skipped (no .env here)"
    for path in pv.git("ls-files", "-z").split("\0"):
        if path and Path(path).is_file() and not Path(path).is_symlink():
            if pv.contains_secret(Path(path).read_bytes(), values):
                problems.append(f"{path}: contains an API key value")
    return "checked"


def main():
    refs = sys.argv[1:] or ["HEAD"]
    branches = pv.git("branch", "--list", pv.MUSE_BRANCH).strip()
    if not sys.argv[1:] and branches:
        refs.append(pv.MUSE_BRANCH)
    try:
        era = set(
            pv.git("rev-list", f"{pv.AGENT_ERA_TAG}..{refs[0]}", *refs[1:]).split()
        )
    except subprocess.CalledProcessError:
        era = set()
    agent_tips = set(pv.git("rev-list", pv.MUSE_BRANCH).split()) if branches else set()
    problems, rows = [], Counter()
    for c in commits(refs):
        tag = c["hash"][:7]
        if c["author"] != pv.AUTHOR or c["committer"] != pv.AUTHOR:
            problems.append(
                f"{tag}: author {c['author']} / committer {c['committer']}, not surucuonur"
            )
        if c["kind"] == "both":
            problems.append(f"{tag}: names both Muse Spark and Claude")
        era_label = "agent era" if c["hash"] in era else "before"
        rows[
            (era_label, c["kind"], "merge" if len(c["parents"]) > 1 else "commit")
        ] += 1
        if c["hash"] not in era:
            continue
        paths = changed(c)
        in_muse = [p for p in paths if p.startswith(pv.MUSE_DIR)]
        if c["kind"] == "muse" and len(in_muse) != len(paths):
            problems.append(
                f"{tag}: a Muse Spark commit changes files outside {pv.MUSE_DIR}"
            )
        if c["kind"] != "muse" and in_muse and not is_muse_merge(c, agent_tips):
            problems.append(
                f"{tag}: a non-Muse Spark commit changes {pv.MUSE_DIR}: {in_muse[:5]}"
            )
    models = check_transcripts(problems)
    secrets = check_secrets(problems)

    print(f"refs: {' '.join(refs)}; agent era starts at tag {pv.AGENT_ERA_TAG}")
    print(f"{'era':<10} {'co-author':<8} {'type':<7} count")
    for (era_label, kind, typ), n in sorted(rows.items()):
        print(f"{era_label:<10} {kind:<8} {typ:<7} {n}")
    print(f"transcript turns by model: {dict(models) or 'none yet'}")
    print(f"secret scan: {secrets}")
    print("\n".join(problems) if problems else "OK: no mixed authorship found")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
