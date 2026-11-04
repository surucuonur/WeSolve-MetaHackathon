"""Audit who made every commit, and that the Muse agent's and Claude's work never mix.

Checks, over the history of the given branches (default: HEAD, plus
muse-agent when it exists):

- every commit is authored and committed by surucuonur;
- no commit names both a Muse agent (Muse Spark, Muse Wonder Full) and Claude as
  co-authors;
- from the tag agent-era on: Muse agent commits change only muse-spark/, and
  every other commit that isn't a merge of muse-agent leaves muse-spark/ alone;
- no commit after a finished round's tag (muse-round-<N>) changes or deletes a
  file that round left in muse-spark/;
- every model call in muse-spark/transcripts/, the lead's, every helper's and
  every reminder agent's, was made by a Meta model (muse-*);
- no tracked file contains an API key value from the main checkout's .env files
  (when present).

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
            "muse": [n for n in pv.coauthors(body) if pv.is_muse(n)],
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
        calls = 0
        for n, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            if row.get("event") != "model_completed":
                continue
            calls += 1
            model = row.get("model") or ""
            models[model] += 1
            if not pv.is_meta_model(model):
                problems.append(f"{path}:{n}: model call by {model!r}, not a Meta model")
        if not calls:
            problems.append(f"{path}: no model calls recorded")
    return models


def check_rounds(refs, problems):
    """No commit after a finished round's tag changes or deletes that round's files."""
    finished = pv.finished_rounds()
    for n in finished:
        tag = f"{pv.ROUND_TAG}{n}"
        frozen = set(
            filter(None, pv.git("ls-tree", "-r", "-z", "--name-only", tag, "--", pv.MUSE_DIR).split("\0"))
        )
        for h in pv.git("rev-list", "--no-merges", f"^{tag}", *refs).split():
            out = pv.git(
                "diff-tree", "--no-commit-id", "--name-status", "--no-renames", "-r", "-z", h
            ).split("\0")
            for status, path in zip(out[::2], out[1::2]):
                if status in ("M", "D", "T") and path in frozen:
                    problems.append(f"{h[:7]}: changes {path}, a file of the finished round {tag}")
    return [f"{pv.ROUND_TAG}{n}" for n in finished]


def check_secrets(problems):
    values = pv.secrets()
    if not values:
        return "skipped (no API keys found)"
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
    problems, rows, muse_authors = [], Counter(), Counter()
    for c in commits(refs):
        muse_authors.update(n.split(" <")[0] for n in c["muse"])
        tag = c["hash"][:7]
        if c["author"] != pv.AUTHOR or c["committer"] != pv.AUTHOR:
            problems.append(
                f"{tag}: author {c['author']} / committer {c['committer']}, not surucuonur"
            )
        if c["kind"] == "both":
            problems.append(f"{tag}: names both a Muse agent and Claude")
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
                f"{tag}: a Muse agent commit changes files outside {pv.MUSE_DIR}"
            )
        if c["kind"] != "muse" and in_muse and not is_muse_merge(c, agent_tips):
            problems.append(
                f"{tag}: a commit not by a Muse agent changes {pv.MUSE_DIR}: {in_muse[:5]}"
            )
    models = check_transcripts(problems)
    rounds = check_rounds(refs, problems)
    secrets = check_secrets(problems)

    print(f"refs: {' '.join(refs)}; agent era starts at tag {pv.AGENT_ERA_TAG}")
    print(f"{'era':<10} {'co-author':<8} {'type':<7} count")
    for (era_label, kind, typ), n in sorted(rows.items()):
        print(f"{era_label:<10} {kind:<8} {typ:<7} {n}")
    print(f"Muse agent commits by co-author: {dict(muse_authors) or 'none yet'}")
    print(f"finished rounds, files frozen: {', '.join(rounds) or 'none yet'}")
    print(f"transcript model calls by model: {dict(models) or 'none yet'}")
    print(f"secret scan: {secrets}")
    print("\n".join(problems) if problems else "OK: no mixed authorship found")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
