"""Copy a Muse Spark agent session's transcript into muse-spark/transcripts/.

Claude Code keeps each session as JSON lines under <config dir>/projects/.
This keeps only the conversation: user and assistant entries, with each
assistant turn's model, message id and request id, which show which model
wrote it. It drops the api-request* entries, which carry Claude Code's own
system prompt and tool definitions (not ours to publish), and attachments.
It refuses to write if an API key value from .env appears anywhere, and
writes a summary next to the transcript.

  .venv/bin/python tools/export_transcript.py [--session ID] [--config-dir ~/.claude-muse]

Run it from the Muse Spark worktree, then commit the two files there.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as pv  # noqa: E402

KEEP = ("type", "uuid", "parentUuid", "timestamp", "sessionId", "version", "gitBranch")
KEEP_MESSAGE = ("role", "model", "id", "content", "stop_reason", "usage")


def find(config_dir, session):
    files = sorted((config_dir / "projects").glob(f"*/{session or '*'}.jsonl"))
    if not files:
        raise SystemExit(f"no session transcript under {config_dir / 'projects'}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", help="session id (default: the newest)")
    ap.add_argument("--config-dir", type=Path, default=Path.home() / ".claude-muse")
    ap.add_argument("--out", type=Path, default=Path(pv.MUSE_DIR) / "transcripts")
    args = ap.parse_args()
    src = find(args.config_dir.expanduser(), args.session)

    kept, models, turns, times, versions = [], Counter(), Counter(), [], set()
    for line in src.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("type") not in ("user", "assistant"):
            continue
        out = {k: entry[k] for k in KEEP if k in entry}
        if entry.get("requestId"):
            out["requestId"] = entry["requestId"]
        msg = entry.get("message") or {}
        out["message"] = {k: msg[k] for k in KEEP_MESSAGE if k in msg}
        kept.append(out)
        turns[entry["type"]] += 1
        if entry["type"] == "assistant":
            models[msg.get("model") or ""] += 1
        times.append(entry.get("timestamp"))
        versions.add(entry.get("version"))

    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept)
    values = pv.secrets()
    hashes = pv.parse_hashes(os.environ.get("MUSE_SECRET_HASHES"))
    if not values and not hashes:
        raise SystemExit("can't check for API keys: run this outside the agent session")
    if pv.contains_secret(text.encode(), values, hashes):
        raise SystemExit("refusing to export: the transcript contains an API key value")
    session = src.stem
    summary = {
        "session": session,
        "first": min(filter(None, times), default=None),
        "last": max(filter(None, times), default=None),
        "claude_code_versions": sorted(filter(None, versions)),
        "turns": dict(turns),
        "assistant_turns_by_model": dict(models),
        "dropped": "api-request*, attachment and other non-conversation entries",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{session}.jsonl").write_text(text)
    (args.out / f"{session}.summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))
    if any(m.startswith("claude") for m in models):
        print("warning: some turns were written by a Claude model", file=sys.stderr)


if __name__ == "__main__":
    main()
