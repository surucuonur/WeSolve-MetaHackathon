"""Copy the Muse Spark sessions' transcripts into muse-spark/transcripts/.

Claude Code keeps each session as JSON lines under <config dir>/projects/.
Every session in the Muse Spark config folder is a Muse Spark session, so all
of them are exported, unless --session picks one. This keeps only the
conversation: user and assistant entries, with each assistant turn's model,
message id and request id, which show which model wrote it. It drops the
api-request* entries, which carry Claude Code's own system prompt and tool
definitions (not ours to publish), and attachments. It refuses to write if an
API key value appears anywhere, and writes a summary next to each transcript.

  .venv/bin/python tools/export_transcript.py [--session ID] [--config-dir ~/.claude-muse]

Run it from the Muse Spark worktree, then commit the files there with the
Muse Spark co-author (tools/muse-agent/start.sh shows the command).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as pv  # noqa: E402

KEEP = ("type", "uuid", "parentUuid", "timestamp", "sessionId", "version", "gitBranch")
KEEP_MESSAGE = ("role", "model", "id", "content", "stop_reason", "usage")


def find(config_dir, session):
    files = sorted(
        (config_dir / "projects").glob(f"*/{session or '*'}.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise SystemExit(f"no session transcript under {config_dir / 'projects'}")
    return files


def convert(src, values):
    """The kept entries of *src* as JSON lines, and their summary."""
    kept, models, turns, times, versions = [], Counter(), Counter(), [], set()
    for line in src.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("type") not in ("user", "assistant"):
            continue
        row = {k: entry[k] for k in KEEP if k in entry}
        if entry.get("requestId"):
            row["requestId"] = entry["requestId"]
        msg = entry.get("message") or {}
        row["message"] = {k: msg[k] for k in KEEP_MESSAGE if k in msg}
        kept.append(row)
        turns[entry["type"]] += 1
        if entry["type"] == "assistant":
            models[msg.get("model") or ""] += 1
        times.append(entry.get("timestamp"))
        versions.add(entry.get("version"))

    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept)
    if pv.contains_secret(text.encode(), values):
        raise SystemExit(f"refusing to export {src.stem}: it contains an API key value")
    summary = {
        "session": src.stem,
        "first": min(filter(None, times), default=None),
        "last": max(filter(None, times), default=None),
        "claude_code_versions": sorted(filter(None, versions)),
        "turns": dict(turns),
        "assistant_turns_by_model": dict(models),
        "dropped": "api-request*, attachment and other non-conversation entries",
    }
    return text, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", help="session id (default: every session in the folder)")
    ap.add_argument("--config-dir", type=Path, default=pv.muse_config_dir())
    ap.add_argument("--out", type=Path, default=Path(pv.MUSE_DIR) / "transcripts")
    args = ap.parse_args()
    files = find(args.config_dir.expanduser(), args.session)

    values = pv.secrets()
    if not values:
        raise SystemExit("can't check for API keys: none found in the .env files")
    converted = [convert(src, values) for src in files]  # all checked before any is written
    args.out.mkdir(parents=True, exist_ok=True)
    for text, summary in converted:
        session = summary["session"]
        (args.out / f"{session}.jsonl").write_text(text)
        (args.out / f"{session}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        if any(m.startswith("claude") for m in summary["assistant_turns_by_model"]):
            print(f"warning: some turns of {session} were written by a Claude model", file=sys.stderr)


if __name__ == "__main__":
    main()
