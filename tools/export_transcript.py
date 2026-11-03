"""Copy the Muse Spark sessions' transcripts into muse-spark/transcripts/.

Muse Code logs each session under <Muse home>/.local/share/muse/sessions/: the
lead agent's in <session>/session.jsonl, and every other agent's below it in
subagent/<id>/session.jsonl. Those are the helper agents the lead starts and
Muse Code's own reminder agents, which watch the lead and may nudge it.
tools/muse-agent/start.sh gives muse a home folder of its own, so every session
there is a Muse Spark session. All are exported, unless --session picks one
(by id, or by folder for a session logged elsewhere).

Each agent's log is read through `muse export`, which reads only local files.
One transcript per session keeps every agent's conversation: the prompt, the
model's messages and reasoning summaries, its tool calls and their results, how
each tool call was approved, and each model call's model and token usage, which
show which model wrote it. It drops Muse Code's own instructions and context
(its system prompt, the skill catalog, the reminder agents' prompts), encrypted
reasoning and runtime bookkeeping; the summary counts what was dropped. It
refuses to write if an API key value appears anywhere.

  .venv/bin/python tools/export_transcript.py [--session ID-or-folder]

Run it from the Muse Spark worktree, then commit the files there with the
Muse Spark co-author (tools/muse-agent/start.sh shows the command).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as pv  # noqa: E402

# The run events kept, and the fields kept from each.
KEEP = {
    "started": ("prompt",),
    "assistant_message_committed": ("text", "phase"),
    "reasoning_summary_committed": ("text",),
    "assistant_tool_calls_committed": ("tool_calls",),
    "tool_result_batch_committed": ("results",),
    "model_completed": ("model", "usage", "finish_reason", "duration_ms"),
    "terminal": ("terminal", "reason"),
}
KEEP_CALL = ("call_id", "name", "args")
KEEP_RESULT = ("tool_call_id", "text")


def find(store, session):
    if session and Path(session).is_dir():
        return [Path(session)]
    logs = sorted(store.glob("[0-9]*/*/*/*/session.jsonl"))
    folders = [log.parent for log in logs if session in (None, log.parent.name)]
    if not folders:
        raise SystemExit(f"no Muse Code session {session or ''} under {store}")
    return folders


def export(log):
    """The `muse export` document of one agent's session log."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "export.json"
        subprocess.run(
            ["muse", "export", "--session", str(log), "--out", str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(out.read_text())


def records(doc):
    """(step, time, record type, payload) for each runtime record of an export."""
    for e in doc["events"]:
        envelope = e.get("envelope") or {}
        payload = envelope.get("payload")
        if e.get("kind") == "record" and isinstance(payload, dict):
            step = (e.get("derived") or {}).get("step_id")
            yield step, e.get("recorded_at"), envelope.get("payload_type"), payload


def labels(docs):
    """Each non-lead agent's label, by session id, from the spawn and link records."""
    spawned, bound, reminders = {}, {}, {}
    for doc in docs.values():
        for _, _, _, p in records(doc):
            rec, event = p.get("record") or {}, p.get("event") or {}
            if rec.get("kind") == "spawn_accepted":
                spawned[rec["subagent_id"]] = rec.get("agent_path") or rec.get("role")
            elif rec.get("kind") == "child_session_bound":
                bound[rec["child_session_id"]] = rec["subagent_id"]
            elif event.get("kind") == "memory_reminder_child_session_linked":
                reminders[event["child_session_id"]] = event.get("reminder_agent_id")
    out = {sid: f"helper {spawned.get(agent, agent)}" for sid, agent in bound.items()}
    out.update({sid: f"reminder {name}" for sid, name in reminders.items()})
    return out


def when(micros):
    return datetime.fromtimestamp(micros / 1e6, timezone.utc).isoformat() if micros else None


def convert(folder, values):
    """The transcript of the session in *folder* as JSON lines, and its summary."""
    logs = [folder / "session.jsonl"] + sorted((folder / "subagent").rglob("session.jsonl"))
    docs = {log.parent.name: export(log) for log in logs}
    names = labels(docs)
    rows, agents, models, tokens, approvals, dropped = [], [], Counter(), Counter(), Counter(), Counter()
    for sid, doc in docs.items():
        agent = "lead" if sid == folder.name else names.get(sid, "unlinked")
        calls = 0
        for step, at, ptype, p in records(doc):
            event = p.get("event") or {}
            kind = event.get("kind")
            row = {"agent": agent, "session": sid, "step": step, "at": when(at)}
            if p.get("kind") == "run" and kind in KEEP:
                if kind == "started" and agent.startswith("reminder"):
                    dropped["reminder prompt"] += 1
                    continue
                row["event"] = kind
                row.update({k: event[k] for k in KEEP[kind] if k in event})
                if kind == "assistant_tool_calls_committed":
                    row["tool_calls"] = [{k: c.get(k) for k in KEEP_CALL} for c in event["tool_calls"]]
                elif kind == "tool_result_batch_committed":
                    row["results"] = [{k: r.get(k) for k in KEEP_RESULT} for r in event["results"]]
                elif kind == "model_completed":
                    calls += 1
                    models[event.get("model") or ""] += 1
                    tokens.update({k: v for k, v in (event.get("usage") or {}).items() if v})
            elif kind == "side_effect_intent" and str(event.get("operation")).startswith("tool:"):
                row.update(event="approval", operation=event["operation"], decision=event.get("policy_decision"))
                approvals[event.get("policy_decision")] += 1
            else:
                dropped[f"{ptype}.{kind}" if kind else ptype] += 1
                continue
            rows.append(row)
        agents.append({"session": sid, "agent": agent, "model_calls": calls})

    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if pv.contains_secret(text.encode(), values):
        raise SystemExit(f"refusing to export {folder.name}: it contains an API key value")
    times = [r["at"] for r in rows if r["at"]]
    lead = docs[folder.name]
    summary = {
        "session": folder.name,
        "first": min(times, default=None),
        "last": max(times, default=None),
        "recorded_by": (lead.get("session_build") or {}).get("display"),
        "agents": agents,
        "model_calls_by_model": dict(models),
        "tokens": dict(tokens),
        "tool_calls_by_approval": dict(approvals),
        "dropped": "Muse Code's instructions and context blocks, reminder prompts, "
        "encrypted reasoning and runtime bookkeeping",
        "dropped_records": dict(sorted(dropped.items())),
    }
    return text, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", help="session id or folder (default: every session in the store)")
    ap.add_argument("--store", type=Path, default=pv.muse_sessions_dir())
    ap.add_argument("--out", type=Path, default=Path(pv.MUSE_DIR) / "transcripts")
    args = ap.parse_args()
    folders = find(args.store, args.session)

    values = pv.secrets()
    if not values:
        raise SystemExit("can't check for API keys: none found in the .env files")
    converted = [convert(f, values) for f in folders]  # all checked before any is written
    args.out.mkdir(parents=True, exist_ok=True)
    for text, summary in converted:
        session = summary["session"]
        (args.out / f"{session}.jsonl").write_text(text)
        (args.out / f"{session}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        if any(not pv.is_meta_model(m) for m in summary["model_calls_by_model"]):
            print(f"warning: some model calls of {session} were not by a Meta model", file=sys.stderr)


if __name__ == "__main__":
    main()
