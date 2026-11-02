"""Call Muse Spark through the Meta Model API and keep a verbatim record of it.

Each call gets a folder muse-spark/calls/NNNN-<slug>/ with:

  request.json   what was sent: model and messages (never the key or headers)
  response.json  the API's response, verbatim (response.model_dump())
  meta.json      time, response id, token counts, cost, and the SHA-256 of
                 the answer text

The key is read from .env in this process only and is never printed or saved.
Files that carry Muse Spark's content start with lines marked [harness],
then MARKER; everything after MARKER is Muse Spark's text, byte for byte.
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALLS = ROOT / "muse-spark" / "calls"
BRIEF = ROOT / "claude" / "muse-spark-brief.md"
BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3-contributor"
# Dollars per million tokens, from dev.meta.ai/docs/pricing-rate-limits (2026-11-02)
PRICE = {"input": 0.10, "cached_input": 0.002, "output": 0.20}
MARKER = "--- Muse Spark output below, unmodified ---"
FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)


def _client():
    from dotenv import dotenv_values
    from openai import OpenAI

    key = dotenv_values(ROOT / ".env").get("MODEL_API_KEY")
    if not key:
        raise SystemExit("MODEL_API_KEY is not set in .env")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=900, max_retries=2)


def _next_dir(slug):
    CALLS.mkdir(parents=True, exist_ok=True)
    n = 1 + max(
        (int(p.name[:4]) for p in CALLS.glob("[0-9][0-9][0-9][0-9]-*")), default=0
    )
    return CALLS / f"{n:04d}-{slug}"


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cost(usage):
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    fresh = usage["input_tokens"] - cached
    return (
        fresh * PRICE["input"]
        + cached * PRICE["cached_input"]
        + usage["output_tokens"] * PRICE["output"]
    ) / 1e6


def call(slug, messages):
    """Send *messages* ([{"role", "content"}]) and return (answer text, call dir)."""
    out = _next_dir(slug)
    out.mkdir()
    request = {"model": MODEL, "input": messages}
    (out / "request.json").write_text(json.dumps(request, indent=1) + "\n")
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    response = _client().responses.create(**request)
    seconds = time.monotonic() - t0
    raw = response.model_dump(mode="json")
    (out / "response.json").write_text(json.dumps(raw, indent=1) + "\n")
    text = response.output_text
    usage = raw.get("usage") or {}
    meta = {
        "call": out.name,
        "model": MODEL,
        "response_id": raw.get("id"),
        "started_utc": started.isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "usage": usage,
        "cost_usd": round(cost(usage), 6) if usage else None,
        "answer_sha256": sha256(text),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(
        f"{out.name}: {raw.get('id')}, {usage.get('input_tokens')} in / "
        f"{usage.get('output_tokens')} out, ${meta['cost_usd']}, {seconds:.0f} s",
        flush=True,
    )
    return text, out


def answer_text(response_json):
    """The answer text of a saved response, as the SDK's output_text builds it."""
    raw = json.loads(Path(response_json).read_text())
    parts = []
    for item in raw.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text") or "")
    return "".join(parts)


def blocks(text, lang):
    """The fenced code blocks of *text* whose language is *lang*, in order."""
    return [body for tag, body in FENCE.findall(text) if tag.lower() == lang]


def write_verbatim(path, body, call_dir, comment):
    """Write Muse Spark's *body* under a [harness] header in *comment* syntax."""
    rel = call_dir.relative_to(ROOT)
    meta = json.loads((call_dir / "meta.json").read_text())
    header = [
        f"[harness] Written by Meta Muse Spark ({MODEL}) via the Meta Model API.",
        f"[harness] Call {rel}, response {meta['response_id']}.",
        f"[harness] {MARKER}",
    ]
    lines = [f"{comment} {h}" for h in header]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" + body)


def read_verbatim(path):
    """The part of a write_verbatim file after the marker line."""
    text = path.read_text()
    head, sep, body = text.partition(MARKER + "\n")
    if not sep:
        raise ValueError(f"{path}: no '{MARKER}' line")
    return body
