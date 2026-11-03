"""Shared rules for keeping Muse Spark's work apart from Claude's (PROVENANCE.md).

Used by the git hooks in tools/githooks/, tools/provenance_check.py and
tools/export_transcript.py.
"""

import json
import os
import re
import subprocess
from pathlib import Path

MUSE_BRANCH = "muse-agent"
MUSE_DIR = "muse-spark/"
AGENT_ERA_TAG = "agent-era"
AUTHOR = ("surucuonur", "25654558+surucuonur@users.noreply.github.com")
MUSE_COAUTHOR = (
    "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
)
SECRET_NAMES = ("MODEL_API_KEY", "HUGGINGFACE_API_KEY")
# The key in the Muse Spark config folder's settings.json (ADR 0005).
MUSE_SECRET_NAMES = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
COAUTHOR = re.compile(r"^co-authored-by:\s*(.+)$", re.I | re.M)


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=cwd
    ).stdout


def coauthors(message):
    return [m.strip() for m in COAUTHOR.findall(message)]


def is_muse(name):
    return name.lower().startswith("muse spark")


def is_claude(name):
    low = name.lower()
    return "claude" in low or "anthropic.com" in low


def kind(message):
    """'muse', 'claude', 'both' or 'human', from a commit message's co-authors."""
    names = coauthors(message)
    muse = any(is_muse(n) for n in names)
    claude = any(is_claude(n) for n in names)
    return (
        "both"
        if muse and claude
        else "muse" if muse else "claude" if claude else "human"
    )


def main_checkout():
    """The main checkout's root, also from inside a linked worktree."""
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    )
    return common.parent


def muse_config_dir():
    """The Muse Spark session's Claude Code config folder (tools/muse-agent/start.sh)."""
    return Path(os.environ.get("MUSE_CONFIG_DIR") or "~/.claude-muse").expanduser()


def secrets():
    """The API key values from the main checkout's .env and from the env block of
    the Muse Spark config folder's settings.json; [] where neither can be read.

    The values are only compared, never printed.
    """
    values = []
    try:
        text = (main_checkout() / ".env").read_text()
    except OSError:
        text = ""
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip().removeprefix("export ").strip() in SECRET_NAMES:
            values.append(value.strip().strip("'\""))
    try:
        data = json.loads((muse_config_dir() / "settings.json").read_text())
    except (OSError, ValueError):
        data = {}
    env = data.get("env") if isinstance(data, dict) else None
    if isinstance(env, dict):
        values += [str(env[n]) for n in MUSE_SECRET_NAMES if n in env]
    return list(dict.fromkeys(v for v in values if len(v) >= 12))


def contains_secret(data, values):
    """Whether *data* (bytes) contains any of the secret *values*."""
    return any(v.encode() in data for v in values)
